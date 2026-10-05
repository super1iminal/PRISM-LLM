"""The refinement loop: LLM writes symbolic rules, PRISM checks best/worst case, feedback targets the gap.

Each round, for the best rule set so far (keep-best, as in the legacy loop):
  * worst case meets every threshold   -> done (every completion of the partial policy is safe)
  * no completion can meet them all    -> refine: existing rules are bad; the LLM rewrites the list,
                                          guided by per-rule blame (probability mass)
  * some completion can                -> extend: rules are fine but incomplete; the LLM adds rules
                                          (appended, so existing decisions are unchanged) for the
                                          uncovered states carrying the most mass
"Can any completion meet them" is PRISM's multi-objective query (`planner.branch: joint`) or, as in
the runs before Phase A, each requirement's best case on its own (`per_requirement`). The retry
policy (`planner.retry`, see core/retry.py) decides when to drop feedback and start from the initial
prompt instead. With `planner.feedback: table` (ablation S1) there is no branch: every round after a
failure shows the results table and asks for a complete new rule list. Semantics: docs/semantics.md.
"""
import zlib
from dataclasses import dataclass
from time import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, create_model

from config import Config
from core.analysis import MassAnalyzer
from core.domain import Domain, Instance, Requirement, failing
from core.llm import LLMCall, OllamaLLM
from core.prism import PrismError, PrismRunner
from core.retry import RetryPolicy
from core.rules import RuleError, SymbolicPolicy
from core.verifier import PolicyVerifier, Verification


def rule_schema(actions: List[str], max_rules: int = 64, max_condition_chars: int = 200) -> type[BaseModel]:
    """JSON schema for the LLM's answer. The length bounds are enforced by constrained decoding,
    which stops degenerate repetition loops from running into the token limit."""
    rule = create_model(
        "Rule",
        condition=(str, Field(description="Boolean condition over the state variables",
                              max_length=max_condition_chars)),
        action=(Literal.__getitem__(tuple(actions)), Field(description="Action to take")),
    )
    return create_model("RuleList", rules=(List[rule], Field(..., max_length=max_rules)))


def _score(v: Verification, reqs: List[Requirement]) -> Tuple:
    """Lower is better: worst-case failures, best-case failures, then total shortfalls."""
    return (len(failing(reqs, v.worst)),
            len(failing(reqs, v.best)),
            sum(r.shortfall(v.worst[r.name]) for r in reqs),
            sum(r.shortfall(v.best[r.name]) for r in reqs))


@dataclass
class _Episode:
    """What stays fixed while one instance is solved."""
    instance: Instance
    verifier: PolicyVerifier
    analyzer: MassAnalyzer
    schema: type
    context: Dict[str, Any]           # prompt context shared by every template
    log: Callable[[str], None]

    @property
    def requirements(self) -> List[Requirement]:
        return self.verifier.spec.requirements


@dataclass
class _Kept:
    """The best rule set so far (keep-best), with its verification and score."""
    policy: SymbolicPolicy
    v: Verification
    score: Tuple
    joint: Optional[bool] = None      # can one completion meet every threshold? (None: undecided)
    joint_queried: bool = False


def _record(attempt: int, mode: str, prompt: str, candidate: SymbolicPolicy, v: Verification,
            reqs: List[Requirement], *, improved: bool, gain: float, errors: List[str],
            calls: List[LLMCall]) -> Dict[str, Any]:
    """One entry of the result's `iterations`. The feedback step fills in the joint fields."""
    return {
        "iteration": attempt,
        "mode": mode,
        "rules": candidate.to_dicts(),
        "num_rules": len(candidate.rules),
        "best": dict(v.best),
        "worst": dict(v.worst),
        "reachable_situations": v.reachable_situations,
        "uncovered_situations": v.uncovered_situations,
        "best_case_success": not failing(reqs, v.best),
        "worst_case_success": not failing(reqs, v.worst),
        "improved": improved,
        "shortfall_gain": gain,
        "invalid_answers": errors,
        "llm_calls": len(calls),
        "llm_time": sum(c.seconds for c in calls),
        "llm_server_time": sum(c.server_seconds for c in calls),
        "llm_output_tokens": sum(c.output_tokens for c in calls),
        "llm_prompt_tokens": sum(c.prompt_tokens for c in calls),
        "prism_time": v.seconds,
        "joint_time": 0.0,
        "kept_joint_feasible": None,        # joint best case of the kept policy (if queried)
        "branch_disagreement": False,       # per-requirement best passes but joint fails
        "prompt": prompt,
        "raw_outputs": [c.raw_output for c in calls],
    }


class SymbolicPlanner:
    def __init__(self, domain: Domain, llm: OllamaLLM, config: Optional[Config] = None,
                 runner: Optional[PrismRunner] = None):
        self.domain = domain
        self.llm = llm
        self.config = config or Config()
        self.runner = runner or PrismRunner(config=self.config.prism)
        self.retry = RetryPolicy.parse(self.config.planner.retry)

    # ---------------------------------------------------------------- prompting

    def _prompt_context(self, instance: Instance, spec) -> Dict[str, Any]:
        prompt = self.config.prompt
        return {
            "description": self.domain.description(instance),
            "visual": self.domain.visual(instance),
            "examples": self.domain.examples(instance) if prompt.examples else "",
            "catch_all_instruction": prompt.catch_all_instruction,
            "variables": spec.variables,
            "actions": spec.actions,
            "requirements": spec.requirements,
        }

    def _results_context(self, policy: SymbolicPolicy, v: Verification, reqs: List[Requirement]) -> Dict[str, Any]:
        rows = []
        for r in reqs:
            best_ok, worst_ok = r.satisfied(v.best[r.name]), r.satisfied(v.worst[r.name])
            status = "ok" if worst_ok else ("fails in worst case" if best_ok else "FAILS even in best case")
            rows.append({"name": r.name, "best": v.best[r.name], "worst": v.worst[r.name],
                         "bound": ">=" if r.maximize else "<=", "threshold": r.threshold, "status": status})
        return {"rules_listing": policy.listing(), "num_rules": len(policy.rules), "results": rows,
                "reachable": v.reachable_situations, "uncovered": v.uncovered_situations}

    def _render(self, ep: _Episode, template: str, **extra) -> str:
        return self.domain.render(template, ep.instance, **ep.context, **extra)

    def _ask(self, prompt: str, schema, spec, log) -> Tuple[Optional[SymbolicPolicy], List[str]]:
        """Query the LLM, re-asking with the errors if the rules do not parse."""
        errors: List[str] = []
        current = prompt
        for _ in range(1 + self.config.planner.max_fixups):
            raw = self.llm.invoke_raw(current, schema)
            try:
                parsed = schema.model_validate_json(raw)
                policy = SymbolicPolicy.from_raw(spec.variables, list(spec.actions),
                                                 [(r.condition, r.action) for r in parsed.rules])
                return policy, errors
            except (ValidationError, RuleError) as e:
                message = str(e)
                errors.append(message)
                log(f"Invalid LLM answer: {message}")
                current = self.domain.render("invalid.md.j2", prompt=prompt, errors=message)
        return None, errors

    # ---------------------------------------------------------------- main loop

    def solve(self, instance: Instance, logger) -> Dict[str, Any]:
        ep = self._episode(instance, logger.info)
        reqs, max_rounds = ep.requirements, self.config.planner.max_rounds

        opt_start = time()
        optimum, _, _ = ep.verifier.optimum()
        optimum_seconds = time() - opt_start
        ep.log(f"Unconstrained optimum per requirement: {optimum}")

        iterations: List[Dict[str, Any]] = []
        kept: Optional[_Kept] = None
        mode, prompt, stall = "initial", self._render(ep, "initial.md.j2"), 0
        for attempt in range(1, max_rounds + 1):
            iter_start = time()
            ep.log(f"=== Attempt {attempt}/{max_rounds} ({mode}) ===\n{prompt}")
            candidate, v, errors, calls = self._round(ep, mode, prompt, kept)
            score = _score(v, reqs)
            previous_shortfall = kept.score[2] if kept else None
            improved = kept is None or score < kept.score
            if improved:
                kept = _Kept(candidate, v, score)
            gain = previous_shortfall - kept.score[2] if previous_shortfall is not None else float("inf")
            ep.log("Results: " + ", ".join(f"{r.name}: best={v.best[r.name]:.4f} worst={v.worst[r.name]:.4f}"
                                           for r in reqs))
            ep.log(f"Score {score} ({'new best' if improved else 'no improvement, keeping best'})")
            record = _record(attempt, mode, prompt, candidate, v, reqs,
                             improved=improved, gain=gain, errors=errors, calls=calls)
            iterations.append(record)

            done = not failing(reqs, kept.v.worst)
            if done:
                ep.log(f"Success after {attempt} attempts")
            elif attempt < max_rounds:
                stall = 0 if improved else stall + 1
                if self.retry.restart(attempt + 1, stall, gain):
                    mode, prompt, stall = "initial", self._render(ep, "initial.md.j2"), 0
                else:
                    mode, prompt = self._feedback(ep, kept, record)
            record["iteration_time"] = time() - iter_start
            if done:
                break

        return {
            "success": not failing(reqs, kept.v.worst),
            "best_case_success": not failing(reqs, kept.v.best),
            "final_best": dict(kept.v.best),
            "final_worst": dict(kept.v.worst),
            "final_rules": kept.policy.to_dicts(),
            "optimum": optimum,
            "optimum_time": optimum_seconds,
            "iterations": iterations,
        }

    def _episode(self, instance: Instance, log: Callable[[str], None]) -> _Episode:
        cfg = self.config
        verifier = PolicyVerifier(self.domain, instance, self.runner, cfg.rules.max_enumeration)
        analyzer = MassAnalyzer(verifier, cfg.feedback.horizon_for(self.domain, instance), cfg.feedback.top_k,
                                cfg.feedback.states_per_rule, method=cfg.feedback.blame,
                                seed=zlib.crc32(f"{cfg.llm.seed}/{instance.id}".encode()))
        schema = rule_schema(list(verifier.spec.actions), cfg.planner.max_rules, cfg.planner.max_condition_chars)
        return _Episode(instance, verifier, analyzer, schema, self._prompt_context(instance, verifier.spec), log)

    def _round(self, ep: _Episode, mode: str, prompt: str,
               kept: Optional[_Kept]) -> Tuple[SymbolicPolicy, Verification, List[str], List[LLMCall]]:
        """Ask for rules and verify the candidate: the new rules, or the kept ones followed by them when
        extending. Returns the candidate, its verification, the invalid answers and the LLM calls."""
        calls_before = len(self.llm.usage().calls)
        new_rules, errors = self._ask(prompt, ep.schema, ep.verifier.spec, ep.log)
        calls = self.llm.usage().calls[calls_before:]
        if new_rules is None:
            new_rules = ep.verifier.empty_policy()
        candidate = kept.policy.extended(new_rules) if mode == "extend" else new_rules
        ep.log(f"Candidate policy ({len(candidate.rules)} rules):\n{candidate.listing()}")
        try:
            return candidate, ep.verifier.verify(candidate), errors, calls
        except PrismError as e:
            # Rare: PRISM cannot solve this candidate's induced model even with the fallback methods.
            # Count the round as producing nothing (the empty policy) instead of losing the instance.
            reason = str(e).splitlines()[0]
            ep.log(f"Verification failed ({reason}); scoring the round as an empty policy")
            errors.append(f"verification failed: {reason}")
            candidate = kept.policy if mode == "extend" and kept else ep.verifier.empty_policy()
            return candidate, ep.verifier.verify(candidate), errors, calls

    # ---------------------------------------------------------------- feedback

    def _feedback(self, ep: _Episode, kept: _Kept, record: Dict[str, Any]) -> Tuple[str, str]:
        """Mode and prompt of the next round, from the kept rule set's results."""
        reqs = ep.requirements
        results = self._results_context(kept.policy, kept.v, reqs)
        if self.config.planner.feedback == "table":
            return "table", self._render(ep, "table.md.j2", **results)
        failing_best, failing_worst = failing(reqs, kept.v.best), failing(reqs, kept.v.worst)
        joint_conflict = (not failing_best and self.config.planner.branch == "joint"
                          and self._joint_conflict(ep, kept, record))
        if failing_best or joint_conflict:
            return "refine", self._refine_prompt(ep, kept, failing_best or failing_worst, joint_conflict, results)
        return "extend", self._extend_prompt(ep, kept, failing_worst, results)

    def _joint_conflict(self, ep: _Episode, kept: _Kept, record: Dict[str, Any]) -> bool:
        """Whether no single completion of the kept rules meets every threshold, although each requirement
        passes its best case. Asked once per kept rule set; undecided (None) counts as no conflict."""
        if not kept.joint_queried:
            start = time()
            kept.joint, kept.joint_queried = ep.verifier.jointly_feasible(kept.policy), True
            record["joint_time"] = time() - start
        conflict = kept.joint is False
        record["kept_joint_feasible"], record["branch_disagreement"] = kept.joint, conflict
        if conflict:
            ep.log("Each requirement passes its best case, but no single completion passes all: refine")
        return conflict

    def _refine_prompt(self, ep: _Episode, kept: _Kept, blamed: List[Requirement], joint_conflict: bool,
                       results: Dict[str, Any]) -> str:
        """Rewrite the rules, shown how much of the lost probability each rule's states carry."""
        kind = self.config.feedback.blame
        rules = kept.policy.rules
        blame = [{
            "rule": b.rule, "mass": b.mass,
            "text": f"{rules[b.rule].condition} (action {rules[b.rule].action})" if b.rule is not None else "",
            "states": [self.domain.format_state(h.valuation) for h in b.hotspots],
        } for b in (ep.analyzer.rule_blame(kept.v, blamed) if kind != "none" else [])]
        return self._render(ep, "refine.md.j2", **results, failing=[r.name for r in blamed], blame=blame,
                            with_cost=any(r.reward is not None for r in blamed), joint_conflict=joint_conflict,
                            show_blame=kind != "none", blame_kind=kind)

    def _extend_prompt(self, ep: _Episode, kept: _Kept, failing_worst: List[Requirement],
                       results: Dict[str, Any]) -> str:
        """Add rules, shown the uncovered states where the worst case loses the most probability."""
        kind = self.config.feedback.blame
        hotspots = ep.analyzer.uncovered_hotspots(kept.v, failing_worst) if kind != "none" else []
        return self._render(ep, "extend.md.j2", **results, failing=[r.name for r in failing_worst],
                            hotspots=[{"mass": h.mass, "state": self.domain.format_state(h.valuation)}
                                      for h in hotspots],
                            with_cost=any(r.reward is not None for r in failing_worst),
                            show_blame=kind != "none", blame_kind=kind)
