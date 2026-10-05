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
from time import time
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, create_model

from config import Config
from core.analysis import MassAnalyzer
from core.backends import LLMBackend
from core.domain import Domain, Instance, Requirement
from core.prism import PrismError, PrismRunner
from core.retry import RetryPolicy
from core.rules import RuleError, SymbolicPolicy
from core.scheduler import drive
from core.tasks import LLMError, LLMResult, LLMTask, TaskFactory
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


AskSteps = Generator[LLMTask, LLMResult, Tuple[Optional[SymbolicPolicy], List[str], List[LLMResult]]]
SolveSteps = Generator[LLMTask, LLMResult, Dict[str, Any]]


def _score(v: Verification, reqs: List[Requirement]) -> Tuple:
    """Lower is better: worst-case failures, best-case failures, then total shortfalls."""
    return (sum(not r.satisfied(v.worst[r.name]) for r in reqs),
            sum(not r.satisfied(v.best[r.name]) for r in reqs),
            sum(r.shortfall(v.worst[r.name]) for r in reqs),
            sum(r.shortfall(v.best[r.name]) for r in reqs))


class SymbolicPlanner:
    def __init__(self, domain: Domain, backend: Optional[LLMBackend] = None, config: Optional[Config] = None,
                 runner: Optional[PrismRunner] = None):
        self.domain = domain
        self.backend = backend   # used by solve(); schedulers drive solve_steps() with their own
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

    def _ask(self, prompt: str, schema, spec, log, tasks: TaskFactory, round_no: int,
             mode: str) -> AskSteps:
        """Query the LLM, re-asking with the errors if the rules do not parse.

        A generator: yields each `LLMTask` and is sent its `LLMResult`. Returns the policy (None if
        every answer was invalid), the parse errors and the results of every call.
        """
        errors: List[str] = []
        results: List[LLMResult] = []
        current = prompt
        json_schema = schema.model_json_schema()
        for fixup in range(1 + self.config.planner.max_fixups):
            result = yield tasks.make(current, json_schema, round=round_no, mode=mode, fixup=fixup)
            results.append(result)
            if result.error is not None:
                raise LLMError(result.error)
            try:
                parsed = schema.model_validate_json(result.text)
                policy = SymbolicPolicy.from_raw(spec.variables, list(spec.actions),
                                                 [(r.condition, r.action) for r in parsed.rules])
                return policy, errors, results
            except (ValidationError, RuleError) as e:
                message = str(e)
                errors.append(message)
                log(f"Invalid LLM answer: {message}")
                current = self.domain.render("invalid.md.j2", prompt=prompt, errors=message)
        return None, errors, results

    # ---------------------------------------------------------------- main loop

    def solve(self, instance: Instance, logger) -> Dict[str, Any]:
        """Run the refinement loop for one instance, answering each task with `self.backend` in turn."""
        return drive(self.solve_steps(instance, logger), self.backend.execute)

    def solve_steps(self, instance: Instance, logger) -> SolveSteps:
        """The refinement loop for one instance, as a generator: yields each `LLMTask`, is sent its
        `LLMResult`, and returns the result dict. Schedulers (core/scheduler.py) drive it."""
        log = logger.info
        cfg = self.config
        tasks = TaskFactory(cfg.llm, instance.id)
        verifier = PolicyVerifier(self.domain, instance, self.runner, cfg.rules.max_enumeration)
        analyzer = MassAnalyzer(verifier, cfg.feedback.horizon_for(self.domain, instance), cfg.feedback.top_k,
                                cfg.feedback.states_per_rule, method=cfg.feedback.blame,
                                seed=zlib.crc32(f"{cfg.llm.seed}/{instance.id}".encode()))
        show_blame = cfg.feedback.blame != "none"
        spec = verifier.spec
        reqs = spec.requirements
        schema = rule_schema(list(spec.actions), cfg.planner.max_rules, cfg.planner.max_condition_chars)
        base_ctx = self._prompt_context(instance, spec)
        max_rounds = cfg.planner.max_rounds

        opt_start = time()
        optimum, _, _ = verifier.optimum()
        optimum_seconds = time() - opt_start
        log(f"Unconstrained optimum per requirement: {optimum}")

        iterations: List[Dict[str, Any]] = []
        best: Optional[Tuple[SymbolicPolicy, Verification, Tuple]] = None
        kept_joint: Optional[bool] = None   # joint feasibility of the kept policy, once queried
        joint_queried = False
        mode, prompt, stall = "initial", self.domain.render("initial.md.j2", instance, **base_ctx), 0

        for attempt in range(1, max_rounds + 1):
            iter_start = time()
            log(f"=== Attempt {attempt}/{max_rounds} ({mode}) ===\n{prompt}")
            ask_start = time()
            new_rules, errors, calls = yield from self._ask(prompt, schema, spec, log, tasks, attempt, mode)
            ask_seconds = time() - ask_start
            if new_rules is None:
                new_rules = verifier.empty_policy()
            candidate = best[0].extended(new_rules) if mode == "extend" else new_rules
            log(f"Candidate policy ({len(candidate.rules)} rules):\n{candidate.listing()}")

            try:
                v = verifier.verify(candidate)
            except PrismError as e:
                # Rare: PRISM cannot solve this candidate's induced model even with the fallback methods.
                # Count the round as producing nothing (the empty policy) instead of losing the instance.
                log(f"Verification failed ({str(e).splitlines()[0]}); scoring the round as an empty policy")
                errors.append(f"verification failed: {str(e).splitlines()[0]}")
                candidate = best[0] if mode == "extend" and best else verifier.empty_policy()
                v = verifier.verify(candidate)
            score = _score(v, reqs)
            previous_shortfall = best[2][2] if best else None
            improved = best is None or score < best[2]
            if improved:
                best = (candidate, v, score)
                kept_joint, joint_queried = None, False
            gain = (previous_shortfall - best[2][2]) if previous_shortfall is not None else float("inf")
            log("Results: " + ", ".join(f"{r.name}: best={v.best[r.name]:.4f} worst={v.worst[r.name]:.4f}"
                                        for r in reqs))
            log(f"Score {score} ({'new best' if improved else 'no improvement, keeping best'})")

            record = {
                "iteration": attempt,
                "mode": mode,
                "rules": candidate.to_dicts(),
                "num_rules": len(candidate.rules),
                "best": dict(v.best),
                "worst": dict(v.worst),
                "reachable_situations": v.reachable_situations,
                "uncovered_situations": v.uncovered_situations,
                "best_case_success": all(r.satisfied(v.best[r.name]) for r in reqs),
                "worst_case_success": all(r.satisfied(v.worst[r.name]) for r in reqs),
                "improved": improved,
                "shortfall_gain": gain,
                "invalid_answers": errors,
                "llm_calls": len(calls),
                "llm_time": sum(c.seconds for c in calls),
                "llm_server_time": sum(c.server_seconds for c in calls),
                "llm_wall_time": ask_seconds,        # includes queueing and, in lockstep, waiting for the batch
                "llm_output_tokens": sum(c.output_tokens for c in calls),
                "llm_prompt_tokens": sum(c.prompt_tokens for c in calls),
                "prism_time": v.seconds,
                "joint_time": 0.0,
                "kept_joint_feasible": None,        # joint best case of the kept policy (if queried)
                "branch_disagreement": False,       # per-requirement best passes but joint fails
                "prompt": prompt,
                "raw_outputs": [c.text for c in calls],
            }
            iterations.append(record)

            best_policy, best_v, _ = best
            failing_best = [r for r in reqs if not r.satisfied(best_v.best[r.name])]
            failing_worst = [r for r in reqs if not r.satisfied(best_v.worst[r.name])]
            if not failing_worst:
                log(f"Success after {attempt} attempts")
                record["iteration_time"] = time() - iter_start
                break

            if attempt < max_rounds:
                stall = 0 if improved else stall + 1
                if self.retry.restart(attempt + 1, stall, gain):
                    mode, stall = "initial", 0
                    prompt = self.domain.render("initial.md.j2", instance, **base_ctx)
                elif cfg.planner.feedback == "table":
                    mode = "table"
                    prompt = self.domain.render("table.md.j2", instance, **base_ctx,
                                                **self._results_context(best_policy, best_v, reqs))
                else:
                    joint_conflict = False
                    if not failing_best and cfg.planner.branch == "joint":
                        if not joint_queried:
                            t = time()
                            kept_joint, joint_queried = verifier.jointly_feasible(best_policy), True
                            record["joint_time"] = time() - t
                        record["kept_joint_feasible"] = kept_joint
                        joint_conflict = kept_joint is False   # undecided: fall back to per-requirement
                        record["branch_disagreement"] = joint_conflict
                        if joint_conflict:
                            log("Each requirement passes its best case, but no single completion passes all: refine")
                    results_ctx = self._results_context(best_policy, best_v, reqs)
                    if failing_best or joint_conflict:
                        mode = "refine"
                        blamed = failing_best or failing_worst
                        blame = analyzer.rule_blame(best_v, blamed) if show_blame else []
                        blame_ctx = [{
                            "rule": b.rule, "mass": b.mass,
                            "text": (f"{best_policy.rules[b.rule].condition} (action {best_policy.rules[b.rule].action})"
                                     if b.rule is not None else ""),
                            "states": [self.domain.format_state(h.valuation) for h in b.hotspots],
                        } for b in blame]
                        prompt = self.domain.render("refine.md.j2", instance, **base_ctx, **results_ctx,
                                                    failing=[r.name for r in blamed], blame=blame_ctx,
                                                    with_cost=any(r.reward is not None for r in blamed),
                                                    joint_conflict=joint_conflict, show_blame=show_blame, blame_kind=cfg.feedback.blame)
                    else:
                        mode = "extend"
                        hotspots = analyzer.uncovered_hotspots(best_v, failing_worst) if show_blame else []
                        hot_ctx = [{"mass": h.mass, "state": self.domain.format_state(h.valuation)} for h in hotspots]
                        prompt = self.domain.render("extend.md.j2", instance, **base_ctx, **results_ctx,
                                                    failing=[r.name for r in failing_worst], hotspots=hot_ctx,
                                                    with_cost=any(r.reward is not None for r in failing_worst),
                                                    show_blame=show_blame, blame_kind=cfg.feedback.blame)
            record["iteration_time"] = time() - iter_start

        best_policy, best_v, _ = best
        return {
            "success": all(r.satisfied(best_v.worst[r.name]) for r in reqs),
            "best_case_success": all(r.satisfied(best_v.best[r.name]) for r in reqs),
            "final_best": dict(best_v.best),
            "final_worst": dict(best_v.worst),
            "final_rules": best_policy.to_dicts(),
            "optimum": optimum,
            "optimum_time": optimum_seconds,
            "iterations": iterations,
        }
