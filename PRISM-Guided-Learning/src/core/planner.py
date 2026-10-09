"""The refinement loop: LLM writes symbolic rules, PRISM checks best/worst case, feedback targets the gap.

Each round, for the best rule set so far (keep-best, as in the legacy loop):
  * worst case meets every threshold   -> done (every completion of the partial policy is safe)
  * no completion can meet them all    -> refine: existing rules are bad; the LLM rewrites the list,
                                          guided by per-rule blame (probability mass)
  * some completion can                -> extend: rules are fine but incomplete; the LLM adds rules
                                          (appended, so existing decisions are unchanged) for the
                                          uncovered states carrying the most mass
"Can any completion meet them" is PRISM's multi-objective query (`planner.branch: joint`) or each
requirement's best case on its own (`per_requirement`, as in the `pre_phase_a` condition). The retry
policy (`planner.retry`, see core/retry.py) decides when to drop feedback and start from the initial
prompt instead. With `planner.feedback: table` (ablation S1) there is no branch: every round after a
failure shows the results table and asks for a complete new rule list. The loop's solver is fast but
not sound; the final rule set is re-verified exactly (`prism.exact_check`), and the result reports
those values. Semantics: docs/semantics.md.

A training set (core/training.py) is solved the same way with one rule list for all its members: each
round verifies the list on every member, keep-best sums the members' scores, the set is done when every
member passes, and the feedback is about the member that fails worst (with a summary of the others).

The planner never calls a model: `solve_steps` is the loop as a generator that yields each LLM task
(core/tasks.py) and is sent its result. `solve` answers the tasks one at a time with a backend
(core/backends); core/scheduler.py can instead batch the tasks of many instances.
"""
import zlib
from dataclasses import dataclass
from time import time
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Tuple, TypeVar, Union

from pydantic import BaseModel, Field, ValidationError, create_model

from config import Config
from core.analysis import MassAnalyzer
from core.backends import LLMBackend
from core.domain import Domain, Instance, Requirement, failing
from core.prism import PrismError, PrismRunner
from core.retry import RetryPolicy
from core.rules import ANY, RuleError, SymbolicPolicy, Vocabulary
from core.scheduler import drive
from core.tasks import LLMError, LLMResult, LLMTask, TaskFactory
from core.training import TrainingSet
from core.verifier import PolicyVerifier, Verification

T = TypeVar("T")
Steps = Generator[LLMTask, LLMResult, T]   # yields LLM tasks, is sent their results, returns T


def rule_schema(actions: List[str], max_rules: int, max_condition_chars: int) -> type[BaseModel]:
    """JSON schema for the LLM's answer; `actions` are the values a rule's action may take (with `any` for the
    extended vocabulary). The length bounds are enforced by constrained decoding, which stops degenerate
    repetition loops from running into the token limit."""
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


def _pessimistic(values: List[Dict[str, float]], reqs: List[Requirement]) -> Dict[str, float]:
    """Per requirement, the least favourable of several members' values."""
    return {r.name: (min if r.maximize else max)(d[r.name] for d in values) for r in reqs}


@dataclass
class _Member:
    """One instance the rules are written for, with what stays fixed while it is checked."""
    label: str                        # how prompts and logs name it in a training set
    instance: Instance
    verifier: PolicyVerifier
    analyzer: MassAnalyzer
    vocabulary: Vocabulary            # what rule conditions may name besides the policy variables
    context: Dict[str, Any]           # prompt context of this instance

    @property
    def requirements(self) -> List[Requirement]:
        return self.verifier.spec.requirements


@dataclass
class _Episode:
    """What stays fixed while one instance, or one training set, is solved."""
    members: List[_Member]            # one for a single instance
    schema: type
    log: Callable[[str], None]
    tasks: TaskFactory                # numbers the episode's LLM calls (their seeds)
    training: Dict[str, Any]          # members and variables as the prompt describes them; {} for one instance

    @property
    def primary(self) -> _Member:
        return self.members[0]


@dataclass
class _Answer:
    """What one round's questions to the LLM produced."""
    policies: Optional[List[SymbolicPolicy]]   # the rules parsed for each member; None if every answer was invalid
    errors: List[str]                 # invalid answers (and a failed verification), in order
    results: List[LLMResult]          # one per LLM call, re-asks included
    seconds: float = 0.0              # wall time, including queueing and, in lockstep, waiting for the batch


@dataclass
class _Check:
    """One rule list checked on every member: parsed for each (instance constants differ) and verified."""
    policies: List[SymbolicPolicy]
    vs: List[Verification]

    @property
    def policy(self) -> SymbolicPolicy:
        """The rules as written, the same text for every member."""
        return self.policies[0]

    def score(self, members: List[_Member]) -> Tuple:
        """The members' scores, summed."""
        return tuple(map(sum, zip(*(_score(v, m.requirements) for m, v in zip(members, self.vs)))))

    def passes(self, members: List[_Member], case: str = "worst") -> bool:
        return all(not failing(m.requirements, getattr(v, case)) for m, v in zip(members, self.vs))

    def focus(self, members: List[_Member]) -> int:
        """The member the feedback is about: the one that fails worst (the first, on ties)."""
        scores = [_score(v, m.requirements) for m, v in zip(members, self.vs)]
        return scores.index(max(scores))


@dataclass
class _Kept:
    """The best rule set so far (keep-best), with its check and score."""
    check: _Check
    score: Tuple
    focus: int                        # index of the member the feedback is about
    joint: Optional[bool] = None      # can one completion meet every threshold there? (None: undecided)
    joint_queried: bool = False


def _record(attempt: int, mode: str, prompt: str, check: _Check, members: List[_Member], *, improved: bool,
            gain: float, answer: _Answer) -> Dict[str, Any]:
    """One entry of the result's `iterations`. Values over a training set are each requirement's least
    favourable member value, with each member's own under `members`. The feedback step fills in the joint
    fields."""
    reqs = members[0].requirements
    record = {
        "iteration": attempt,
        "mode": mode,
        "rules": check.policy.to_dicts(),
        "num_rules": len(check.policy.rules),
        "best": _pessimistic([v.best for v in check.vs], reqs),
        "worst": _pessimistic([v.worst for v in check.vs], reqs),
        "reachable_situations": sum(v.reachable_situations for v in check.vs),
        "uncovered_situations": sum(v.uncovered_situations for v in check.vs),
        "best_case_success": check.passes(members, "best"),
        "worst_case_success": check.passes(members),
        "improved": improved,
        "shortfall_gain": gain,
        "invalid_answers": answer.errors,
        "llm_calls": len(answer.results),
        "llm_time": sum(c.seconds for c in answer.results),
        "llm_server_time": sum(c.server_seconds for c in answer.results),
        "llm_wall_time": answer.seconds,
        "llm_output_tokens": sum(c.output_tokens for c in answer.results),
        "llm_prompt_tokens": sum(c.prompt_tokens for c in answer.results),
        "prism_time": sum(v.seconds for v in check.vs),
        "joint_time": 0.0,
        "kept_joint_feasible": None,        # joint best case of the kept policy (if queried)
        "branch_disagreement": False,       # per-requirement best passes but joint fails
        "prompt": prompt,
        "raw_outputs": [c.text for c in answer.results],
    }
    if len(members) > 1:
        record["members"] = [{"instance": m.instance.id, "best": dict(v.best), "worst": dict(v.worst),
                              "reachable_situations": v.reachable_situations,
                              "uncovered_situations": v.uncovered_situations}
                             for m, v in zip(members, check.vs)]
    return record


class SymbolicPlanner:
    def __init__(self, domain: Domain, backend: Optional[LLMBackend], config: Config,
                 runner: Optional[PrismRunner] = None):
        """`backend` answers the tasks of `solve`; schedulers drive `solve_steps` with their own (None)."""
        self.domain = domain
        self.backend = backend
        self.config = config
        self.runner = runner or PrismRunner(config.prism)
        self.retry = RetryPolicy.parse(self.config.planner.retry)

    # ---------------------------------------------------------------- prompting

    def _prompt_context(self, instance: Instance, spec, vocabulary: Vocabulary) -> Dict[str, Any]:
        prompt = self.config.prompt
        return {
            "description": self.domain.description(instance),
            "visual": self.domain.visual(instance),
            "examples": self.domain.examples(instance, extended=vocabulary.allow_any) if prompt.examples else "",
            "catch_all_instruction": prompt.catch_all_instruction,
            "variables": spec.variables,
            "actions": spec.actions,
            "requirements": spec.requirements,
            "constants": vocabulary.constants,
            "features": vocabulary.features,
            "any_action": vocabulary.allow_any,
            "general": vocabulary.general,
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

    @staticmethod
    def _training_results(ep: _Episode, kept: _Kept) -> List[Dict[str, Any]]:
        """Per member of a training set: requirements met in the worst case and the failing ones."""
        rows = []
        for m, v in zip(ep.members, kept.check.vs):
            fails = failing(m.requirements, v.worst)
            rows.append({"label": m.label.capitalize(), "met": len(m.requirements) - len(fails),
                         "total": len(m.requirements), "failing": [r.name for r in fails]})
        return rows

    def _render(self, ep: _Episode, template: str, member: Optional[_Member] = None, **extra) -> str:
        """`template` for `member` (default: the first), with every member described for a training set."""
        member = member or ep.primary
        return self.domain.render(template, member.instance, **member.context, training=ep.training, **extra)

    def _parse(self, ep: _Episode, raw: List[Tuple[str, str]]) -> List[SymbolicPolicy]:
        """The rules parsed for each member. In a training set, an error names the member it occurred on."""
        policies = []
        for m in ep.members:
            spec = m.verifier.spec
            try:
                policies.append(SymbolicPolicy.from_raw(spec.variables, list(spec.actions), raw, m.vocabulary))
            except RuleError as e:
                raise RuleError(f"on {m.label}: {e}" if len(ep.members) > 1 else str(e)) from e
        return policies

    def _ask(self, ep: _Episode, prompt: str, attempt: int, mode: str) -> Steps[_Answer]:
        """Query the LLM, re-asking with the errors if the rules do not parse. A backend failure
        raises `LLMError`, which ends the instance."""
        start = time()
        answer = _Answer(None, [], [])
        json_schema = ep.schema.model_json_schema()
        current = prompt
        for fixup in range(1 + self.config.planner.max_fixups):
            result = yield ep.tasks.make(current, json_schema, round=attempt, mode=mode, fixup=fixup)
            answer.results.append(result)
            if result.error is not None:
                raise LLMError(result.error)
            try:
                parsed = ep.schema.model_validate_json(result.text)
                answer.policies = self._parse(ep, [(r.condition, r.action) for r in parsed.rules])
                break
            except (ValidationError, RuleError) as e:
                message = str(e)
                answer.errors.append(message)
                ep.log(f"Invalid LLM answer: {message}")
                current = self.domain.render("invalid.md.j2", prompt=prompt, errors=message)
        answer.seconds = time() - start
        return answer

    # ---------------------------------------------------------------- main loop

    def solve(self, target: Union[Instance, TrainingSet], logger) -> Dict[str, Any]:
        """Run the loop for one instance or training set, answering each task with the planner's backend."""
        return drive(self.solve_steps(target, logger), self.backend.execute)

    def solve_steps(self, target: Union[Instance, TrainingSet], logger) -> Steps[Dict[str, Any]]:
        """The loop for one instance or training set, as a generator: yields each LLM task, is sent its
        result, and returns the result dict. `solve` and the schedulers (core/scheduler.py) drive it."""
        ep = self._episode(target, logger.info)
        members, max_rounds = ep.members, self.config.planner.max_rounds

        opt_start = time()
        optima = [m.verifier.optimum()[0] for m in members]
        optimum_seconds = time() - opt_start
        for m, optimum in zip(members, optima):
            ep.log(f"Unconstrained optimum per requirement{self._on(ep, m)}: {optimum}")

        iterations: List[Dict[str, Any]] = []
        kept: Optional[_Kept] = None
        mode, prompt, stall = "initial", self._render(ep, "initial.md.j2"), 0
        for attempt in range(1, max_rounds + 1):
            iter_start = time()
            ep.log(f"=== Attempt {attempt}/{max_rounds} ({mode}) ===\n{prompt}")
            check, answer = yield from self._round(ep, mode, prompt, attempt, kept)
            score = check.score(members)
            previous_shortfall = kept.score[2] if kept else None
            improved = kept is None or score < kept.score
            if improved:
                kept = _Kept(check, score, check.focus(members))
            gain = previous_shortfall - kept.score[2] if previous_shortfall is not None else float("inf")
            for m, v in zip(members, check.vs):
                ep.log(f"Results{self._on(ep, m)}: " + ", ".join(
                    f"{r.name}: best={v.best[r.name]:.4f} worst={v.worst[r.name]:.4f}" for r in m.requirements))
            ep.log(f"Score {score} ({'new best' if improved else 'no improvement, keeping best'})")
            record = _record(attempt, mode, prompt, check, members, improved=improved, gain=gain, answer=answer)
            iterations.append(record)

            done = kept.check.passes(members)
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

        finals, checks, check_start = list(kept.check.vs), ["off"] * len(members), time()
        if self.config.prism.exact_check:
            for i, (m, policy) in enumerate(zip(members, kept.check.policies)):
                exact, solver = m.verifier.verify_exact(policy)
                finals[i], checks[i] = (exact, solver) if exact else (kept.check.vs[i], "failed")
                ep.log(f"Exact check{self._on(ep, m)} ({checks[i]}): " + ", ".join(
                    f"{r.name}: best={finals[i].best[r.name]:.4f} worst={finals[i].worst[r.name]:.4f}"
                    for r in m.requirements))
        final = _Check(kept.check.policies, finals)
        reqs = ep.primary.requirements
        result = {
            "success": final.passes(members),
            "best_case_success": final.passes(members, "best"),
            "final_best": _pessimistic([v.best for v in finals], reqs),
            "final_worst": _pessimistic([v.worst for v in finals], reqs),
            "final_check": ", ".join(dict.fromkeys(checks)),   # solver of the exact check, "off", or "failed"
            "final_check_time": time() - check_start,
            "loop_best": _pessimistic([v.best for v in kept.check.vs], reqs),   # the loop solver's values
            "loop_worst": _pessimistic([v.worst for v in kept.check.vs], reqs),
            "final_rules": kept.check.policy.to_dicts(),
            "optimum": _pessimistic(optima, reqs),
            "optimum_time": optimum_seconds,
            "iterations": iterations,
        }
        if len(members) > 1:
            result["members"] = [{"instance": m.instance.id, "success": not failing(m.requirements, v.worst),
                                  "final_best": dict(v.best), "final_worst": dict(v.worst), "optimum": optimum}
                                 for m, v, optimum in zip(members, finals, optima)]
        return result

    @staticmethod
    def _on(ep: _Episode, member: _Member) -> str:
        """Log suffix naming the member in a training set."""
        return f" on {member.label}" if len(ep.members) > 1 else ""

    def _episode(self, target: Union[Instance, TrainingSet], log: Callable[[str], None]) -> _Episode:
        cfg = self.config
        instances = target.members if isinstance(target, TrainingSet) else (target,)
        members = []
        for k, instance in enumerate(instances, start=1):
            verifier = PolicyVerifier(self.domain, instance, cfg, self.runner)
            analyzer = MassAnalyzer(verifier, cfg.feedback.horizon_for(self.domain, instance), cfg.feedback.top_k,
                                    cfg.feedback.states_per_rule, method=cfg.feedback.blame,
                                    seed=zlib.crc32(f"{cfg.llm.seed}/{instance.id}".encode()))
            vocabulary = verifier.spec.vocabulary(cfg.rules)
            members.append(_Member(f"instance {k}", instance, verifier, analyzer, vocabulary,
                                   self._prompt_context(instance, verifier.spec, vocabulary)))
        self._check_compatible(members)
        primary = members[0]
        answers = list(primary.verifier.spec.actions) + ([ANY] if primary.vocabulary.allow_any else [])
        schema = rule_schema(answers, cfg.planner.max_rules, cfg.planner.max_condition_chars)
        return _Episode(members, schema, log, TaskFactory(cfg.llm, target.id),
                        self._training_context(members) if len(members) > 1 else {})

    @staticmethod
    def _training_context(members: List[_Member]) -> Dict[str, Any]:
        """How the prompt describes a training set: every member, and every variable, with each member's range
        or description where they differ."""
        def per_member(values: List[str]) -> str:
            return values[0] if len(set(values)) == 1 else "; ".join(
                f"{v} on {m.label}" for m, v in zip(members, values))
        variables = []
        for i, var in enumerate(members[0].verifier.spec.variables):
            versions = [m.verifier.spec.variables[i] for m in members]
            variables.append({"name": var.name, "type": per_member([v.describe_type() for v in versions]),
                              "description": per_member([v.description or "" for v in versions])})
        return {"members": [{"label": m.label.capitalize(), "description": m.context["description"],
                             "visual": m.context["visual"], "constants": m.vocabulary.constants} for m in members],
                "variables": variables}

    @staticmethod
    def _check_compatible(members: List[_Member]) -> None:
        """Members of a training set must share the policy variables, actions and requirement names, so one
        rule list and one results table mean the same on each."""
        def shape(m: _Member):
            spec = m.verifier.spec
            return ([(v.name, v.type) for v in spec.variables], list(spec.actions), [r.name for r in m.requirements])
        for m in members[1:]:
            if shape(m) != shape(members[0]):
                raise ValueError(f"{m.label} has other variables, actions or requirements than instance 1")

    def _round(self, ep: _Episode, mode: str, prompt: str, attempt: int,
               kept: Optional[_Kept]) -> Steps[Tuple[_Check, _Answer]]:
        """Ask for rules and check the candidate on every member: the new rules, or the kept ones followed by
        them when extending. Returns the check and what the LLM calls produced."""
        answer = yield from self._ask(ep, prompt, attempt, mode)
        new = answer.policies or [m.verifier.empty_policy() for m in ep.members]
        policies = ([k.extended(n) for k, n in zip(kept.check.policies, new)] if mode == "extend" else new)
        ep.log(f"Candidate policy ({len(policies[0].rules)} rules):\n{policies[0].listing()}")
        try:
            return _Check(policies, [m.verifier.verify(p) for m, p in zip(ep.members, policies)]), answer
        except PrismError as e:
            # Rare: PRISM cannot solve this candidate's induced model even with the fallback methods.
            # Count the round as producing nothing (the empty policy) instead of losing the instance.
            reason = str(e).splitlines()[0]
            ep.log(f"Verification failed ({reason}); scoring the round as an empty policy")
            answer.errors.append(f"verification failed: {reason}")
            policies = (kept.check.policies if mode == "extend" and kept
                        else [m.verifier.empty_policy() for m in ep.members])
            return _Check(policies, [m.verifier.verify(p) for m, p in zip(ep.members, policies)]), answer

    # ---------------------------------------------------------------- feedback

    def _feedback(self, ep: _Episode, kept: _Kept, record: Dict[str, Any]) -> Tuple[str, str]:
        """Mode and prompt of the next round, from the kept rule set's results on the member that fails worst."""
        member, policy, v = ep.members[kept.focus], kept.check.policies[kept.focus], kept.check.vs[kept.focus]
        reqs = member.requirements
        results = self._results_context(policy, v, reqs)
        if ep.training:
            results.update(training_results=self._training_results(ep, kept), focus=member.label)
        if self.config.planner.feedback == "table":
            return "table", self._render(ep, "table.md.j2", member, **results)
        failing_best, failing_worst = failing(reqs, v.best), failing(reqs, v.worst)
        joint_conflict = (not failing_best and self.config.planner.branch == "joint"
                          and self._joint_conflict(ep, kept, record))
        # Every situation covered, yet best and worst differ: rules allow several actions (`any`) and some of
        # them are harmful. Appending rules cannot change covered states, so rewrite them.
        open_choices = bool(failing_worst) and v.uncovered_situations == 0
        if failing_best or joint_conflict or open_choices:
            return "refine", self._refine_prompt(ep, member, policy, v, failing_best or failing_worst,
                                                 joint_conflict, results)
        return "extend", self._extend_prompt(ep, member, v, failing_worst, results)

    def _joint_conflict(self, ep: _Episode, kept: _Kept, record: Dict[str, Any]) -> bool:
        """Whether no single completion of the kept rules meets every threshold (on the member the feedback is
        about), although each requirement passes its best case. Asked once per kept rule set; undecided (None)
        counts as no conflict."""
        if not kept.joint_queried:
            start = time()
            member = ep.members[kept.focus]
            kept.joint, kept.joint_queried = member.verifier.jointly_feasible(kept.check.policies[kept.focus]), True
            record["joint_time"] = time() - start
        conflict = kept.joint is False
        record["kept_joint_feasible"], record["branch_disagreement"] = kept.joint, conflict
        if conflict:
            ep.log("Each requirement passes its best case, but no single completion passes all: refine")
        return conflict

    def _refine_prompt(self, ep: _Episode, member: _Member, policy: SymbolicPolicy, v: Verification,
                       blamed: List[Requirement], joint_conflict: bool, results: Dict[str, Any]) -> str:
        """Rewrite the rules, shown how much of the lost probability each rule's states carry."""
        kind = self.config.feedback.blame
        rules = policy.rules
        blame = [{
            "rule": b.rule, "mass": b.mass,
            "text": f"{rules[b.rule].condition} (action {rules[b.rule].action})" if b.rule is not None else "",
            "states": [self.domain.format_state(h.valuation) for h in b.hotspots],
        } for b in (member.analyzer.rule_blame(v, blamed) if kind != "none" else [])]
        return self._render(ep, "refine.md.j2", member, **results, failing=[r.name for r in blamed], blame=blame,
                            with_cost=any(r.reward is not None for r in blamed), joint_conflict=joint_conflict,
                            show_blame=kind != "none", blame_kind=kind)

    def _extend_prompt(self, ep: _Episode, member: _Member, v: Verification, failing_worst: List[Requirement],
                       results: Dict[str, Any]) -> str:
        """Add rules, shown the uncovered states where the worst case loses the most probability."""
        kind = self.config.feedback.blame
        hotspots = member.analyzer.uncovered_hotspots(v, failing_worst) if kind != "none" else []
        return self._render(ep, "extend.md.j2", member, **results, failing=[r.name for r in failing_worst],
                            hotspots=[{"mass": h.mass, "state": self.domain.format_state(h.valuation)}
                                      for h in hotspots],
                            with_cost=any(r.reward is not None for r in failing_worst),
                            show_blame=kind != "none", blame_kind=kind)
