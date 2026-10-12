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

The planner never calls a model or starts PRISM itself: `solve_steps` is the loop as a generator that
yields each LLM task (core/tasks.py) and each PRISM process (`PrismCall`, core/prism.py), and is sent
the result. `solve` runs them one at a time with a backend (core/backends) and PRISM; the schedulers in
core/scheduler.py run many instances' tasks and PRISM processes at once from one thread.
"""
import zlib
from dataclasses import dataclass, field, replace
from time import time
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Tuple, TypeVar, Union

from pydantic import BaseModel, Field, ValidationError, create_model

from config import Config
from core.analysis import MassAnalyzer
from core.backends import LLMBackend
from core.definitions import Definition, DefinitionError, Evaluator, domain_definitions, model_label_texts
from core.domain import Domain, Instance, Requirement, failing
from core.prism import PrismCall, PrismError, PrismRunner
from core.retry import RetryPolicy
from core.rules import ANY, Feature, RuleError, SymbolicPolicy, Vocabulary
from core.scheduler import drive
from core.tasks import LLMError, LLMResult, LLMTask, TaskFactory
from core.training import TrainingSet
from core.verifier import PolicyVerifier, Verification

T = TypeVar("T")
# Yields LLM tasks and PRISM calls, is sent their results (an LLMResult, PRISM's output), returns T.
Steps = Generator[Union[LLMTask, PrismCall], Union[LLMResult, str], T]


def rule_schema(actions: List[str], max_rules: int, max_condition_chars: int,
                max_new_definitions: int = 0) -> type[BaseModel]:
    """JSON schema for the LLM's answer; `actions` are the values a rule's action may take (with `any` for the
    extended vocabulary). The length bounds are enforced by constrained decoding, which stops degenerate
    repetition loops from running into the token limit. With `max_new_definitions`, the answer may also
    define up to that many features (rules.vocabulary=llm)."""
    rule = create_model(
        "Rule",
        condition=(str, Field(description="Boolean condition over the state variables",
                              max_length=max_condition_chars)),
        action=(Literal.__getitem__(tuple(actions)), Field(description="Action to take")),
    )
    if max_new_definitions > 0:
        return create_model("RuleList", rules=(List[rule], Field(..., max_length=max_rules)),
                            definitions=(List[_definition_model()],
                                         Field(default_factory=list, max_length=max_new_definitions)))
    return create_model("RuleList", rules=(List[rule], Field(..., max_length=max_rules)))


def _definition_model() -> type[BaseModel]:
    check = create_model(
        "Check",
        state=(str, Field(description="Condition describing example states of instance 1")),
        action=(Optional[str], Field(default=None, description="The action, for a per-action feature")),
        holds=(bool, Field(description="Whether the feature should hold there")),
    )
    return create_model(
        "Definition",
        name=(str, Field(description="Feature name: letters, digits and _")),
        description=(str, Field(description="What the feature means, in English")),
        definition=(Union[str, Dict[str, str]],
                    Field(description="The condition in the definition language, or one per action")),
        checks=(List[check], Field(..., min_length=2, max_length=4,
                                   description="Example states where the feature should and should not hold")),
    )


def definition_schema(max_definitions: int) -> type[BaseModel]:
    """JSON schema for the vocabulary step's answer: up to `max_definitions` feature definitions."""
    return create_model("DefinitionList", definitions=(List[_definition_model()],
                                                       Field(..., max_length=max_definitions)))


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
    evaluator: Optional[Evaluator] = None   # turns definitions into features on this instance (not `domain`)

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
    vocabulary: str = "domain"        # where features come from (rules.vocabulary)
    definitions: List[Definition] = field(default_factory=list)   # every definition added, in order
    definition_schema: Optional[type] = None                     # the vocabulary step's answer (llm)

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
    definitions: List[Definition] = field(default_factory=list)   # features the accepted answer defined


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
        "new_definitions": [d.to_dict() for d in answer.definitions],
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
            # With rules.vocabulary=llm the domain's examples would name features that do not exist
            "examples": (self.domain.examples(instance, extended=vocabulary.allow_any)
                         if prompt.examples and self.config.rules.vocabulary != "llm" else ""),
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
        if ep.vocabulary == "llm":
            extra["definitions"] = self._definition_context(ep, extra.pop("vocabulary_step", False))
        return self.domain.render(template, member.instance, **member.context, training=ep.training, **extra)

    def _definition_context(self, ep: _Episode, vocabulary_step: bool) -> Dict[str, Any]:
        """How the prompt explains the definition language, and how many more features may be defined."""
        rules = self.config.rules
        remaining = max(0, rules.max_definitions - len(ep.definitions))
        return {"level": rules.definition_level, "general": rules.general, "training": len(ep.members) > 1,
                "labels": model_label_texts(ep.primary.verifier.model), "vocabulary_step": vocabulary_step,
                "remaining": remaining,
                "max_new": remaining if vocabulary_step else min(remaining, rules.max_new_definitions)}

    def _parse(self, ep: _Episode, raw: List[Tuple[str, str]],
               extra: Optional[List[List[Feature]]] = None) -> List[SymbolicPolicy]:
        """The rules parsed for each member (with `extra` features per member, not yet committed). In a training
        set, an error names the member it occurred on."""
        policies = []
        for i, m in enumerate(ep.members):
            spec, vocab = m.verifier.spec, m.vocabulary
            if extra:
                vocab = Vocabulary(vocab.constants, vocab.features + extra[i], vocab.allow_any, vocab.general)
            try:
                policies.append(SymbolicPolicy.from_raw(spec.variables, list(spec.actions), raw, vocab))
            except RuleError as e:
                raise RuleError(f"on {m.label}: {e}" if len(ep.members) > 1 else str(e)) from e
        return policies

    def _evaluate(self, ep: _Episode, definitions: List[Definition]) -> Steps[List[List[Feature]]]:
        """`definitions` as features on each member, not yet added. With LLM-written definitions, a feature's
        prompt description also shows its definition. Raises DefinitionError (naming the member in a set)."""
        out = []
        for m in ep.members:
            try:
                features = yield from m.evaluator.features_steps(definitions, [f.name for f in m.vocabulary.features])
            except PrismError as e:
                raise DefinitionError(f"PRISM could not evaluate a definition: {str(e).splitlines()[0]}") from e
            except DefinitionError as e:
                raise DefinitionError(f"on {m.label}: {e}" if len(ep.members) > 1 else str(e)) from e
            if ep.vocabulary == "llm":
                if m is ep.primary:          # the definer's checks describe states of instance 1
                    for d in definitions:
                        m.evaluator.check_examples(d)
                features = [replace(f, description=f"{d.description} (defined as `{d.listing()}`)")
                            for f, d in zip(features, definitions)]
            out.append(features)
        return out

    @staticmethod
    def _commit(ep: _Episode, definitions: List[Definition], features: List[List[Feature]]) -> None:
        """Add the features for good (add-only): every member's vocabulary, and so its prompt context."""
        for m, fs in zip(ep.members, features):
            m.vocabulary.features.extend(fs)
        ep.definitions.extend(definitions)

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
                new = [Definition.from_dict(d.model_dump()) for d in getattr(parsed, "definitions", [])]
                extra = None
                if new:
                    room = self.config.rules.max_definitions - len(ep.definitions)
                    if len(new) > room:
                        raise DefinitionError(f"{len(new)} new definitions, but only {room} more are allowed")
                    extra = yield from self._evaluate(ep, new)
                answer.policies = self._parse(ep, [(r.condition, r.action) for r in parsed.rules], extra)
                if new:
                    self._commit(ep, new, extra)
                    answer.definitions = new
                    ep.log("New definitions:\n" + "\n".join(f"{d.name}: {d.listing()}" for d in new))
                break
            except (ValidationError, RuleError, DefinitionError) as e:
                message = str(e)
                answer.errors.append(message)
                ep.log(f"Invalid LLM answer: {message}")
                current = self.domain.render("invalid.md.j2", prompt=prompt, errors=message)
        answer.seconds = time() - start
        return answer

    # ---------------------------------------------------------------- main loop

    def solve(self, target: Union[Instance, TrainingSet], logger) -> Dict[str, Any]:
        """Run the loop for one instance or training set, answering each task with the planner's backend and
        running each PRISM call, one at a time."""
        return drive(self.solve_steps(target, logger), self.backend.execute)

    def solve_steps(self, target: Union[Instance, TrainingSet], logger) -> Steps[Dict[str, Any]]:
        """The loop for one instance or training set, as a generator: yields each LLM task and PRISM call, is
        sent its result, and returns the result dict. `solve` and the schedulers (core/scheduler.py) drive it."""
        ep = self._episode(target, logger.info)
        members, max_rounds = ep.members, self.config.planner.max_rounds

        opt_start = time()
        optima = []
        for m in members:
            optimum, _, _ = yield from m.verifier.optimum_steps()
            optima.append(optimum)
        optimum_seconds = time() - opt_start
        for m, optimum in zip(members, optima):
            ep.log(f"Unconstrained optimum per requirement{self._on(ep, m)}: {optimum}")

        vocabulary_record = None
        if ep.vocabulary == "definitions":
            definitions = domain_definitions(self.domain, self.config.rules,
                                             [v.name for v in ep.primary.verifier.spec.variables])
            self._commit(ep, definitions, (yield from self._evaluate(ep, definitions)))
        elif ep.vocabulary == "llm":
            vocabulary_record = yield from self._vocabulary_step(ep)

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
                    mode, prompt = yield from self._feedback(ep, kept, record)
            record["iteration_time"] = time() - iter_start
            if done:
                break

        finals, checks, check_start = list(kept.check.vs), ["off"] * len(members), time()
        if self.config.prism.exact_check:
            for i, (m, policy) in enumerate(zip(members, kept.check.policies)):
                exact, solver = yield from m.verifier.verify_exact_steps(policy)
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
        if ep.vocabulary != "domain":
            result["definitions"] = [d.to_dict() for d in ep.definitions]
        if vocabulary_record is not None:
            result["vocabulary_step"] = vocabulary_record
        if len(members) > 1:
            result["members"] = [{"instance": m.instance.id, "success": not failing(m.requirements, v.worst),
                                  "final_best": dict(v.best), "final_worst": dict(v.worst), "optimum": optimum}
                                 for m, v, optimum in zip(members, finals, optima)]
        return result

    def _vocabulary_step(self, ep: _Episode) -> Steps[Dict[str, Any]]:
        """Before the rule loop (rules.vocabulary=llm): ask for the initial feature definitions, re-asking with
        the errors if they are invalid. If every answer is invalid, the loop starts with no features."""
        start = time()
        prompt = self._render(ep, "vocabulary.md.j2", vocabulary_step=True)
        ep.log(f"=== Vocabulary ===\n{prompt}")
        json_schema = ep.definition_schema.model_json_schema()
        results, errors, current = [], [], prompt
        for fixup in range(1 + self.config.planner.max_fixups):
            result = yield ep.tasks.make(current, json_schema, round=0, mode="vocabulary", fixup=fixup)
            results.append(result)
            if result.error is not None:
                raise LLMError(result.error)
            try:
                parsed = ep.definition_schema.model_validate_json(result.text)
                definitions = [Definition.from_dict(d.model_dump()) for d in parsed.definitions]
                self._commit(ep, definitions, (yield from self._evaluate(ep, definitions)))
                ep.log("Definitions:\n" + "\n".join(f"{d.name}: {d.listing()}" for d in definitions))
                break
            except (ValidationError, DefinitionError) as e:
                errors.append(str(e))
                ep.log(f"Invalid definitions: {e}")
                current = self.domain.render("invalid.md.j2", prompt=prompt, errors=str(e))
        return {"prompt": prompt, "definitions": [d.to_dict() for d in ep.definitions], "invalid_answers": errors,
                "llm_calls": len(results), "llm_time": sum(c.seconds for c in results),
                "llm_output_tokens": sum(c.output_tokens for c in results),
                "llm_prompt_tokens": sum(c.prompt_tokens for c in results),
                "raw_outputs": [c.text for c in results], "seconds": time() - start}

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
            evaluator = (Evaluator(verifier, cfg.rules.definition_level, cfg.rules.general, cfg.rules.definition_penalty)
                         if cfg.rules.extended and cfg.rules.vocabulary != "domain" else None)
            members.append(_Member(f"instance {k}", instance, verifier, analyzer, vocabulary,
                                   self._prompt_context(instance, verifier.spec, vocabulary), evaluator))
        self._check_compatible(members)
        primary = members[0]
        answers = list(primary.verifier.spec.actions) + ([ANY] if primary.vocabulary.allow_any else [])
        source = cfg.rules.vocabulary if cfg.rules.extended else "domain"
        llm = source == "llm"
        schema = rule_schema(answers, cfg.planner.max_rules, cfg.planner.max_condition_chars,
                             cfg.rules.max_new_definitions if llm else 0)
        return _Episode(members, schema, log, TaskFactory(cfg.llm, target.id),
                        self._training_context(members) if len(members) > 1 else {}, source,
                        definition_schema=definition_schema(cfg.rules.max_definitions) if llm else None)

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
            return (yield from self._check(ep, policies)), answer
        except PrismError as e:
            # Rare: PRISM cannot solve this candidate's induced model even with the fallback methods.
            # Count the round as producing nothing (the empty policy) instead of losing the instance.
            reason = str(e).splitlines()[0]
            ep.log(f"Verification failed ({reason}); scoring the round as an empty policy")
            answer.errors.append(f"verification failed: {reason}")
            policies = (kept.check.policies if mode == "extend" and kept
                        else [m.verifier.empty_policy() for m in ep.members])
            return (yield from self._check(ep, policies)), answer

    @staticmethod
    def _check(ep: _Episode, policies: List[SymbolicPolicy]) -> Steps[_Check]:
        """Verify each member's policy, one member after another."""
        vs = []
        for m, p in zip(ep.members, policies):
            vs.append((yield from m.verifier.verify_steps(p)))
        return _Check(policies, vs)

    # ---------------------------------------------------------------- feedback

    def _feedback(self, ep: _Episode, kept: _Kept, record: Dict[str, Any]) -> Steps[Tuple[str, str]]:
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
                          and (yield from self._joint_conflict(ep, kept, record)))
        # Every situation covered, yet best and worst differ: rules allow several actions (`any`) and some of
        # them are harmful. Appending rules cannot change covered states, so rewrite them.
        open_choices = bool(failing_worst) and v.uncovered_situations == 0
        if failing_best or joint_conflict or open_choices:
            return "refine", self._refine_prompt(ep, member, policy, v, failing_best or failing_worst,
                                                 joint_conflict, results)
        return "extend", self._extend_prompt(ep, member, v, failing_worst, results)

    def _joint_conflict(self, ep: _Episode, kept: _Kept, record: Dict[str, Any]) -> Steps[bool]:
        """Whether no single completion of the kept rules meets every threshold (on the member the feedback is
        about), although each requirement passes its best case. Asked once per kept rule set; undecided (None)
        counts as no conflict."""
        if not kept.joint_queried:
            start = time()
            member = ep.members[kept.focus]
            kept.joint = yield from member.verifier.jointly_feasible_steps(kept.check.policies[kept.focus])
            kept.joint_queried = True
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
