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
core/scheduler.py run many instances' tasks and PRISM processes at once from one thread. What the loop
works with is in core/episode.py; the prompts it sends are built by core/prompts.py.
"""
import re
import zlib
from dataclasses import replace
from time import time
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Tuple, TypeVar, Union

from pydantic import BaseModel, Field, ValidationError, create_model

from config import Config
from core import early
from core.analysis import MassAnalyzer
from core.backends import LLMBackend
from core.domain import Domain, Instance, Requirement, failing
from core.episode import Answer, Check, Episode, Kept, Member, margin, pessimistic
from core.prism import PrismCall, PrismError, PrismRunner
from core.prompts import Prompts
from core.retry import RetryPolicy
from core.rules import ANY, OpenChoices, RuleError, SymbolicPolicy, action_orders, fixed_order
from core.scheduler import drive
from core.tasks import LLMError, LLMResult, LLMTask, TaskFactory
from core.training import TrainingSet
from core.verifier import PolicyVerifier, Verification

T = TypeVar("T")
LITERAL_SUM = re.compile(r"(?<!\w)\d+\s*[-+]\s*\d+(?!\w)")    # `1 + 1`: rejected with `rules.strict` where the domain says
# Yields LLM tasks and PRISM calls, is sent their results (an LLMResult, PRISM's output), returns T.
Steps = Generator[Union[LLMTask, PrismCall], Union[LLMResult, str], T]


def rule_schema(actions: List[str], max_rules: int, max_condition_chars: int,
                strategy: bool = False) -> type[BaseModel]:
    """JSON schema for the LLM's answer; `actions` are the values a rule's action may take (with `any` for the
    extended vocabulary). The length bounds are enforced by constrained decoding, which stops degenerate
    repetition loops from running into the token limit. With `strategy`, the answer starts with the strategy
    in words (`prompt.strategy_first`)."""
    rule = create_model(
        "Rule",
        condition=(str, Field(description="Boolean condition over the state variables",
                              max_length=max_condition_chars)),
        action=(Literal.__getitem__(tuple(actions)), Field(description="Action to take")),
    )
    rules = (List[rule], Field(..., max_length=max_rules))
    if strategy:
        return create_model("RuleList", strategy=(str, Field(description="The strategy the rules implement")),
                            rules=rules)
    return create_model("RuleList", rules=rules)


def _record(attempt: int, mode: str, prompt: str, check: Check, members: List[Member], *, improved: bool,
            gain: float, answer: Answer) -> Dict[str, Any]:
    """One entry of the result's `iterations`. Values over a training set are each requirement's least
    favourable member value, with each member's own under `members`. The feedback step fills in the joint
    fields."""
    reqs = members[0].requirements
    record = {
        "iteration": attempt,
        "mode": mode,
        "rules": check.policy.to_dicts(),
        "num_rules": len(check.policy.rules),
        "best": pessimistic([v.best for v in check.vs], reqs),
        "worst": pessimistic([v.worst for v in check.vs], reqs),
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
    if answer.any_order is not None:
        record["any_order"] = answer.any_order
    if check.early is not None:
        record["early_loss"] = check.early.mean
    if answer.narrowed:
        record["narrowed"] = answer.narrowed
    if len(members) > 1:
        record["members"] = [{"instance": m.instance.id, "best": dict(v.best), "worst": dict(v.worst),
                              "reachable_situations": v.reachable_situations,
                              "uncovered_situations": v.uncovered_situations, **({"hidden": True} if m.hidden else {})}
                             for m, v in zip(members, check.vs)]
    return record


# `planner.chains`: call numbers reserved per chain, so that seeded chains on one training set get distinct seeds.
CHAIN_CALLS = 100_000

# `rules.any_prune`: how far PRISM's values may be from the domain's fast model before an order is ruled out wrongly.
# The loop's solver stops early, mostly by less than 1e-4; where a verified order is off by more than this (138 of 2,113
# recorded gridworld rounds), every order is verified (DECISIONS.md).
PRUNE_SLACK = 1e-3

# `rules.early_narrow`: a rule is kept from its move at a place only where that move loses in at least this share of the
# checked situations the rule decides there
NARROW_SHARE = 0.75


def _score_bounds(reqs: List[Requirement], best: Dict[str, float], worst: Dict[str, float]) -> Tuple[Tuple, Tuple]:
    """Bounds on `core.episode.score` of a member whose values are each within PRUNE_SLACK of these."""
    lo, hi = [0, 0, 0.0, 0.0], [0, 0, 0.0, 0.0]
    for r in reqs:
        for c, value in ((0, worst[r.name]), (1, best[r.name])):
            ends = (value - PRUNE_SLACK, value + PRUNE_SLACK)
            passes = [r.satisfied(e) for e in ends]
            lo[c] += not any(passes)
            hi[c] += not all(passes)
            gaps = [r.shortfall(e) for e in ends]
            lo[c + 2] += min(gaps) - 1e-12    # rounding in the sums
            hi[c + 2] += max(gaps) + 1e-12
    return tuple(lo), tuple(hi)


def _contenders(members: List[Member], values: List[List[Tuple[Dict[str, float], Dict[str, float]]]]) -> List[int]:
    """The orders that can score best on the members when PRISM's values are within PRUNE_SLACK of `values` (per
    member, the fast model's (best, worst) for each order): those whose lowest possible score is at most the least
    highest possible one. A score is a tuple compared in order, and summed bounds bound it in each place."""
    bounds = []
    for k in range(len(values[0])):
        parts = [_score_bounds(m.requirements, *v[k]) for m, v in zip(members, values)]
        bounds.append([tuple(map(sum, zip(*side))) for side in zip(*parts)])
    cap = min(hi for _, hi in bounds)
    return [k for k, (lo, _) in enumerate(bounds) if lo <= cap]


def _gap(v: Verification, best: Dict[str, float], worst: Dict[str, float]) -> float:
    """The largest difference between PRISM's values and the fast model's."""
    return max(max(abs(v.best[n] - best[n]), abs(v.worst[n] - worst[n])) for n in best)


class SymbolicPlanner:
    def __init__(self, domain: Domain, backend: Optional[LLMBackend], config: Config,
                 runner: Optional[PrismRunner] = None):
        """`backend` answers the tasks of `solve`; schedulers drive `solve_steps` with their own (None)."""
        self.domain = domain
        self.backend = backend
        self.config = config
        self.runner = runner or PrismRunner(config.prism)
        self.retry = RetryPolicy.parse(self.config.planner.retry)
        self.prompts = Prompts(domain, config)

    def _parse(self, ep: Episode, raw: List[Tuple[str, str]]) -> List[SymbolicPolicy]:
        """The rules parsed for each member. In a training set, an error names the member it occurred on. With
        `rules.strict`, conditions that name what the domain withholds are an error (`Domain.strict`)."""
        strict = self.prompts.strict([m.instance for m in ep.members])
        if strict:
            errors = []
            for i, (condition, action) in enumerate(raw, start=1):
                names = sorted(set(re.findall(r"[A-Za-z_]\w*", condition)) & strict["hide"])
                if names:
                    errors.append(f"rule {i} ({condition} -> {action}): uses {', '.join(names)}, which conditions may "
                                  f"not name")
                if not strict["literal_sums"] and LITERAL_SUM.search(condition):
                    errors.append(f"rule {i} ({condition} -> {action}): adds or subtracts number literals (such as "
                                  f"`1 + 1`), which is not allowed")
            if errors:
                raise RuleError("\n".join(errors))
        policies = []
        for m in ep.members:
            try:
                policies.append(m.verifier.spec.policy(raw, m.vocabulary))
            except RuleError as e:
                raise RuleError(f"on {m.label}: {e}" if len(ep.members) > 1 else str(e)) from e
        return policies

    def _ask(self, ep: Episode, prompt: str, attempt: int, mode: str) -> Steps[Answer]:
        """Query the LLM, re-asking with the errors if the rules do not parse. A backend failure
        raises `LLMError`, which ends the instance."""
        start = time()
        answer = Answer(None, [], [])
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
                current = self.prompts.invalid(prompt, message)
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
            ep.log(f"Unconstrained optimum per requirement{ep.on(m)}: {optimum}")
        if self.config.feedback.early:
            for m in members:
                if not m.hidden:
                    m.early = early.EarlyCheck(m.verifier, self.config.feedback.early, m.label)
                    ep.log(f"Early check{ep.on(m)}: {len(m.early.valuations)} states where the move matters")

        iterations: List[Dict[str, Any]] = []
        kept: Optional[Kept] = None
        mode, prompt, stall = "initial", self.prompts.initial(ep), 0
        for attempt in range(1, max_rounds + 1):
            iter_start = time()
            ep.log(f"=== Attempt {attempt}/{max_rounds} ({mode}) ===\n{prompt}")
            check, answer = yield from self._round(ep, mode, prompt, attempt, kept)
            score = check.score(members)
            previous_shortfall = kept.score[2] if kept else None
            improved = kept is None or score < kept.score or (
                self.config.planner.keep_ties == "newer" and score == kept.score)
            if improved:
                kept = Kept(check, score, check.focus(members))
            gain = previous_shortfall - kept.score[2] if previous_shortfall is not None else float("inf")
            for m, v in zip(members, check.vs):
                ep.log(f"Results{ep.on(m)}: " + ", ".join(
                    f"{r.name}: best={v.best[r.name]:.4f} worst={v.worst[r.name]:.4f}" for r in m.requirements))
            ep.log(f"Score {score} ({'new best' if improved else 'no improvement, keeping best'})")
            record = _record(attempt, mode, prompt, check, members, improved=improved, gain=gain, answer=answer)
            iterations.append(record)

            done = (kept.check.passes(members) and self._robust(kept, members)
                    and self._early_done(ep, kept.check))
            if done and ep.pool:
                grown = yield from self._counterexample(ep, kept, optima)
                if grown is not None:
                    kept, done = grown, False
            if done:
                ep.log(f"Success after {attempt} attempts")
            elif attempt < max_rounds:
                stall = 0 if improved else stall + 1
                if self.retry.restart(attempt + 1, stall, gain):
                    mode, prompt, stall = "initial", self.prompts.initial(ep), 0
                else:
                    mode, prompt = yield from self._feedback(ep, kept, record, iterations)
            record["iteration_time"] = time() - iter_start
            if done:
                break

        return (yield from self._final(ep, kept, optima, optimum_seconds, iterations))

    def _final(self, ep: Episode, kept: Kept, optima: List[Dict[str, float]], optimum_seconds: float,
               iterations: List[Dict[str, Any]]) -> Steps[Dict[str, Any]]:
        """The result: the kept rules, re-verified exactly (`prism.exact_check`; else the loop's values), and
        every round's record."""
        members = ep.members
        finals, checks, check_start = list(kept.check.vs), ["off"] * len(members), time()
        if self.config.prism.exact_check:
            for i, (m, policy) in enumerate(zip(members, kept.check.policies)):
                exact, solver = yield from m.verifier.verify_exact_steps(policy)
                finals[i], checks[i] = (exact, solver) if exact else (kept.check.vs[i], "failed")
                ep.log(f"Exact check{ep.on(m)} ({checks[i]}): " + ", ".join(
                    f"{r.name}: best={finals[i].best[r.name]:.4f} worst={finals[i].worst[r.name]:.4f}"
                    for r in m.requirements))
        final = Check(kept.check.policies, finals)
        reqs = ep.primary.requirements
        result = {
            "success": final.passes(members),
            "best_case_success": final.passes(members, "best"),
            "final_best": pessimistic([v.best for v in finals], reqs),
            "final_worst": pessimistic([v.worst for v in finals], reqs),
            "final_check": ", ".join(dict.fromkeys(checks)),   # solver of the exact check, "off", or "failed"
            "final_check_time": time() - check_start,
            "loop_best": pessimistic([v.best for v in kept.check.vs], reqs),   # the loop solver's values
            "loop_worst": pessimistic([v.worst for v in kept.check.vs], reqs),
            "final_rules": kept.check.policy.to_dicts(),
            **({"early_loss": kept.check.early.mean} if kept.check.early is not None else {}),
            "optimum": pessimistic(optima, reqs),
            "optimum_time": optimum_seconds,
            "iterations": iterations,
            **({"counterexamples": list(ep.added)} if ep.added else {}),
        }
        if len(members) > 1:
            result["members"] = [{"instance": m.instance.id, "success": not failing(m.requirements, v.worst),
                                  "final_best": dict(v.best), "final_worst": dict(v.worst), "optimum": optimum,
                                  **({"hidden": True} if m.hidden else {})}
                                 for m, v, optimum in zip(members, finals, optima)]
        return result

    def select_chain(self, target: TrainingSet, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """One result for the `planner.chains` independent loops on `target` (core.training.chains), in chain order:
        the chain whose final rules have the largest worst-case margin on the members (the smallest margin over
        members and requirements, on the reported values; the first chain on ties), with every chain's rounds,
        numbered on (`chain`, `chain_iteration`), and a summary per chain (`chains`)."""
        by_id = {inst.id: inst for inst in target.members + target.hidden}

        def chain_margin(result: Dict[str, Any]) -> Optional[float]:
            if "final_rules" not in result:
                return None
            members = result.get("members") or [{"instance": target.members[0].id, "final_worst": result["final_worst"]}]
            return min(margin(m["final_worst"], self.domain.spec(by_id[m["instance"]]).requirements) for m in members)

        margins = [chain_margin(r) for r in results]
        finished = [k for k, m in enumerate(margins) if m is not None]
        best = max(finished, key=lambda k: (margins[k], -k)) if finished else 0
        rounds = [dict(it, chain=k, chain_iteration=it["iteration"])
                  for k, r in enumerate(results) for it in r.get("iterations", [])]
        for n, it in enumerate(rounds, start=1):
            it["iteration"] = n
        summary = [{"chain": k, "success": r.get("success", False), "margin": m, "rounds": len(r.get("iterations", [])),
                    **({"error": r["error"]} if r.get("error") else {})} for k, (r, m) in enumerate(zip(results, margins))]
        return {**results[best], "iterations": rounds, "chain": best, "chains": summary, "instance": target.id,
                "total_time": sum(r.get("total_time", 0.0) for r in results)}

    def pick_candidates(self, target: TrainingSet, results: List[Dict[str, Any]], log: Callable[[str], None] = print
                        ) -> Dict[str, Any]:
        """`planner.pick: candidates` (core/pick.py): the final list of `target` from every round of its chains'
        `results`, repaired (`planner.repair`) and compared with the domain's fast checker (PRISM where there is none,
        then without repair or resolved scores), then re-verified exactly on every member. The result is
        `select_chain`'s with the chosen list and its values, and what the pick did (`pick`)."""
        from core.pick import candidates, choose, evaluator, repair, resolved, score
        cfg = self.config
        base = self.select_chain(target, results)
        members = list(target.members)
        specs = [self.domain.spec(i) for i in members]
        vocabs = [s.vocabulary(cfg.rules) for s in specs]
        reqs = {i.id: s.requirements for i, s in zip(members, specs)}
        req_lists = [s.requirements for s in specs]
        found = candidates(results, reqs, cfg.planner.pick_top)
        if not found:
            log(f"{target.id}: no valid candidate; keeping chain {base['chain']}")
            return base
        checkers = [self.domain.fast_checker(i, cfg, v) for i, v in zip(members, vocabs)]
        fast = all(c is not None for c in checkers)
        verifiers = [PolicyVerifier(self.domain, i, cfg, self.runner) for i in members]
        if fast:
            evaluate = evaluator(checkers)
        else:
            def evaluate(rules):
                try:
                    return [vf.verify(s.policy(rules, v), analysis=False).worst
                            for vf, s, v in zip(verifiers, specs, vocabs)]
                except (RuleError, PrismError):
                    return None
            log(f"{target.id}: the {self.domain.name} domain has no fast checker: no repair, no resolved scores")
        strict = self.prompts.strict(members)
        actions = list(specs[0].actions)

        def valid(rules) -> bool:
            """`rules.strict`: repair may not bring in what conditions may not name."""
            if not strict:
                return True
            names = set().union(*(re.findall(r"[A-Za-z_]\w*", c) for c, _ in rules))
            return not (names & strict["hide"]) and (strict["literal_sums"] or
                                                     not any(LITERAL_SUM.search(c) for c, _ in rules))
        literals = self.domain.repair_literals(members[0], vocabs[0])
        repairing = bool(cfg.planner.repair and fast)
        early_loss = self._early_scorer(verifiers, specs, vocabs) if cfg.feedback.early else None

        def final(cand: Dict[str, Any], rules) -> Optional[Dict[str, Any]]:
            values = evaluate(rules)
            if values is None:
                return None
            res_score, order = resolved(rules, evaluate, req_lists, actions) if fast else (None, None)
            return {"rules": [list(r) for r in rules], "from": [cand["chain"], cand["iteration"]],
                    "loop_score": list(cand["score"]), "final_score": score(values, req_lists),
                    "resolved_score": res_score, "resolved_order": list(order) if order else None,
                    "repaired": [tuple(r) for r in rules] != [tuple(r) for r in cand["rules"]],
                    **({"early_loss": early_loss(rules).mean} if early_loss and cfg.feedback.early_rank else {})}

        finals, plain = [], []        # plain: with repair, the same candidates as the loops wrote them
        for k, cand in enumerate(found):
            rules = cand["rules"]
            if repairing:
                rules = repair(rules, evaluate, req_lists, literals, actions, cfg.planner.repair,
                               cfg.planner.repair_tries, seed=k, valid=valid)
            f = final(cand, rules)
            if f is None:
                continue
            finals.append(f)
            p = (final(cand, cand["rules"]) if f["repaired"] else f) if repairing else None
            if p is not None:
                plain.append(p)
            log(f"{target.id}: candidate {k + 1} from chain {cand['chain']} round {cand['iteration']}: loop "
                f"{cand['score']}, {'repaired ' if f['repaired'] else ''}{f['final_score']}, "
                f"resolved {f['resolved_score']} {f['resolved_order'] or ''}")
        if not finals:
            return base

        def verified(rules) -> Dict[str, Any]:
            """The exact values of `rules` on every member."""
            exact, checks = [], []
            for vf, s, v in zip(verifiers, specs, vocabs):
                policy = s.policy(rules, v)
                ex, solver = vf.verify_exact(policy) if cfg.prism.exact_check else (None, None)
                if ex is None:
                    ex, solver = vf.verify(policy, analysis=False), ("off" if not cfg.prism.exact_check else "failed")
                exact.append(ex)
                checks.append(solver)
            reqs0 = specs[0].requirements
            return {"success": all(not failing(s.requirements, e.worst) for s, e in zip(specs, exact)),
                    "best_case_success": all(not failing(s.requirements, e.best) for s, e in zip(specs, exact)),
                    "final_best": pessimistic([e.best for e in exact], reqs0),
                    "final_worst": pessimistic([e.worst for e in exact], reqs0),
                    "final_check": ", ".join(dict.fromkeys(checks)),
                    "members": [{"instance": i.id, "success": not failing(s.requirements, e.worst),
                                 "final_best": dict(e.best), "final_worst": dict(e.worst)}
                                for i, s, e in zip(members, specs, exact)]}

        chosen = choose(finals)
        rules = [tuple(r) for r in chosen["rules"]]
        narrowed = None
        if cfg.rules.early_narrow and early_loss:
            if chosen["resolved_order"]:
                rules = fixed_order(rules, chosen["resolved_order"])
            rules, narrowed = self._narrow_rules(rules, evaluate, req_lists, early_loss, log)
        values = verified(rules)
        result = dict(base)
        result.update({k: v for k, v in values.items() if k != "members"})
        result.update({
            "final_rules": [{"condition": c, "action": a} for c, a in rules],
            "members": [{**next((m for m in base.get("members", []) if m["instance"] == row["instance"]), {}), **row}
                        for row in values["members"]],
            "pick": {"chosen": finals.index(chosen), "candidates": finals,
                     **({"narrowed": narrowed} if narrowed is not None else {})},
        })
        log(f"{target.id}: picked candidate {finals.index(chosen) + 1} (from chain {chosen['from'][0]} round "
            f"{chosen['from'][1]}), exact worst {result['final_worst']}")
        if plain:
            # repair's control from the same chains: the pick among the candidates without repair
            alt = choose(plain)
            result["pick"]["norepair"] = {"chosen": plain.index(alt), "candidates": plain,
                                          **verified([tuple(r) for r in alt["rules"]])}
            log(f"{target.id}: without repair, candidate {plain.index(alt) + 1} (from chain {alt['from'][0]} round "
                f"{alt['from'][1]}), exact worst {result['pick']['norepair']['final_worst']}")
        return result

    def _early_scorer(self, verifiers: List[PolicyVerifier], specs, vocabs) -> Callable:
        """The pick's early check (`feedback.early`, core/early.py) of a rule list on the members; the verifier's
        narrowings may name places with numbers."""
        cfg = self.config
        checks = [early.EarlyCheck(vf, cfg.feedback.early, f"instance {vf.instance.id}") for vf in verifiers]
        places = [replace(v, literal_vars=tuple(dict.fromkeys(v.literal_vars + tuple(cfg.feedback.early_group))))
                  for v in vocabs]
        return lambda rules: early.check([(c, s.policy(rules, v)) for c, s, v in zip(checks, specs, places)],
                                         cfg.feedback.states_per_rule, cfg.feedback.early_group)

    def _narrow_rules(self, rules, evaluate, req_lists, early_loss, log) -> Tuple[List[Tuple[str, str]], List[str]]:
        """`rules.early_narrow` on the picked list (`any` already resolved): as `_narrow` in the loop, with the members
        checked by `evaluate` (the domain's fast checker, or PRISM)."""
        from core.pick import score
        k, result, done = self.config.rules.early_narrow, early_loss(rules), []
        for _ in range(k):
            best = None
            for group in [g for g in result.rules if g.move and g.place and g.losing >= NARROW_SHARE * g.decided][:k]:
                place = " & ".join(f"{name} = {value}" for name, value in group.place)
                variant = [(f"({c}) & !({place})" if a == group.move else c, a) for c, a in rules]
                values = evaluate(variant)
                if values is None or score(values, req_lists)[0] < len(req_lists):
                    continue
                r = early_loss(variant)
                if r.mean < (best[1].mean if best else result.mean) - 1e-12:
                    best = (variant, r, f"{group.move} kept away from {place}")
            if best is None:
                break
            log(f"Narrowed {best[2]}: early loss {result.mean:.4f} -> {best[1].mean:.4f}")
            rules, result = best[0], best[1]
            done.append(best[2])
        return rules, done

    def _episode(self, target: Union[Instance, TrainingSet], log: Callable[[str], None]) -> Episode:
        cfg = self.config
        instances = target.members if isinstance(target, TrainingSet) else (target,)
        hidden = target.hidden if isinstance(target, TrainingSet) else ()
        shared = self._shared_variables(instances) if cfg.rules.shared_literals else None
        members = [self._member(instance, f"instance {k}", literal_vars=shared)
                   for k, instance in enumerate(instances, start=1)]
        members += [self._member(instance, f"hidden instance {k}", hidden=True, literal_vars=shared)
                    for k, instance in enumerate(hidden, 1)]
        self._check_compatible(members)
        primary = members[0]
        shown = [m for m in members if not m.hidden]
        answers = list(primary.verifier.spec.actions) + ([ANY] if primary.vocabulary.allow_any else [])
        schema = rule_schema(answers, cfg.planner.max_rules, cfg.planner.max_condition_chars, cfg.prompt.strategy_first)
        chain = target.chain if isinstance(target, TrainingSet) else 0
        return Episode(members, schema, log, TaskFactory(cfg.llm, target.id, chain * CHAIN_CALLS),
                       self.prompts.training_context(shown) if len(shown) > 1 else {},
                       pool=list(target.pool) if isinstance(target, TrainingSet) else [])

    def _shared_variables(self, instances) -> List[str]:
        """The policy variables whose type and range are the same on every one of `instances` (`rules.shared_literals`:
        numbers may be compared with these)."""
        specs = [self.domain.spec(i).variables for i in instances]
        return [v.name for k, v in enumerate(specs[0])
                if all((s[k].type, s[k].low, s[k].high) == (v.type, v.low, v.high) for s in specs)]

    def _member(self, instance: Instance, label: str, hidden: bool = False,
                literal_vars: Optional[List[str]] = None) -> Member:
        """An instance with its verifier, mass analyzer, vocabulary and prompt context (`literal_vars`: the variables
        numbers may be compared with, `rules.shared_literals`)."""
        cfg = self.config
        verifier = PolicyVerifier(self.domain, instance, cfg, self.runner)
        analyzer = MassAnalyzer(verifier, cfg.feedback.horizon_for(self.domain, instance), cfg.feedback.top_k,
                                cfg.feedback.states_per_rule, method=cfg.feedback.blame,
                                seed=zlib.crc32(f"{cfg.llm.seed}/{instance.id}".encode()))
        vocabulary = verifier.spec.vocabulary(cfg.rules, literal_vars)
        scorer = None
        if cfg.rules.any_order != "none" and (cfg.rules.any_scorer == "domain" or cfg.rules.any_prune):
            scorer = self.domain.rule_scorer(instance)
            if scorer is None and cfg.rules.any_scorer == "domain":
                raise ValueError(f"rules.any_scorer=domain: the {self.domain.name} domain has no fast model")
        return Member(label, instance, verifier, analyzer, vocabulary,
                      self.prompts.context(instance, verifier.spec, vocabulary), hidden=hidden, scorer=scorer)

    def _counterexample(self, ep: Episode, kept: Kept, optima: List[Dict[str, float]]) -> Steps[Optional[Kept]]:
        """Once the members pass, check the kept rules on the pool (`run.pool_sets`) in order; the first instance
        where they fail joins the training set (shown to the LLM from the next round on). Returns the kept rule set
        re-checked with the new member, or None if every pool instance passes."""
        raw = [(r.condition, r.action) for r in kept.check.policy.rules]
        for instance in list(ep.pool):
            member = self._member(instance, f"instance {sum(not m.hidden for m in ep.members) + 1}",
                                  literal_vars=ep.primary.vocabulary.literal_vars if self.config.rules.shared_literals
                                  else None)
            self._check_compatible([ep.primary, member])
            policy = member.verifier.spec.policy(raw, member.vocabulary)
            v = yield from member.verifier.verify_steps(policy)
            if not failing(member.requirements, v.worst):
                continue
            ep.pool.remove(instance)
            optimum, _, _ = yield from member.verifier.optimum_steps()
            shown = [m for m in ep.members if not m.hidden]
            optima.insert(len(shown), optimum)                 # optima follow the members' order
            ep.members.insert(len(shown), member)              # shown members stay before the hidden ones
            ep.training = self.prompts.training_context(shown + [member])
            policies, vs = list(kept.check.policies), list(kept.check.vs)
            policies.insert(len(shown), policy)
            vs.insert(len(shown), v)
            check = self._with_early(ep, Check(policies, vs, kept.check.any_report))
            ep.log(f"Counterexample: the rules fail on instance {instance.id}, which joins the training set")
            ep.added.append(instance.id)
            return Kept(check, check.score(ep.members), check.focus(ep.members))
        return None

    @staticmethod
    def _check_compatible(members: List[Member]) -> None:
        """Members of a training set must share the policy variables, actions and requirement names, so one
        rule list and one results table mean the same on each."""
        def shape(m: Member):
            spec = m.verifier.spec
            return ([(v.name, v.type) for v in spec.variables], list(spec.actions), [r.name for r in m.requirements])
        for m in members[1:]:
            if shape(m) != shape(members[0]):
                raise ValueError(f"{m.label} has other variables, actions or requirements than instance 1")

    def _round(self, ep: Episode, mode: str, prompt: str, attempt: int,
               kept: Optional[Kept]) -> Steps[Tuple[Check, Answer]]:
        """Ask for rules and check the candidate on every member: the new rules, or the kept ones followed by
        them when extending. Returns the check and what the LLM calls produced."""
        answer = yield from self._ask(ep, prompt, attempt, mode)
        new = answer.policies or [m.verifier.empty_policy() for m in ep.members]
        policies = ([k.extended(n) for k, n in zip(kept.check.policies, new)] if mode == "extend" else new)
        try:
            policies = yield from self._resolve_any(ep, policies, answer)
        except PrismError as e:
            ep.log(f"Choosing an order for `any` failed ({str(e).splitlines()[0]}); keeping `any`")
        ep.log(f"Candidate policy ({len(policies[0].rules)} rules):\n{policies[0].listing()}")
        try:
            check = self._with_early(ep, (yield from self._check(ep, policies)))
            check.any_report = answer.any_report
            if self.config.rules.early_narrow:
                check = yield from self._narrow(ep, check, answer)
            return check, answer
        except PrismError as e:
            # Rare: PRISM cannot solve this candidate's induced model even with the fallback methods.
            # Count the round as producing nothing (the empty policy) instead of losing the instance.
            reason = str(e).splitlines()[0]
            ep.log(f"Verification failed ({reason}); scoring the round as an empty policy")
            answer.errors.append(f"verification failed: {reason}")
            policies = (kept.check.policies if mode == "extend" and kept
                        else [m.verifier.empty_policy() for m in ep.members])
            return self._with_early(ep, (yield from self._check(ep, policies))), answer

    def _resolve_any(self, ep: Episode, policies: List[SymbolicPolicy], answer: Answer) -> Steps[List[SymbolicPolicy]]:
        """With `rules.any_order`, the candidate's `any` rules resolved into the action order that scores best on
        the members (ties go to the first order); otherwise unchanged."""
        kind = self.config.rules.any_order
        raw = [(r.condition, r.action) for r in policies[0].rules]
        if kind == "none" or not any(a == ANY for _, a in raw):
            return policies
        orders = action_orders(list(ep.primary.verifier.spec.actions), "all" if kind == "report" else kind)
        scored = yield from self._score_orders(ep, policies, raw, orders)
        k = min(scored, key=lambda k: (scored[k][0], k))
        best = (scored[k][0], orders[k], *scored[k][1:])
        if kind == "report":   # feedback only: the rules stay as written, the LLM is told which order would do best
            check = best[3]
            answer.any_report = {"order": list(best[1]), "members": len(ep.members),
                                 "passed": sum(not failing(m.requirements, v.worst) for m, v in zip(ep.members, check.vs)),
                                 "met": sum(len(m.requirements) - len(failing(m.requirements, v.worst))
                                            for m, v in zip(ep.members, check.vs)),
                                 "total": sum(len(m.requirements) for m in ep.members)}
            ep.log(f"Best order for `any` (reported only): {', '.join(best[1])}")
            return policies
        answer.any_order = list(best[1])
        ep.log(f"`any` resolved in the order {', '.join(best[1])}")
        return best[2]

    def _score_orders(self, ep: Episode, policies: List[SymbolicPolicy], raw: List[Tuple[str, str]],
                      orders: List[Tuple[str, ...]]) -> Steps[Dict[int, Tuple[Tuple, List[SymbolicPolicy], Check]]]:
        """Order index -> (score, the rules resolved in that order for each member, their check), for each order that
        can score best. Each order is verified on each member without analysis, except where it resolves the rules
        into the same action as an order already verified there at every state reachable under that order: PRISM
        then builds the same model, so the order takes that order's values. With `rules.any_scorer: domain`, the
        domain's fast model scores every order instead; with `rules.any_prune`, it only rules out the orders that
        cannot score best, and PRISM verifies the rest (all of them if PRISM and the fast model disagree by more
        than PRUNE_SLACK on one). Both only where the fast model scores every order (gridworld: the rules decide
        every reachable state); otherwise PRISM verifies every order."""
        rules = self.config.rules
        fast = None
        if rules.any_scorer == "domain" or (rules.any_prune and all(m.scorer for m in ep.members)):
            fast = [m.scorer.order_values(p, orders) for m, p in zip(ep.members, policies)]
            if any(values is None for f in fast for values in f):
                fast = None
        out = {}
        if rules.any_scorer == "domain" and fast:
            for k, order in enumerate(orders):
                candidate = self._parse(ep, fixed_order(raw, order))
                check = self._with_early(ep, Check(candidate, [Verification(best, worst, None)
                                                               for best, worst in (f[k] for f in fast)]))
                out[k] = (check.score(ep.members), candidate, check)
            return out
        todo = _contenders(ep.members, fast) if fast else list(range(len(orders)))
        choices = [OpenChoices(p) for p in policies]
        verified: List[List[Tuple[Tuple[str, ...], set, Verification]]] = [[] for _ in ep.members]
        checks = 0
        while todo:
            k = todo.pop(0)
            candidate = self._parse(ep, fixed_order(raw, orders[k]))
            vs, off = [], False
            for i, (m, p) in enumerate(zip(ep.members, candidate)):
                v = next((v for order, sets, v in verified[i] if OpenChoices.agree(orders[k], order, sets)), None)
                if v is None:
                    v = yield from m.verifier.verify_steps(p, analysis=False)
                    checks += 1
                    reachable = (m.verifier.policy_valuation(v.result, s) for s in range(len(v.result.states)))
                    verified[i].append((orders[k], choices[i].at(reachable), v))
                    off = off or (fast is not None and _gap(v, *fast[i][k]) > PRUNE_SLACK)
                vs.append(v)
            check = self._with_early(ep, Check(candidate, vs))
            out[k] = (check.score(ep.members), candidate, check)
            if off:
                ep.log(f"PRISM and the fast model differ by more than {PRUNE_SLACK} on the order "
                       f"{', '.join(orders[k])}: verifying every order")
                fast, todo = None, [j for j in range(len(orders)) if j not in out]
        ep.log(f"Orders for `any`: {checks} PRISM checks for {len(orders)} orders on {len(ep.members)} member(s)")
        return out

    def _with_early(self, ep: Episode, check: Check) -> Check:
        """`check` with the rules' loss early in the training games (`feedback.early`), if the members have it."""
        pairs = [(m.early, p) for m, p in zip(ep.members, check.policies) if m.early is not None]
        if pairs:
            check.early = early.check(pairs, self.config.feedback.states_per_rule, self.config.feedback.early_group)
            check.early_ranked = self.config.feedback.early_rank
        return check

    def _narrow(self, ep: Episode, check: Check, answer: Answer) -> Steps[Check]:
        """`rules.early_narrow` (k): when the candidate passes every member, keep the moves that lose most early in the
        games (core/early.py, by rule, place and move) away from their places: of the k such groups with the largest
        loss, try each with `& !(place)` added to every rule that chooses its move (another rule then decides there),
        verified on every member; keep the variant that still passes and loses least early, if it loses less than the
        candidate; repeat up to k times. Only where the move loses in at least NARROW_SHARE of the situations its rule
        decides there: elsewhere it is often the best move. The verifier may name places with numbers whatever the
        LLM may write (`rules.shared_literals`)."""
        k = self.config.rules.early_narrow
        if check.early is None or not self.config.feedback.early_group or not check.passes(ep.members):
            return check
        for _ in range(k):
            raw = [(r.condition, r.action) for r in check.policy.rules]
            best = None
            for group in [g for g in check.early.rules if g.move and g.place
                          and g.losing >= NARROW_SHARE * g.decided][:k]:
                place = " & ".join(f"{name} = {value}" for name, value in group.place)
                variant = [(f"({c}) & !({place})" if a == group.move else c, a) for c, a in raw]
                policies = [m.verifier.spec.policy(variant, replace(m.vocabulary, literal_vars=tuple(dict.fromkeys(
                    m.vocabulary.literal_vars + tuple(self.config.feedback.early_group))))) for m in ep.members]
                candidate = self._with_early(ep, (yield from self._check(ep, policies)))
                if candidate.passes(ep.members) and candidate.early.mean < (best or check).early.mean - 1e-12:
                    best = candidate
                    narrowed = f"{group.move} kept away from {place}"
            if best is None:
                break
            ep.log(f"Narrowed {narrowed}: early loss {check.early.mean:.4f} -> {best.early.mean:.4f}")
            answer.narrowed.append(narrowed)
            check = best
        return check

    def _early_done(self, ep: Episode, check: Check) -> bool:
        """Whether the rules' early loss is within `feedback.early_tolerance` (always, without `feedback.early`)."""
        if check.early is None:
            return True
        ep.log(f"Early loss {check.early.mean:.4f} ({check.early.losing} of {check.early.states} states lose)")
        return check.early.mean <= self.config.feedback.early_tolerance

    @staticmethod
    def _check(ep: Episode, policies: List[SymbolicPolicy]) -> Steps[Check]:
        """Verify each member's policy, one member after another."""
        vs = []
        for m, p in zip(ep.members, policies):
            vs.append((yield from m.verifier.verify_steps(p)))
        return Check(policies, vs)

    # ---------------------------------------------------------------- feedback

    def _robust(self, kept: Kept, members: List[Member]) -> bool:
        """`planner.done_margin`: whether every member passes by more than it (always, at 0)."""
        target = self.config.planner.done_margin
        return target <= 0 or min(margin(v.worst, m.requirements) for m, v in zip(members, kept.check.vs)) > target

    def _feedback(self, ep: Episode, kept: Kept, record: Dict[str, Any],
                  iterations: Optional[List[Dict[str, Any]]] = None) -> Steps[Tuple[str, str]]:
        """Mode and prompt of the next round, from the kept rule set's results on the member that fails worst (with
        `planner.feedback: failures`, on every member, and the complete list is revised)."""
        if self.config.planner.feedback == "failures":
            return "revise", self.prompts.revise(ep, kept, iterations or [])
        member, policy, v = ep.members[kept.focus], kept.check.policies[kept.focus], kept.check.vs[kept.focus]
        reqs = member.requirements
        results = self.prompts.results(policy, v, reqs)
        if kept.check.any_report:
            results["any_report"] = kept.check.any_report
        hidden = self.prompts.hidden_results(ep, kept)
        if hidden:
            results["hidden_results"] = hidden
        if ep.training:
            results.update(training_results=self.prompts.training_results(ep, kept), focus=member.label)
        if kept.check.early is not None and kept.check.passes(ep.members):
            return "early", self.prompts.early(ep, member, kept.check, results)
        failing_best, failing_worst = failing(reqs, v.best), failing(reqs, v.worst)
        # Every shown member passes but a hidden one fails: nothing on the shown members to blame, so ask for a
        # rewrite with the results only (the hidden members' details are never shown)
        if self.config.planner.feedback == "table" or (not failing_worst and hidden and hidden["passed"] < hidden["count"]):
            return "table", self.prompts.table(ep, member, results)
        joint_conflict = (not failing_best and self.config.planner.branch == "joint"
                          and (yield from self._joint_conflict(ep, kept, record)))
        # Every situation covered, yet best and worst differ: rules allow several actions (`any`) and some of
        # them are harmful. Appending rules cannot change covered states, so rewrite them.
        open_choices = bool(failing_worst) and v.uncovered_situations == 0
        early_shown = kept.check if self.config.feedback.early_refine and kept.check.early is not None else None
        if failing_best or joint_conflict or open_choices:
            return "refine", self.prompts.refine(ep, member, policy, v, failing_best or failing_worst,
                                                 joint_conflict, results, early_shown)
        return "extend", self.prompts.extend(ep, member, v, failing_worst, results, policy, early_shown)

    def _joint_conflict(self, ep: Episode, kept: Kept, record: Dict[str, Any]) -> Steps[bool]:
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
