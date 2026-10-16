"""What the LLM is shown: the refinement loop's prompts, rendered from the templates in core/templates (or a
domain's own) with an episode's context (core/episode.py). The planner decides which prompt comes next;
this module only builds them.
"""
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from config import Config
from core.analysis import failing_trace
from core.domain import Domain, Instance, Requirement, Spec, failing
from core.episode import Check, Episode, Kept, Member, margin
from core.failures import action_lines, allowed_here, choice_label, failure_states, lost_sets
from core.optimal import Strategy, at, demonstration, deviations
from core.rules import SymbolicPolicy, Vocabulary, feature_values
from core.verifier import Verification

OPTIMAL_LEGEND = ("`optimal` names what the best strategy for this instance (computed by the model checker) does in "
                  "that state, and `not` the action your rules take there instead{features}. Use it to see what tells "
                  "a good action from a bad one, and write general rules from that, not rules for single states.")
OPTIMAL_FEATURES = "; brackets list the features that hold for each of these actions"


class Prompts:
    def __init__(self, domain: Domain, config: Config):
        self.domain = domain
        self.config = config
        self._feature_values: Dict[int, Callable] = {}   # id(member) -> its `feature_values` function
        self._strategies: Dict[int, Strategy] = {}       # id(member) -> its optimal strategy (core/optimal.py)
        self._demos: Dict[int, Dict[str, Any]] = {}      # id(member) -> the demonstration on it
        self._lost: Dict[int, Any] = {}                   # id(member) -> its lost sets (core/failures.py)

    # ---------------------------------------------------------------- context

    def strict(self, instances: Sequence[Instance]) -> Optional[Dict[str, Any]]:
        """With `rules.strict`, what conditions may not name on any of `instances` (`Domain.strict`); else None."""
        if not self.config.rules.strict:
            return None
        parts = [self.domain.strict(i) for i in instances]
        return {"hide": set().union(*(p["hide"] for p in parts)), "literal_sums": all(p["literal_sums"] for p in parts),
                "note": parts[0]["note"]}

    def context(self, instance: Instance, spec: Spec, vocabulary: Vocabulary) -> Dict[str, Any]:
        """The context every template of one instance shares: the problem, the vocabulary, the prompt options."""
        prompt = self.config.prompt
        strict = self.strict([instance])
        hide = strict["hide"] if strict else set()
        return {
            "description": self.domain.description(instance),
            "visual": self.domain.visual(instance),
            "examples": self.domain.examples(instance, extended=vocabulary.allow_any) if prompt.examples else "",
            "catch_all_instruction": prompt.catch_all_instruction,
            "variables": [v for v in spec.variables if v.name not in hide],
            "actions": spec.actions,
            "requirements": spec.requirements,
            "constants": [c for c in vocabulary.constants if c.name not in hide],
            **({"strict_note": strict["note"]} if strict and strict["note"] and not vocabulary.literal_vars else {}),
            **({"worst_case": True, "worst_case_notes": self.domain.worst_case_notes(instance)}
               if prompt.worst_case else {}),
            "features": vocabulary.features,
            "any_action": vocabulary.allow_any,
            "general": vocabulary.general,
            **({"literal_vars": list(vocabulary.literal_vars)} if vocabulary.literal_vars else {}),
            **({"generality": True} if prompt.generality else {}),
            **({"tie_break": True} if prompt.tie_break else {}),
            **({"any_order": self.config.rules.any_order, "action_list": ", ".join(spec.actions)}
               if self.config.rules.any_order != "none" and vocabulary.allow_any else {}),
            **({"hint": prompt.hint} if prompt.hint else {}),
            **({"strategy_first": True,
                "answer_format": '{"strategy": "...", "rules": [{"condition": "...", "action": "..."}, ...]}'}
               if prompt.strategy_first else {}),
        }

    def training_context(self, members: List[Member]) -> Dict[str, Any]:
        """How the prompt describes a training set: every member, and every variable, with each member's range
        or description where they differ (with `rules.strict`, without the names conditions may not use)."""
        def per_member(values: List[str]) -> str:
            return values[0] if len(set(values)) == 1 else "; ".join(
                f"{v} on {m.label}" for m, v in zip(members, values))
        strict = self.strict([m.instance for m in members])
        hide = strict["hide"] if strict else set()
        variables = []
        for i, var in enumerate(members[0].verifier.spec.variables):
            if var.name in hide:
                continue
            versions = [m.verifier.spec.variables[i] for m in members]
            variables.append({"name": var.name, "type": per_member([v.describe_type() for v in versions]),
                              "description": per_member([v.description or "" for v in versions])})
        return {"members": [{"label": m.label.capitalize(), "description": m.context["description"],
                             "visual": m.context["visual"],
                             "constants": [c for c in m.vocabulary.constants if c.name not in hide]} for m in members],
                "variables": variables}

    @staticmethod
    def results(policy: SymbolicPolicy, v: Verification, reqs: List[Requirement]) -> Dict[str, Any]:
        """The results table and the rule listing of a checked rule set."""
        rows = []
        for r in reqs:
            best_ok, worst_ok = r.satisfied(v.best[r.name]), r.satisfied(v.worst[r.name])
            status = "ok" if worst_ok else ("fails in worst case" if best_ok else "FAILS even in best case")
            rows.append({"name": r.name, "best": v.best[r.name], "worst": v.worst[r.name],
                         "bound": ">=" if r.maximize else "<=", "threshold": r.threshold, "status": status})
        return {"rules_listing": policy.listing(), "num_rules": len(policy.rules), "results": rows,
                "reachable": v.reachable_situations, "uncovered": v.uncovered_situations}

    @staticmethod
    def hidden_results(ep: Episode, kept: Kept) -> Optional[Dict[str, int]]:
        """How the rules do on the hidden members: how many there are and how many pass; None without any."""
        hidden = [(m, v) for m, v in zip(ep.members, kept.check.vs) if m.hidden]
        if not hidden:
            return None
        return {"count": len(hidden), "passed": sum(not failing(m.requirements, v.worst) for m, v in hidden)}

    @staticmethod
    def training_results(ep: Episode, kept: Kept) -> List[Dict[str, Any]]:
        """Per member of a training set: requirements met in the worst case and the failing ones."""
        rows = []
        for m, v in zip(ep.members, kept.check.vs):
            if m.hidden:
                continue
            fails = failing(m.requirements, v.worst)
            rows.append({"label": m.label.capitalize(), "met": len(m.requirements) - len(fails),
                         "total": len(m.requirements), "failing": [r.name for r in fails]})
        return rows

    # ---------------------------------------------------------------- prompts

    def render(self, ep: Episode, template: str, member: Optional[Member] = None, **extra) -> str:
        """`template` for `member` (default: the first), with every member described for a training set."""
        member = member or ep.primary
        return self.domain.render(template, member.instance, **member.context, training=ep.training, **extra)

    def initial(self, ep: Episode) -> str:
        """The first round's prompt, also sent when the loop restarts; with `prompt.demonstration`, it shows one
        optimal run on the first member."""
        demo = self.demonstration(ep) if self.config.prompt.demonstration else None
        return self.render(ep, "initial.md.j2", **({"demonstration": demo} if demo and demo["steps"] else {}))

    def invalid(self, prompt: str, errors: str) -> str:
        """`prompt` again, with why the previous answer could not be parsed."""
        return self.domain.render("invalid.md.j2", prompt=prompt, errors=errors)

    def _state(self, member: Member, valuation: Dict[str, Any]) -> str:
        """How feedback names a state: the domain's rendering of its valuation, or (`feedback.space: features`) its
        boolean variables, the state features (by name if boolean and true, as `name=value` if integer) and, per
        action, the action features that hold. Integer variables (positions) are left out."""
        if self.config.feedback.space != "features":
            return self.domain.format_state(valuation)
        spec, vocab = member.verifier.spec, member.vocabulary
        values = self._values(member)(valuation)

        flags = [f"{k}={str(v).lower()}" for k, v in valuation.items() if isinstance(v, bool)]
        for f in vocab.features:
            value = values.get((f.name, None))
            if isinstance(value, bool):
                flags += [f.name] if value else []
            elif value is not None:
                flags.append(f"{f.name}={value}")
        moves = [f"{a}: {', '.join(f.name for f in vocab.features if f.is_action_feature and values[(f.name, a)]) or 'none'}"
                 for a in spec.actions]
        return " & ".join(flags) + (" | " if flags else "") + "; ".join(moves)

    def _values(self, member: Member) -> Callable:
        """The member's `feature_values` function, built once."""
        if id(member) not in self._feature_values:
            spec = member.verifier.spec
            self._feature_values[id(member)] = feature_values(spec.variables, list(spec.actions), member.vocabulary)
        return self._feature_values[id(member)]

    def _merged(self, member: Member, hotspots, v: Optional[Verification] = None) -> List[Dict[str, Any]]:
        """Extend hotspots named by `_state`; states that read the same are merged, their masses summed. With
        `feedback.optimal` (and `v`), each also says what the optimal strategy does there (in its heaviest state)."""
        merged: Dict[str, List] = {}
        for h in hotspots:
            text = self._state(member, h.valuation)
            merged.setdefault(text, [0.0, h])[0] += h.mass
        out = [{"mass": m, "state": t} for t, (m, _) in merged.items()]
        if v is not None and self.config.feedback.optimal:
            for row, (_, h) in zip(out, merged.values()):
                row["note"] = self._note(member, v, h.state, [])
        return out

    # ---------------------------------------------------------------- the optimal strategy (core/optimal.py)

    def _strategy(self, member: Member) -> Strategy:
        """The member's optimal strategy (`Domain.optimal_strategy`), computed once."""
        if id(member) not in self._strategies:
            strategy = self.domain.optimal_strategy(member.instance, self.config)
            if strategy is None:
                raise ValueError(f"feedback.optimal and prompt.demonstration: the {self.domain.name} domain has no "
                                 "optimal strategy")
            self._strategies[id(member)] = strategy
        return self._strategies[id(member)]

    def _action_features(self, member: Member, valuation: Dict[str, Any], actions: Sequence[str]) -> str:
        """Per action, the action features that hold for it (`right: closer, toward; down: none`); '' without any."""
        names = [f.name for f in member.vocabulary.features if f.is_action_feature]
        if not names:
            return ""
        values = self._values(member)(valuation)

        def holding(a: str) -> str:
            return ", ".join(n if values[(n, a)] is True else f"{n}={values[(n, a)]}" for n in names
                             if values[(n, a)] is not False) or "none"
        return "; ".join(f"{a}: {holding(a)}" for a in actions)

    def _note(self, member: Member, v: Verification, s: Optional[int], chosen: Sequence[str]) -> str:
        """What the feedback adds about state `s` of `v`, where the rules take the actions `chosen` (none where no
        rule decides): what the optimal strategy does there (`feedback.optimal`), and which of `chosen` change nothing
        there (`feedback.stuck`)."""
        idle = self._idle(v, s, chosen) if self.config.feedback.stuck and s is not None else []
        if self.config.feedback.optimal:
            return self._optimal_note(member, v, s, chosen, idle)
        return f"{' or '.join(idle)} changes nothing here, so the run stays in this state" if idle else ""

    @staticmethod
    def _idle(v: Verification, s: int, actions: Sequence[str]) -> List[str]:
        """The `actions` whose every choice in state `s` of `v` leaves the state unchanged (a self-loop)."""
        loops: Dict[str, bool] = {}
        for c in (v.result.choices[s] if v.result.choices else []):
            if c.action in actions:
                loops[c.action] = loops.get(c.action, True) and all(t == s for t, _ in c.successors)
        return [a for a in actions if loops.get(a)]

    def _optimal_note(self, member: Member, v: Verification, s: Optional[int], chosen: Sequence[str],
                      idle: Sequence[str] = ()) -> str:
        """What the optimal strategy does in state `s` of `v`, against the actions the rules take there (`chosen`;
        none where no rule decides): `optimal: down, not right [right: closer; down: toward]`, saying of the rules'
        actions in `idle` that they change nothing. '' where the strategy does not know the state or finds every
        action optimal."""
        if s is None:
            return ""
        info = at(self._strategy(member), v.result, s)
        actions = list(member.verifier.spec.actions)
        if info is None or not info.actions or len(info.actions) == len(actions):
            return ""
        best = " or ".join(info.actions)
        wrong = [a for a in chosen if a not in info.actions]
        if chosen and not wrong:
            return f"optimal: {best}, as the rules"
        named = list(info.actions) + wrong
        features = self._action_features(member, member.verifier.policy_valuation(v.result, s), named)
        bad = " or ".join(f"{a} (it changes nothing here)" if a in idle else a for a in wrong)
        return f"optimal: {best}" + (f", not {bad}" if wrong else "") + (f" [{features}]" if features else "")

    def _situation(self, member: Member, valuation: Dict[str, Any]) -> str:
        """The features that hold in a state: each state feature (by name if boolean and true, `name=value` if
        integer), and per action feature the actions it holds for (`wall: up, left; closer: right`)."""
        values = self._values(member)(valuation)
        actions = list(member.verifier.spec.actions)
        parts = []
        for f in member.vocabulary.features:
            if f.is_action_feature:
                held = [a for a in actions if values[(f.name, a)] is not False]
                parts += [f"{f.name}: {', '.join(held)}"] if held else []
            elif values[(f.name, None)] is not False:
                parts.append(f.name if values[(f.name, None)] is True else f"{f.name}={values[(f.name, None)]}")
        return "; ".join(parts)

    def demonstration(self, ep: Episode) -> Dict[str, Any]:
        """One optimal run on the first member (`core.optimal.demonstration`): per decision, the state, the features
        that hold there (unless feedback names states by their features already) and the optimal action(s)."""
        member = ep.primary
        if id(member) not in self._demos:
            steps = demonstration(member.verifier, self._strategy(member))
            plain, every = self.config.feedback.space != "features", len(member.verifier.spec.actions)
            self._demos[id(member)] = {"member": member.label if ep.training else "this instance", "steps": [{
                "state": self._state(member, st.valuation),
                "features": self._situation(member, st.valuation) if plain else "",
                "actions": "any action" if len(st.actions) == every else " or ".join(st.actions)} for st in steps]}
        return self._demos[id(member)]

    # ---------------------------------------------------------------- planner.feedback: failures

    def revise(self, ep: Episode, kept: Kept, iterations: List[Dict[str, Any]]) -> str:
        """The complete list revised from every shown member's exact worst-case result and where it fails
        (core/failures.py), with the attempt history."""
        shown = [(m, p, v) for m, p, v in zip(ep.members, kept.check.policies, kept.check.vs) if not m.hidden]
        report = "\n".join(self._member_report(m, p, v, len(shown) > 1) for m, p, v in shown)
        holds = all(not failing(m.requirements, v.worst) for m, _, v in shown)
        early = kept.check.early is not None and holds
        return self.render(ep, "revise.md.j2", ep.primary, rules_listing=kept.check.policy.listing(), report=report,
                           history=self._history(ep, iterations), all_hold=holds,
                           done_margin=self.config.planner.done_margin,
                           **({"early": self._early(ep.primary, kept.check)} if early else {}))

    def _lost_sets(self, member: Member):
        if id(member) not in self._lost:
            self._lost[id(member)] = lost_sets(member.verifier)
        return self._lost[id(member)]

    def _member_report(self, m: Member, policy: SymbolicPolicy, v: Verification, several: bool, top_reqs: int = 3,
                       per_req: int = 2) -> str:
        """One member's part of the revise prompt: certified with its smallest margin, or the failing requirements
        (the `top_reqs` worst) with the `per_req` states where the worst-case run fails or stalls."""
        reqs, label = m.requirements, (m.label.capitalize() if several else "This instance")
        fails = sorted(failing(reqs, v.worst), key=lambda r: r.margin(v.worst[r.name]))
        if not fails:
            return f"### {label}: every requirement holds (smallest margin {margin(v.worst, reqs):+.3f})."
        lines = [f"### {label}: FAILS {', '.join(r.name for r in fails)}"
                 + (f"; {v.uncovered_situations} reachable situations are uncovered" if v.uncovered_situations else "")]
        lost, index = self._lost_sets(m)
        where = np.array([index.get(s, -1) for s in v.result.states])
        rules, actions = policy.rules, list(m.verifier.spec.actions)
        names = [f.name for f in m.vocabulary.features if f.is_action_feature]
        seen = set()
        for r in fails[:top_reqs]:
            lines.append(f"- `{r.name}` = {v.worst[r.name]:.3f} (needs {'>=' if r.maximize else '<='} {r.threshold}).")
            if r.name not in lost:
                continue
            here = np.where(where >= 0, lost[r.name][np.maximum(where, 0)], False)
            for fs in failure_states(v, r, here, m.analyzer.horizon, per_req):
                state = self._state(m, m.verifier.policy_valuation(v.result, fs.state))
                if (fs.state, fs.kind) in seen:
                    lines.append(f"    * the same state as above: {state} ({fs.mass:.3f} of runs)")
                    continue
                seen.add((fs.state, fs.kind))
                j = v.state_rules[fs.state]
                who = (f"rule {j + 1} `{rules[j].condition} -> {rules[j].action}`" if j is not None and j < len(rules)
                       else "NO RULE (uncovered: the worst case may take any action)")
                verb = "fail from here" if fs.kind == "fails" else "are still stuck here at the end"
                took = choice_label(v, fs.state, fs.choice)
                lines.append(f"    * {fs.mass:.3f} of runs {verb}: {state}; decided by {who}; allowed "
                             f"{', '.join(allowed_here(m.verifier, v, fs.state)) or 'none'}; the worst case takes "
                             f"{took if took in actions else 'a move the policy does not choose'}.")
                effects: Dict[str, List[str]] = {}
                for a, text in action_lines(m.verifier, v, fs.state, lost, index, self._values(m), names):
                    effects.setdefault(text, []).append(a)
                lines.append("      actions here: " + "; ".join(f"{', '.join(acts)}: {text}"
                                                                for text, acts in effects.items()))
        return "\n".join(lines)

    @staticmethod
    def _history(ep: Episode, iterations: List[Dict[str, Any]]) -> str:
        """One line per earlier attempt of this loop: how many shown members it passes and its total shortfall, or why
        it was rejected; '' after a single attempt."""
        reqs = {m.instance.id: m.requirements for m in ep.members}
        lines = []
        for it in iterations:
            if it["invalid_answers"] and len(it["invalid_answers"]) >= it["llm_calls"]:
                lines.append(f"- attempt {it['iteration']}: rejected ({it['invalid_answers'][-1].splitlines()[0][:200]})")
                continue
            rows = [r for r in it.get("members", []) if not r.get("hidden")] or [
                {"instance": ep.primary.instance.id, "worst": it["worst"]}]
            passed = sum(not failing(reqs[r["instance"]], r["worst"]) for r in rows)
            short = sum(q.shortfall(r["worst"][q.name]) for r in rows for q in reqs[r["instance"]])
            lines.append(f"- attempt {it['iteration']}: every requirement holds on {passed} of {len(rows)} "
                         f"instance{'s' if len(rows) != 1 else ''}, total shortfall {short:.3f}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def table(self, ep: Episode, member: Member, results: Dict[str, Any]) -> str:
        """Results table and previous rules only, asking for a complete new list (`planner.feedback: table`)."""
        return self.render(ep, "table.md.j2", member, **results)

    def refine(self, ep: Episode, member: Member, policy: SymbolicPolicy, v: Verification,
               blamed: List[Requirement], joint_conflict: bool, results: Dict[str, Any],
               early: Optional[Check] = None) -> str:
        """Rewrite the rules, shown how much of the lost probability each rule's states carry."""
        kind = self.config.feedback.blame
        rules = policy.rules
        blame = [{
            "rule": b.rule, "mass": b.mass,
            "text": f"{rules[b.rule].condition} (action {rules[b.rule].action})" if b.rule is not None else "",
            "states": list(dict.fromkeys(self._state(member, h.valuation) for h in b.hotspots)),
            **({"notes": list(dict.fromkeys(self._noted(member, v, h.valuation, h.state,
                                                        policy.allowed(h.valuation) or [] if b.rule is not None else [])
                                            for h in b.hotspots))} if self._noting() else {}),
        } for b in (member.analyzer.rule_blame(v, blamed) if kind != "none" else [])]
        return self.render(ep, "refine.md.j2", member, **results, failing=[r.name for r in blamed], blame=blame,
                           with_cost=any(r.reward is not None for r in blamed), joint_conflict=joint_conflict,
                           show_blame=kind != "none", blame_kind=kind, **self._trace(member, v, blamed, policy),
                           **self._deviations(member, v, policy), **self._legend(member),
                           **({"early": self._early(member, early)} if early else {}))

    def extend(self, ep: Episode, member: Member, v: Verification, failing_worst: List[Requirement],
               results: Dict[str, Any], policy: Optional[SymbolicPolicy] = None,
               early: Optional[Check] = None) -> str:
        """Add rules, shown the uncovered states where the worst case loses the most probability."""
        kind = self.config.feedback.blame
        hotspots = member.analyzer.uncovered_hotspots(v, failing_worst) if kind != "none" else []
        return self.render(ep, "extend.md.j2", member, **results, failing=[r.name for r in failing_worst],
                           hotspots=self._merged(member, hotspots, v),
                           with_cost=any(r.reward is not None for r in failing_worst),
                           show_blame=kind != "none", blame_kind=kind, **self._trace(member, v, failing_worst),
                           **(self._deviations(member, v, policy) if policy is not None else {}), **self._legend(member),
                           **({"early": self._early(member, early)} if early else {}))

    def _deviations(self, member: Member, v: Verification, policy: SymbolicPolicy) -> Dict[str, Any]:
        """With `feedback.deviations`, the states the rules decide where they leave the optimal strategy most
        (`core.optimal.deviations`), each with what the optimal strategy does there instead."""
        if not self.config.feedback.deviations:
            return {}
        rows = []
        for h in deviations(member.verifier, v, self._strategy(member), member.analyzer.horizon,
                            self.config.feedback.top_k):
            chosen = policy.allowed(h.valuation) or []
            idle = self._idle(v, h.state, chosen) if self.config.feedback.stuck else []
            note = self._optimal_note(member, v, h.state, chosen, idle)
            if ", not " in note:
                rows.append({"mass": h.mass, "text": f"{self._state(member, h.valuation)}: {note}"})
        return {"deviations": rows} if rows else {}

    def _noted(self, member: Member, v: Verification, valuation: Dict[str, Any], s: Optional[int],
               chosen: Sequence[str]) -> str:
        """A state as feedback names it, followed by its note (`_note`)."""
        note = self._note(member, v, s, chosen)
        return self._state(member, valuation) + (f": {note}" if note else "")

    def _noting(self) -> bool:
        return self.config.feedback.optimal or self.config.feedback.stuck

    def _legend(self, member: Member) -> Dict[str, str]:
        """How to read the notes of `feedback.optimal` and `feedback.deviations`."""
        if not (self.config.feedback.optimal or self.config.feedback.deviations):
            return {}
        features = any(f.is_action_feature for f in member.vocabulary.features)
        return {"optimal_legend": OPTIMAL_LEGEND.format(features=OPTIMAL_FEATURES if features else "")}

    def early(self, ep: Episode, member: Member, check: Check, results: Dict[str, Any]) -> str:
        """The rules pass every member: show where their moves lose early in the games (`feedback.early`)."""
        return self.render(ep, "early.md.j2", member, **results, early=self._early(member, check))

    def _early(self, member: Member, check: Check) -> Dict[str, Any]:
        """Where the checked rules' moves lose early in the games (core/early.py), as the templates show it."""
        result, rules = check.early, check.policy.rules
        top = [{
            "rule": r.rule, "losing": r.losing, "decided": r.decided, "mean": r.total / max(r.losing, 1),
            "text": f"{rules[r.rule].condition} (action {rules[r.rule].action})" if r.rule is not None else "",
            "place": " & ".join(f"{k} = {v}" for k, v in r.place), "move": r.move,
            "best": ", ".join(f"{a} (in {n})" for a, n in sorted(r.best.items(), key=lambda t: -t[1])),
            "examples": [{"member": e.member, "state": self._state(member, e.valuation), "moves": list(e.moves),
                          "worst": e.worst,
                          "best": list(e.best_moves), "loss": e.loss,
                          "reqs": [{"name": n, "after": a, "optimum": o} for n, (a, o) in e.values.items()]}
                         for e in r.examples],
        } for r in result.rules[:self.config.feedback.top_k]]
        return {"mean": result.mean, "losing": result.losing, "states": result.states,
                "clock": self.config.feedback.early, "rules": top}

    def _trace(self, member: Member, v: Verification, failing_reqs: List[Requirement],
               policy: Optional[SymbolicPolicy] = None) -> Dict[str, Any]:
        """With `feedback.trajectory`, a likely way to fail the failing requirement with the largest shortfall:
        under the best case's choices if even the best case fails it, else under the worst case's."""
        if not self.config.feedback.trajectory or not failing_reqs:
            return {}
        req = max(failing_reqs, key=lambda r: r.shortfall(v.worst[r.name]))
        adversarial = req.satisfied(v.best[req.name])
        steps = failing_trace(member.verifier, v, req, adversarial)
        if not steps:
            return {}
        rules = (policy or member.verifier.empty_policy()).rules

        def move(s) -> str:
            text = (f"rule {s.rule + 1} chose {s.action}" if s.rule is not None and s.rule < len(rules)
                    else f"{s.action} (no rule decides here)" if s.action else "a move the policy does not choose")
            if not self._noting() or s.action not in member.verifier.spec.actions or not v.decisions[s.state]:
                return text
            note = self._note(member, v, s.state, [s.action])
            return text + (f"; {note}" if note else "")
        return {"trace": {"requirement": req.name, "adversarial": adversarial, "steps": [{
            "state": self._state(member, s.valuation), "move": move(s),
            "p": s.probability, "failing": s.failing} for s in steps],
            "end": self._state(member, steps[-1].reached), "certain": steps[-1].failing >= 1.0 - 1e-9}}
