"""What the LLM is shown: the refinement loop's prompts, rendered from the templates in core/templates (or a
domain's own) with an episode's context (core/episode.py). The planner decides which prompt comes next;
this module only builds them.
"""
from typing import Any, Callable, Dict, List, Optional

from config import Config
from core.analysis import failing_trace
from core.domain import Domain, Instance, Requirement, Spec, failing
from core.episode import Check, Episode, Kept, Member
from core.rules import SymbolicPolicy, Vocabulary, feature_values
from core.verifier import Verification


class Prompts:
    def __init__(self, domain: Domain, config: Config):
        self.domain = domain
        self.config = config
        self._feature_values: Dict[int, Callable] = {}   # id(member) -> its `feature_values` function

    # ---------------------------------------------------------------- context

    def context(self, instance: Instance, spec: Spec, vocabulary: Vocabulary) -> Dict[str, Any]:
        """The context every template of one instance shares: the problem, the vocabulary, the prompt options."""
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

    @staticmethod
    def training_context(members: List[Member]) -> Dict[str, Any]:
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
        """The first round's prompt, also sent when the loop restarts."""
        return self.render(ep, "initial.md.j2")

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
        if id(member) not in self._feature_values:
            self._feature_values[id(member)] = feature_values(spec.variables, list(spec.actions), vocab)
        values = self._feature_values[id(member)](valuation)

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

    def _merged(self, member: Member, hotspots) -> List[Dict[str, Any]]:
        """Extend hotspots named by `_state`; states that read the same are merged, their masses summed."""
        merged: Dict[str, float] = {}
        for h in hotspots:
            text = self._state(member, h.valuation)
            merged[text] = merged.get(text, 0.0) + h.mass
        return [{"mass": m, "state": t} for t, m in merged.items()]

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
        } for b in (member.analyzer.rule_blame(v, blamed) if kind != "none" else [])]
        return self.render(ep, "refine.md.j2", member, **results, failing=[r.name for r in blamed], blame=blame,
                           with_cost=any(r.reward is not None for r in blamed), joint_conflict=joint_conflict,
                           show_blame=kind != "none", blame_kind=kind, **self._trace(member, v, blamed, policy),
                           **({"early": self._early(member, early)} if early else {}))

    def extend(self, ep: Episode, member: Member, v: Verification, failing_worst: List[Requirement],
               results: Dict[str, Any], early: Optional[Check] = None) -> str:
        """Add rules, shown the uncovered states where the worst case loses the most probability."""
        kind = self.config.feedback.blame
        hotspots = member.analyzer.uncovered_hotspots(v, failing_worst) if kind != "none" else []
        return self.render(ep, "extend.md.j2", member, **results, failing=[r.name for r in failing_worst],
                           hotspots=self._merged(member, hotspots),
                           with_cost=any(r.reward is not None for r in failing_worst),
                           show_blame=kind != "none", blame_kind=kind, **self._trace(member, v, failing_worst),
                           **({"early": self._early(member, early)} if early else {}))

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
        return {"trace": {"requirement": req.name, "adversarial": adversarial, "steps": [{
            "state": self._state(member, s.valuation),
            "move": (f"rule {s.rule + 1} chose {s.action}" if s.rule is not None and s.rule < len(rules)
                     else f"{s.action} (no rule decides here)" if s.action else "a move the policy does not choose"),
            "p": s.probability, "failing": s.failing} for s in steps],
            "end": self._state(member, steps[-1].reached), "certain": steps[-1].failing >= 1.0 - 1e-9}}
