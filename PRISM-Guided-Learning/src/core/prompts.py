"""What the LLM is shown: the refinement loop's prompts, rendered from the templates in core/templates (or a
domain's own) with an episode's context (core/episode.py). The planner decides which prompt comes next;
this module only builds them.
"""
from typing import Any, Dict, List, Optional

from config import Config
from core.domain import Domain, Instance, Requirement, Spec, failing
from core.episode import Episode, Kept, Member
from core.rules import SymbolicPolicy, Vocabulary
from core.verifier import Verification


class Prompts:
    def __init__(self, domain: Domain, config: Config):
        self.domain = domain
        self.config = config

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
    def training_results(ep: Episode, kept: Kept) -> List[Dict[str, Any]]:
        """Per member of a training set: requirements met in the worst case and the failing ones."""
        rows = []
        for m, v in zip(ep.members, kept.check.vs):
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

    def table(self, ep: Episode, member: Member, results: Dict[str, Any]) -> str:
        """Results table and previous rules only, asking for a complete new list (`planner.feedback: table`)."""
        return self.render(ep, "table.md.j2", member, **results)

    def refine(self, ep: Episode, member: Member, policy: SymbolicPolicy, v: Verification,
               blamed: List[Requirement], joint_conflict: bool, results: Dict[str, Any]) -> str:
        """Rewrite the rules, shown how much of the lost probability each rule's states carry."""
        kind = self.config.feedback.blame
        rules = policy.rules
        blame = [{
            "rule": b.rule, "mass": b.mass,
            "text": f"{rules[b.rule].condition} (action {rules[b.rule].action})" if b.rule is not None else "",
            "states": [self.domain.format_state(h.valuation) for h in b.hotspots],
        } for b in (member.analyzer.rule_blame(v, blamed) if kind != "none" else [])]
        return self.render(ep, "refine.md.j2", member, **results, failing=[r.name for r in blamed], blame=blame,
                           with_cost=any(r.reward is not None for r in blamed), joint_conflict=joint_conflict,
                           show_blame=kind != "none", blame_kind=kind)

    def extend(self, ep: Episode, member: Member, v: Verification, failing_worst: List[Requirement],
               results: Dict[str, Any]) -> str:
        """Add rules, shown the uncovered states where the worst case loses the most probability."""
        kind = self.config.feedback.blame
        hotspots = member.analyzer.uncovered_hotspots(v, failing_worst) if kind != "none" else []
        return self.render(ep, "extend.md.j2", member, **results, failing=[r.name for r in failing_worst],
                           hotspots=[{"mass": h.mass, "state": self.domain.format_state(h.valuation)}
                                     for h in hotspots],
                           with_cost=any(r.reward is not None for r in failing_worst),
                           show_blame=kind != "none", blame_kind=kind)
