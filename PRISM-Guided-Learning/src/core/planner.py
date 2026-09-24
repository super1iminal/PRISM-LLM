"""The refinement loop: LLM writes symbolic rules, PRISM checks best/worst case, feedback targets the gap.

Each iteration:
  * worst case meets every threshold  -> done (every completion of the partial policy is safe)
  * best case fails some requirement  -> refine: existing rules are bad; the LLM rewrites the
                                         list, guided by per-rule blame (probability mass)
  * only the worst case fails         -> extend: rules are fine but incomplete; the LLM adds
                                         rules (appended, so existing decisions are unchanged)
                                         for the uncovered states carrying the most mass
The best policy so far is kept (as in the legacy loop). After `stall_limit` iterations without
improvement, the next attempt starts again from the initial prompt (a blind retry).
"""
from dataclasses import dataclass
from time import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, create_model

from core.analysis import MassAnalyzer
from core.domain import Domain, Instance, Requirement
from core.llm import OllamaLLM
from core.prism import PrismRunner
from core.rules import RuleError, SymbolicPolicy
from core.verifier import PolicyVerifier, Verification


@dataclass
class PlannerConfig:
    max_attempts: int = 5       # LLM generation rounds (1 initial + feedback rounds)
    max_fixups: int = 2         # extra calls per round when the answer has invalid rules
    stall_limit: int = 2        # rounds without improvement before a fresh retry
    horizon: int = 100          # steps for the occupancy in the mass analysis
    top_k: int = 10             # hotspots / rules shown in feedback


def rule_schema(actions: List[str]) -> type[BaseModel]:
    rule = create_model(
        "Rule",
        condition=(str, Field(description="Boolean condition over the state variables")),
        action=(Literal.__getitem__(tuple(actions)), Field(description="Action to take")),
    )
    return create_model("RuleList", rules=(List[rule], ...))


def _score(v: Verification, reqs: List[Requirement]) -> Tuple:
    """Lower is better: worst-case failures, best-case failures, then total shortfalls."""
    return (sum(not r.satisfied(v.worst[r.name]) for r in reqs),
            sum(not r.satisfied(v.best[r.name]) for r in reqs),
            sum(r.shortfall(v.worst[r.name]) for r in reqs),
            sum(r.shortfall(v.best[r.name]) for r in reqs))


class SymbolicPlanner:
    def __init__(self, domain: Domain, llm: OllamaLLM, config: Optional[PlannerConfig] = None,
                 runner: Optional[PrismRunner] = None):
        self.domain = domain
        self.llm = llm
        self.config = config or PlannerConfig()
        self.runner = runner or PrismRunner()

    # ---------------------------------------------------------------- prompting

    def _prompt_context(self, instance: Instance, spec) -> Dict[str, Any]:
        return {
            "description": self.domain.description(instance),
            "visual": self.domain.visual(instance),
            "examples": self.domain.examples(instance),
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

    def _ask(self, prompt: str, schema, spec, log) -> Tuple[Optional[SymbolicPolicy], List[str]]:
        """Query the LLM, re-asking with the errors if the rules do not parse."""
        errors: List[str] = []
        current = prompt
        for _ in range(1 + self.config.max_fixups):
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
        log = logger.info
        verifier = PolicyVerifier(self.domain, instance, self.runner)
        analyzer = MassAnalyzer(verifier, self.config.horizon, self.config.top_k)
        spec = verifier.spec
        reqs = spec.requirements
        schema = rule_schema(list(spec.actions))
        base_ctx = self._prompt_context(instance, spec)

        opt_start = time()
        optimum, _, _ = verifier.optimum()
        optimum_seconds = time() - opt_start
        log(f"Unconstrained optimum per requirement: {optimum}")

        iterations: List[Dict[str, Any]] = []
        best: Optional[Tuple[SymbolicPolicy, Verification, Tuple]] = None
        mode, prompt, stall = "initial", self.domain.render("initial.md.j2", instance, **base_ctx), 0

        for attempt in range(1, self.config.max_attempts + 1):
            iter_start = time()
            calls_before = len(self.llm.usage().calls)
            log(f"=== Attempt {attempt}/{self.config.max_attempts} ({mode}) ===\n{prompt}")
            new_rules, errors = self._ask(prompt, schema, spec, log)
            calls = self.llm.usage().calls[calls_before:]
            if new_rules is None:
                new_rules = verifier.empty_policy()
            candidate = best[0].extended(new_rules) if mode == "extend" else new_rules
            log(f"Candidate policy ({len(candidate.rules)} rules):\n{candidate.listing()}")

            v = verifier.verify(candidate)
            score = _score(v, reqs)
            improved = best is None or score < best[2]
            if improved:
                best = (candidate, v, score)
            log("Results: " + ", ".join(f"{r.name}: best={v.best[r.name]:.4f} worst={v.worst[r.name]:.4f}"
                                        for r in reqs))
            log(f"Score {score} ({'new best' if improved else 'no improvement, keeping best'})")

            iterations.append({
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
                "invalid_answers": errors,
                "llm_calls": len(calls),
                "llm_time": sum(c.seconds for c in calls),
                "llm_server_time": sum(c.server_seconds for c in calls),
                "llm_output_tokens": sum(c.output_tokens for c in calls),
                "llm_prompt_tokens": sum(c.prompt_tokens for c in calls),
                "prism_time": v.seconds,
                "prompt": prompt,
                "raw_outputs": [c.raw_output for c in calls],
            })

            best_policy, best_v, _ = best
            failing_best = [r for r in reqs if not r.satisfied(best_v.best[r.name])]
            failing_worst = [r for r in reqs if not r.satisfied(best_v.worst[r.name])]
            if not failing_worst:
                log(f"Success after {attempt} attempts")
                iterations[-1]["iteration_time"] = time() - iter_start
                break

            if attempt < self.config.max_attempts:
                stall = 0 if improved else stall + 1
                results_ctx = self._results_context(best_policy, best_v, reqs)
                if stall >= self.config.stall_limit:
                    mode, stall = "initial", 0
                    prompt = self.domain.render("initial.md.j2", instance, **base_ctx)
                elif failing_best:
                    mode = "refine"
                    blame = analyzer.rule_blame(best_v, failing_best)
                    blame_ctx = [{
                        "rule": b.rule, "mass": b.mass,
                        "text": (f"{best_policy.rules[b.rule].condition} (action {best_policy.rules[b.rule].action})"
                                 if b.rule is not None else ""),
                        "states": [self.domain.format_state(h.valuation) for h in b.hotspots],
                    } for b in blame]
                    prompt = self.domain.render("refine.md.j2", instance, **base_ctx, **results_ctx,
                                                failing=[r.name for r in failing_best], blame=blame_ctx)
                else:
                    mode = "extend"
                    hotspots = analyzer.uncovered_hotspots(best_v, failing_worst)
                    hot_ctx = [{"mass": h.mass, "state": self.domain.format_state(h.valuation)} for h in hotspots]
                    prompt = self.domain.render("extend.md.j2", instance, **base_ctx, **results_ctx,
                                                failing=[r.name for r in failing_worst], hotspots=hot_ctx)
            iterations[-1]["iteration_time"] = time() - iter_start

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
