# Case studies (domains)

Each directory here is one case study. The planner in `src/core` knows nothing about any
particular domain. Everything domain-specific lives in these files:

| file | purpose |
|---|---|
| `domain.py` | exactly one subclass of `core.domain.Domain`. It implements `load_instances(dataset)` (dataset → list of `Instance`) and `context(instance)`, the variables that every template below is rendered with. |
| `model.prism.j2` | **The MDP, fully specified in PRISM** (`mdp` model type): state variables, initial state, commands, transition probabilities and labels. Every policy action must be the action label of the commands it enables (`[up] guard -> 0.7 : ... + 0.3 : ...;`). The planner adds a `policy` module that synchronizes on these labels. Exactly one initial state. |
| `spec.yaml.j2` | `actions` (label → English), `variables` (the **policy-visible** state variables, with a description. Types and ranges are read from the model unless given as `type`/`low`/`high`), and `requirements` (`name`, PRISM path `formula`, `bound` `>=` or `<=`, `threshold`, English `description`). |
| `description.md.j2` | English description of the MDP for the prompt: states, dynamics, transition probabilities. |
| `visual.txt.j2` | Visual representation of the state space for the prompt. |
| `examples.md.j2` | *(optional)* Example rules in this domain's vocabulary. |
| `initial.md.j2`, `refine.md.j2`, `extend.md.j2`, `_problem.md.j2`, … | *(optional)* Override the core prompt templates in `src/core/templates` for this domain only. |

All templates are Jinja2 (`StrictUndefined`, `trim_blocks`, `lstrip_blocks`).

## What the planner does with a domain

1. Renders the prompt from the description, visual, variables, actions, requirements and examples.
2. The LLM answers with ordered rules `{"condition": ..., "action": ...}`. Conditions are boolean
   expressions over the policy-visible variables, and the first matching rule decides.
3. The rules compile to a PRISM `policy` module. In states no rule covers, every action stays
   enabled, so PRISM's `Pmax`/`Pmin` give the best and worst case over all completions of the policy.
4. The refinement loop (see `src/core/planner.py`) refines rules when even the best case fails,
   extends them when only the worst case fails, and stops when the worst case meets every threshold.

## Adding a case study

```
domains/<name>/
  domain.py          # class MyDomain(Domain): load_instances, context
  model.prism.j2
  spec.yaml.j2
  description.md.j2
  visual.txt.j2
  data/...           # datasets, loaded by your load_instances
```

Then run `python src/run_symbolic.py --domain <name> --data <dataset>`.

Tips:
- Choose the policy-visible variables deliberately. Hidden variables (such as the gridworld
  obstacle phase `obs_idx`) are still part of the MDP, but rules cannot condition on them.
- Keep the policy-visible state space small enough to enumerate (≤ 200k valuations). The
  compiler enumerates it to simplify first-match guards, and falls back to longer guards beyond that.
- `format_state(valuation)` can be overridden to change how states appear in feedback.

## gridworld

The existing case study: sequential goals, static obstacles, one moving obstacle, 0.7/0.15/0.15
slip dynamics. It mirrors the legacy DTMC exactly (verified by `src/regression.py` and
`tests/test_gridworld_equivalence.py`). `legacy_translate.py` converts legacy per-cell policies
into atomic rules.

## uuv

Pipeline inspection by an underwater vehicle, from Päßler et al., "Formal Modelling and Analysis of
a Self-Adaptive Robotic System" (iFM 2023, [arXiv:2308.14663](https://arxiv.org/abs/2308.14663)).
The MDP is the paper's ProFeat model (artifact branch `scp-ifm_artifact` of
[remaro-network/auv_profeat](https://github.com/remaro-network/auv_profeat)) rewritten in plain PRISM.
The policy plays the paper's managing subsystem: while searching, it picks the altitude
(`low`/`med`/`high`) within what the water visibility allows. All other transitions are forced
(`[step]`, which the policy module does not synchronize on). The bare MDP reproduces every number the
paper reports (`tests/test_uuv.py`). `data/uuv_paper.csv` holds the paper's two scenarios, and
`data/calibrate.py` prints the achievable range of each requirement and some reference policies.
