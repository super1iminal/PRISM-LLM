| scenario | policy | P(no thruster failure) | P(done in time) | E[energy] to done | E[time] to done | meets every requirement | size |
|---|---|---|---|---|---|---|---|
| North Sea | paper controller (range) | 0.654..0.674 | 0.516..0.818 | 24.78..44.39 | 23.66..32.40 | — | >= 1620 states |
| North Sea | ours (anthropic/claude-sonnet-5.5) | 0.672 | 0.807 | 26.01 | 23.92 | yes | 3 rules |
| Caribbean Sea | paper controller (range) | 0.321..0.355 | 0.064..0.868 | 59.08..4723.29 | 55.54..1315.58 | — | >= 8850 states |
| Caribbean Sea | ours (anthropic/claude-sonnet-5.5) | 0.347 | 0.824 | 98.52 | 73.92 | no | 6 rules |
| Caribbean Sea | ours, North Sea rules reused | 0.342 | 0.846 | 84.00 | 67.46 | no | 3 rules |

Paper controller rows give its min..max over all resolutions (energy/time = the paper's Table 2). Policy rows are exact (fully covering policies: best = worst). Reward requirements: North Sea energy_budget <= 26.5; Caribbean Sea energy_budget <= 62.5.
