| scenario | policy | P(no thruster failure) | P(done in time) | E[energy] to done | E[time] to done | size |
|---|---|---|---|---|---|---|
| North Sea | paper controller (range) | 0.654..0.674 | 0.516..0.818 | 24.78..44.39 | 23.66..32.40 | >= 1620 states |
| North Sea | ours (qwen3:14b) | 0.672 | 0.806 | 26.04 | 23.93 | 25 rules |
| Caribbean Sea | paper controller (range) | 0.321..0.355 | 0.064..0.868 | 59.08..4723.29 | 55.54..1315.58 | >= 8850 states |
| Caribbean Sea | ours (qwen3:14b) | 0.348 | 0.860 | 62.99 | 55.88 | 25 rules |
| Caribbean Sea | ours, North Sea rules reused | 0.347 | 0.824 | 98.52 | 73.92 | 25 rules |

Paper controller rows give its min..max over all resolutions (energy/time = the paper's Table 2). Policy rows are exact (fully covering policies: best = worst).
