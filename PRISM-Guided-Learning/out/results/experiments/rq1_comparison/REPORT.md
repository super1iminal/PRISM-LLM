# RQ1: LLM-written rule lists against the baselines (validation)

How many held-out instances does one frozen rule list certify, per method, and how many does a library of a run's lists cover? Our cells use the core configuration (the verifier resolves `any` into the best action order on the training members each round, no restarts, the generality instruction).

Certified: held-out targets certified per rule list (on its own group's targets where there are groups). Every target: the same over all targets, whatever the group. Library: targets certified by at least one of a run's (or baseline's) lists. Ties open: certified with the action orders left open, i.e. however ties are broken. Ours: mean over runs, range in parentheses.

**Best configuration per method** (✓, the only one in the figures): per benchmark, the method's configuration with the highest mean certified per list on these validation targets; ties: the larger library, then the lower cost. Only fully certified configurations compete. The test pass uses the configurations chosen here (`chosen.csv`) and never re-chooses on test results.

## Random grids (grid_20_large; trained on sets of three 9x9-13x13 grids, held out: 12-19)

Targets 12-19: 8 held out (each list's own training instances are left out).

| method | configuration | figure | runs | certified | every target | best list | library | ties open | sets solved | $ / set |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours (Haiku 5.5) | **Ours, Haiku 5.5** | ✓ | 5 | 6.25 (5.50–7.00) | 6.25 (5.50–7.00) | 7 (6–8) | 8.0 | 5.50 (5.00–6.25) | 10% (0%–25%) | 0.010 (0.008–0.012) |
| Ours (Sonnet 5.5) | **Ours, Sonnet 5.5** | ✓ | 3 | 6.67 (5.75–7.25) | 6.67 (5.75–7.25) | 8 (7–8) | 8.0 | – | 75% | 0.493 (0.420–0.621) |
| 1-2-3-Go! (reimplemented) | 1-2-3-Go! (reimplemented: trees over raw variables) |  | – | 0.00 | 0.00 | 0 | 0.0 | 0.00 | – | – |
| 1-2-3-Go! (reimplemented) | 1-2-3-Go! with our features | ✓ | – | 3.75 | 3.75 | 7 | 7.0 | 3.75 | – | – |
| Rule-list search | Rule-list search | ✓ | – | 4.50 | 4.50 | 7 | 7.0 | 2.75 | – | – |
| PPO | PPO with our features (1 unchecked) |  | – | – | – | – | – | – | – | – |
| PPO | PPO with a MiniGrid-style local view (1 unchecked) |  | – | – | – | – | – | – | – | – |
| Hand-written template | Hand-written template | ✓ | – | 6.00 | 6.00 | 6 | 6.0 | 6.00 | – | – |

![Random grids (grid_20_large; trained on sets of three 9x9-13x13 grids, held out: 12-19)](rq1_grids.png)

## Random grids without the shortest-path feature `closer`

Targets 12-19: 8 held out (each list's own training instances are left out).

| method | configuration | figure | runs | certified | every target | best list | library | ties open | sets solved | $ / set |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours (Haiku 5.5) | **Ours, Haiku 5.5** | ✓ | 5 | 3.40 (3.00–3.75) | 3.40 (3.00–3.75) | 4 (4–5) | 6.0 (5.0–7.0) | 0.00 | 0% | 0.011 (0.010–0.012) |
| 1-2-3-Go! (reimplemented) | 1-2-3-Go! with our features | ✓ | – | 1.50 | 1.50 | 2 | 3.0 | 1.50 | – | – |
| Rule-list search | Rule-list search | ✓ | – | 1.25 | 1.25 | 2 | 2.0 | 0.50 | – | – |
| Hand-written template | Hand-written template | ✓ | – | 6.00 | 6.00 | 6 | 6.0 | 6.00 | – | – |

![Random grids without the shortest-path feature `closer`](rq1_grids_noclose.png)

## Gridworld families (trained on three 6x6-8x8 members of a family, held out: 10x10-14x14 members)

Targets 16-35,76-95,136-155: 60 held out (each list's own training instances are left out).

| method | configuration | figure | runs | certified | every target | best list | library | ties open | sets solved | $ / set |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours (Haiku 5.5) | **Ours, Haiku 5.5** | ✓ | 5 | 9.67 (5.67–15.00) | 19.20 (9.33–30.67) | 17 (12–19) | 33.8 (24.0–56.0) | – | 40% (33%–67%) | 0.008 (0.007–0.011) |
| Ours (Sonnet 5.5) | **Ours, Sonnet 5.5** | ✓ | 3 | 14.00 (12.67–15.00) | 32.33 (29.00–36.67) | 19 | 54.3 (45.0–59.0) | – | 67% | 0.787 (0.663–0.932) |
| 1-2-3-Go! (reimplemented) | 1-2-3-Go! (reimplemented: trees over raw variables) |  | – | 0.00 | 0.00 | 0 | 0.0 | 0.00 | – | – |
| 1-2-3-Go! (reimplemented) | 1-2-3-Go! with our features | ✓ | – | 12.33 | 20.33 | 19 | 43.0 | 12.33 | – | – |
| Rule-list search | Rule-list search | ✓ | – | 6.00 | 11.00 | 10 | 21.0 | – | – | – |
| PPO | PPO with our features | ✓ | – | 12.67 | 16.67 | 19 | 38.0 | 12.67 | – | – |
| PPO | PPO with a MiniGrid-style local view (1 unchecked) |  | – | – | – | – | – | – | – | – |
| Hand-written template | Hand-written template | ✓ | – | 4.67 | 14.00 | 14 | 14.0 | 4.67 | – | – |

Per group (certified per rule list on its own group):

| configuration | rooms (of 20) | crossing (of 20) | adjacent (of 20) |
|---|---|---|---|
| Ours, Haiku 5.5 | 8.4 | 4.6 | 16.0 |
| Ours, Sonnet 5.5 | 17.7 | 6.7 | 17.7 |
| 1-2-3-Go! (reimplemented: trees over raw variables) | 0.0 | 0.0 | 0.0 |
| 1-2-3-Go! with our features | 2.0 | 19.0 | 16.0 |
| Rule-list search | 10.0 | 7.0 | 1.0 |
| PPO with our features | 19.0 | 0.0 | 19.0 |
| PPO with a MiniGrid-style local view | – | – | – |
| Hand-written template | 0.0 | 0.0 | 14.0 |

![Gridworld families (trained on three 6x6-8x8 members of a family, held out: 10x10-14x14 members)](rq1_families.png)

## Pac-Man (trained on horizons 14, 16, 18; 6 held out: 8, 10, 12, 20, 25, 30)

Targets 0-7,9: 6 held out (each list's own training instances are left out).

| method | configuration | figure | runs | certified | every target | best list | library | ties open | sets solved | $ / set |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours (Haiku 5.5) | **Ours, Haiku 5.5** | ✓ | 5 | 2.40 (1.00–3.00) | 2.40 (1.00–3.00) | 2 (1–3) | 2.4 (1.0–3.0) | 2.40 (1.00–3.00) | 40% (0%–100%) | 0.006 (0.004–0.008) |
| 1-2-3-Go! (reimplemented) | 1-2-3-Go! (reimplemented: trees over raw variables) | ✓ | – | 4.00 | 4.00 | 4 | 4.0 | 4.00 | – | – |
| 1-2-3-Go! (reimplemented) | 1-2-3-Go! trained on horizon 5 (as the paper) |  | – | 2.00 | 2.00 | 2 | 2.0 | 2.00 | – | – |
| 1-2-3-Go! (reimplemented) | 1-2-3-Go! with our features |  | – | 2.00 | 2.00 | 2 | 2.0 | 2.00 | – | – |
| 1-2-3-Go! (reimplemented) | 1-2-3-Go! with our features, trained on horizon 5 |  | – | 0.00 | 0.00 | 0 | 0.0 | 0.00 | – | – |
| PPO | PPO with our features | ✓ | – | 3.00 | 3.00 | 3 | 3.0 | 3.00 | – | – |
| PPO | PPO over raw variables |  | – | 3.00 | 3.00 | 3 | 3.0 | 3.00 | – | – |
| Uniform random policy | Uniform random policy | ✓ | – | 0.00 | 0.00 | 0 | 0.0 | 0.00 | – | – |
| Best of 1000 random policies | Best of 1000 random policies | ✓ | – | 3.00 | 3.00 | 3 | 3.0 | 3.00 | – | – |
| Strategy of one instance | Strategy of horizon 18, by moves made | ✓ | – | 2.00 | 2.00 | 2 | 2.0 | 2.00 | – | – |
| Strategy of one instance | Strategy of horizon 18, by moves left |  | – | 0.00 | 0.00 | 0 | 0.0 | 0.00 | – | – |
| Hand-written maze rules | Hand-written maze rules | ✓ | – | 1.00 | 1.00 | 1 | 1.0 | 1.00 | – | – |

![Pac-Man (trained on horizons 14, 16, 18; 6 held out: 8, 10, 12, 20, 25, 30)](rq1_pacman.png)

Worst-case crash of each method's frozen lists (best configuration) per horizon, up to 30: horizons 50 and 100 are Pac-Man's sealed test set and stay out until the test pass (200: PRISM cannot build it). Line: mean over the lists, band: their range; hollow markers: a list's own training horizons. 1-2-3-Go! (published, different setup): 0.5511 at 5, 0.869 at 25, 0.99998 at 200 (200 beyond the axis).

![Worst-case crash probability per horizon](rq1_pacman_horizon.png)
