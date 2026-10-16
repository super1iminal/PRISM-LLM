# Pac-Man beyond the training horizons (Oct 16)

Training horizons 14, 16, 18 (ids 3-5); held out 8, 10, 12, 20, 25, 30 (ids 0-2, 6, 7, 9); certified: worst-case crash
<= 0.56 (optimum 0.5511 at every horizon). Validation numbers only.

**Why solved lists stop at 18.** On the explicit models (src/baselines/explicit_mdp.py), all the extra crash mass of
the solved core lists at 20 comes from one move at step 12: up at (3,4), into the dead end (3,5)-(1,6), an 8-move
round trip. Games of 14-18 moves end before Pac-Man comes back out, so training never punishes it. At 25 and 30 more
of the same kind add up: long corridors entered late ((3,1)-left, (3,4)-left, (10,4)-up). A step-free strategy reaches
the optimum at 20, 25 and 30, but it is not compact: trees on 14-18 optimal actions without `steps` never certify 25.
1-2-3-Go!'s raw tree is exactly optimal at 20; at 25 its loss accrues at steps 19-23, beyond the step counts it was
trained on, where its `steps` thresholds extrapolate badly.

**What fixes 20.** The states of the first half of the training games (any play reaches them; the optimal-action data
1-2-3-Go! learns from) show the dead-end move losing in 22 of 24 situations at (3,4) (`feedback.early`, src/core/early.py).
`rules.early_narrow` lets the verifier keep such a move away from its place (`& !(xP = 3 & yP = 4)` on every rule that
chooses it) when it loses in at least 3/4 of the situations there and every training member still passes. Haiku given
the same information as feedback (`haiku_pac_early*`) overwrote whole places and broke training instead.

| configuration | runs | certified of 6 (mean, range) | crash 20 / 25 / 30 (mean) | sets solved | $ per set |
|---|---|---|---|---|---|
| core (`haiku_mi_pacman_long`) | 5 | 2.40 (1-3) | 0.681 / 0.772 / 0.831 | 2/5 | 0.006 |
| core, narrowed post hoc (`src/narrow.py`) | 5 | 2.80 (1-4) | 0.639 / 0.748 / 0.800 | 2/5 | 0.006 |
| 18-round restarts (RQ2 cell) | 5 | 2.80 (2-3) | 0.686 / 0.771 / 0.828 | 4/5 | 0.008 |
| 18-round restarts, narrowed post hoc | 5 | **3.60 (2-4)** | 0.603 / 0.724 / 0.766 | 4/5 | 0.008 |
| maze distances shown (`haiku_pac_closer`) | 3 | 1.00 (1-1) | 0.682 / 0.792 / 0.864 | 0/3 | 0.006 |
| place literals only (`haiku_pac_literals`) | 3 | 2.33 (1-3) | 0.669 / 0.771 / 0.829 | 2/3 | 0.007 |
| early feedback in every prompt (`haiku_pac_early_refine`) | 1 | 3.00 | 0.750 / 0.838 / 0.871 | 0/1 | 0.015 |
| 1-2-3-Go! raw tree, 14-18 | 1 | 4.00 | 0.551 / 0.838 / 0.865 | | |
| 1-2-3-Go! over our features (+ positions, `pacman_tree_features_pos`) | 1 | 2.00 (3.00) | 0.776 / 0.879 / 0.941 (0.653 / 0.777 / 0.807) | | |
| PPO over our features (+ positions, `pacman_ppo_features_pos`) | 3 | 3.00 (3.00) | 0.581 / 0.757 / 0.791 (0.604 / 0.720 / 0.760) | | |

Every narrowed list that solves training certifies 8-20 (20 at 0.5585); the unsolved ones are unchanged. Narrowing
post hoc equals the loop's narrowing for these runs (both stop at the first passing list), except that the loop
also breaks order ties by the early loss. The core runs' narrowed lists are not stored (core runs stay untouched):
`python src/narrow.py <run> && python src/transfer.py <run> --rules narrowed --targets 0-7,9`.

**Pending** (the OpenRouter budget ran out on Oct 16, resets Oct 19): `haiku_pac_narrow_resample` (pure resampling, 30
rounds, narrowing in the loop; about 1 round in 8 of the restarts runs passed training, so 30 rounds should solve
almost every set) and `haiku_pac_narrow` (feedback loop, all 24 orders).
