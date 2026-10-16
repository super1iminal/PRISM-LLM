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
chooses it) when it loses in at least 3/4 of the situations there and every training member still passes. Two
caveats from the runs: the early loss must not also break ties between action orders (`feedback.early_rank: false`;
with it, both `haiku_pac_narrow_fb` runs took the down-first rotation, the one where narrowing misses 20), and the LLM
should keep the core prompt (telling it that it may name places, `haiku_pac_narrow_resample`, coincided with training
passing in 1 of 100 resampled rounds). Haiku given the same information as feedback (`haiku_pac_early*`) did not fix 20.

| configuration | runs | certified of 6 (mean, range) | crash 20 / 25 / 30 (mean) | sets solved | $ per set |
|---|---|---|---|---|---|
| core (`haiku_mi_pacman_long`) | 5 | 2.40 (1-3) | 0.681 / 0.772 / 0.831 | 2/5 | 0.006 |
| **core + narrowing (`haiku_pac_narrow_plain`)** | 5 | **3.60 (2-4)** | 0.602 / 0.749 / 0.790 | 4/5 | 0.009 |
| core, narrowed post hoc (`src/narrow.py`) | 5 | 2.80 (1-4) | 0.639 / 0.748 / 0.800 | 2/5 | 0.006 |
| 18-round restarts (RQ2 cell), narrowed post hoc | 5 | 3.60 (2-4) | 0.603 / 0.724 / 0.766 | 4/5 | 0.008 |
| narrowing, early loss ranks orders (`haiku_pac_narrow_fb`) | 2 | 3.00 (3-3) | 0.577 / 0.676 / 0.736 | 2/2 | 0.008 |
| narrowing, resampling, LLM may name places (`haiku_pac_narrow_resample`) | 5 | 1.60 (1-3) | 0.685 / 0.791 / 0.845 | 1/5 | 0.023 |
| early feedback only, after training passes (`haiku_pac_early`) | 3 | 2.00 (1-3) | 0.716 / 0.823 / 0.881 | 1/3 | 0.014 |
| early feedback in every prompt (`haiku_pac_early_refine`) | 1 | 3.00 | 0.750 / 0.838 / 0.871 | 0/1 | 0.015 |
| maze distances shown (`haiku_pac_closer`) | 3 | 1.00 (1-1) | 0.682 / 0.792 / 0.864 | 0/3 | 0.006 |
| place literals only (`haiku_pac_literals`) | 3 | 2.33 (1-3) | 0.669 / 0.771 / 0.829 | 2/3 | 0.007 |
| core + narrowing, 20 rounds (`haiku_pac_narrow_plain20`, stopped) | 3 | 3.00 (2-4) | 0.668 / 0.766 / 0.813 | 2/3 | 0.022 |
| port (`port_pacman`, `any` resolved by the pick's order) | 3 | 3.00 (3-3) | 0.662 / 0.749 / 0.813 | 3/3 | 0.104 |
| port, its lists narrowed post hoc (`src/narrow.py --rules determinized`) | 3 | 4.00 (4-4) | 0.559 / 0.690 / 0.735 | 3/3 | 0.104 |
| **port + narrowing (`port_pacman_narrow_core`)** | 3 | **4.00 (4-4)** | 0.559 / 0.690 / 0.735 | 3/3 | 0.098 |
| port + place literals + narrowing (`port_pacman_narrow`) | 3 | 3.00 (1-4) | 0.600 / 0.724 / 0.774 | 2/3 | 0.101 |
| port + place literals + early feedback (`port_pacman_early`, `any` resolved) | 3 | 3.00 (3-3) | 0.637 / 0.735 / 0.789 | 3/3 | 0.106 |
| 1-2-3-Go! raw tree, 14-18 | 1 | 4.00 | 0.551 / 0.838 / 0.865 | | |
| 1-2-3-Go! over our features (+ positions, `pacman_tree_features_pos`) | 1 | 2.00 (3.00) | 0.776 / 0.879 / 0.941 (0.653 / 0.777 / 0.807) | | |
| PPO over our features (+ positions, `pacman_ppo_features_pos`) | 3 | 3.00 (3.00) | 0.581 / 0.757 / 0.791 (0.604 / 0.720 / 0.760) | | |

With `haiku_pac_narrow_plain`, every run that solves training certifies 8-20 (training solved in rounds 8, 1, 9, 6;
the fifth run not within 10 and certifies 2); the narrowed lists' crash at 25/30 is 0.69/0.74 (one 0.82/0.85), against
0.84/0.87 for 1-2-3-Go!'s raw tree. Nothing trained on 14-18 certifies 25. `haiku_pac_narrow_core` (core prompt,
resampling) is one screening run that never passed training in 30 rounds (certifies 1). The post hoc rows apply
`src/narrow.py` to frozen lists (the core runs' narrowed lists are not stored; core runs stay untouched).

**With the port.** `port_pacman` (the ported fresh method: 8 chains with failure feedback, pick with repair) solves
training in every run but certifies 3 of 6: with `any` resolved by the pick's order, every list takes the dead end at
(3,4). Narrowing the picked list (`port_pacman_narrow_core`: the port's prompts unchanged, the pick resolves `any` and
the verifier narrows) certifies 4 in every run, as does narrowing the three `port_pacman` lists post hoc; every
narrowed list makes the same moves (up kept away from (3,4)). As specified, with place literals in the prompt too
(`port_pacman_narrow`), every run that solves training certifies 4; training was solved in 0, 2 and 2 of 8 chains,
while the twin below, with the same prompts until training passes, solved 3, 2 and 5 (Haiku's variance, not the
literals). Given the same early check as feedback instead (`port_pacman_early`), Haiku writes its own place rules
(down at (3,4) in all three runs), which fix the move whatever the ghosts do: 3 of 6 in every run.
