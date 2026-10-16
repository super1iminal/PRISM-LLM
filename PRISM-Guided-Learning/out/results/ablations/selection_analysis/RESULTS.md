# Choosing the frozen list from training values only (validation)

Certified: held-out validation targets certified per frozen list (own group's targets for families). Every rule
below uses training values only. Scripts in this directory; caches they write are not committed.

## 1. The action order on ties (post hoc, no LLM)

Every finished list of the runs that resolve `any` in the loop, all 24 orders, certified with the numpy model
(`order_cache.py`, `order_eval.py`). `first`: the loop's rule (best keep-best score, first order on ties). The others
break the score ties by: `min` the smallest worst-case margin over members and requirements, `sum` the summed
margins, `large` the smallest margin on the largest member, `leximin`, `rel` (margins relative to 1 - threshold).
`tied_mean`: a random order among the tied ones; `oracle`: the best order on the targets.

| benchmark | lists | solved | tied orders | final | first | min | sum | large | leximin | rel | large_lex | tied_mean | all_mean | oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| families/adjacent | 63 | 81% | 11.6 | 14.25 | 14.05 | 13.79 | 13.95 | 13.71 | 13.79 | 13.48 | 13.73 | 13.93 | 10.11 | 15.65 |
| families/crossing | 63 | 3% | 2.7 | 2.37 | 2.37 | 2.40 | 2.40 | 2.40 | 2.40 | 2.40 | 2.40 | 2.31 | 1.27 | 3.38 |
| families/rooms | 63 | 6% | 3.2 | 7.73 | 7.62 | 7.67 | 7.70 | 7.73 | 7.75 | 7.67 | 7.73 | 8.05 | 4.93 | 9.19 |
| grids | 77 | 8% | 5.4 | 6.22 | 6.14 | 6.08 | 6.12 | 6.04 | 6.03 | 6.08 | 6.04 | 6.09 | 6.16 | 7.00 |
| grids_noclose | 20 | 0% | 2.1 | 3.40 | 3.45 | 3.45 | 3.40 | 3.45 | 3.45 | 3.45 | 3.45 | 3.35 | 3.47 | 4.55 |

No tie-break beats `first`; the oracle is 1-1.6 above, but no training value finds it. No flag.

## 2. Best of K independent lists (post hoc, PRISM transfer CSVs)

Lists of the standard three-member conditions per benchmark (`list_table.py`, `bestofk.py`), Monte Carlo over
K-subsets. Margin: the smallest worst-case margin over training members and requirements (exact values; >= 0 iff
the set is solved).

| benchmark | lists | one list | best of 5 by keep-best score | best of 5 by margin | best of 5 by largest-member margin |
|---|---|---|---|---|---|
| families/adjacent | 53 | 14.13 | 13.74 | 16.67 | 16.80 |
| families/crossing | 53 | 2.21 | 4.11 | 6.53 | 3.90 |
| families/rooms | 53 | 7.79 | 9.12 | 9.11 | 11.12 |
| grids | 216 | 5.96 | 6.37 | 6.52 | 6.66 |
| grids_noclose | 20 | 3.40 | 4.07 | 4.11 | 3.74 |
| mer | 5 | 2.40 | 3.00 | 3.00 | 3.00 |
| mer_n | 5 | 2.40 | 6.00 | 6.00 | 6.00 |
| pacman | 19 | 2.53 | 2.99 | 2.99 | 2.99 |

MER rows: pools of 5, so best of 5 is the whole pool. The keep-best score cannot separate solved lists (adjacent).
Families, margin, by K (numpy, 189 lists): K=1 8.2, 5 11.7, 10 13.5, 15 14.6. Selecting among every round's
candidate instead of each chain's kept list (`cand_cache.py`, `cand_eval.py`): +0.2 at most.

## 3. `planner.chains: 5` runs (Haiku 5.5, PRISM transfer)

| benchmark | condition | runs | certified per list | core (5 runs) | best baseline |
|---|---|---|---|---|---|
| MER across n | haiku_mi_mer_n_chains5 | 3 | 2.00 (0, 0, 6) | 2.40 (0-6) | 6 (1-2-3-Go! raw trees) |
| MER across x | haiku_mi_mer_chains5 | 3 | 3.00 (3, 3, 3) | 2.40 (0-3) | 6 (1-2-3-Go!, one-instance strategy) |
| families | haiku_fam_order_chains5 | 2 | 11.83 (9.67, 14.00*) | 9.67 (5.67-15.00) | 12.67 (PPO, our features) |

Families per run (rooms / crossing / adjacent): run 1 12 / 0 / 17, run 2 15 / 8 / 19 (*numpy; its PRISM transfer was still queued for PRISM slots). Within these runs (numpy, every
chain's kept list, `chain_numpy.py`): the chosen chain certifies 13.5 / 4.0 / 18.0, a single chain 8.1 / 1.5 / 13.0,
the best chain in hindsight 13.5 / 5.0 / 18.0; the margin picked the best chain in 5 of 6 set-runs. The better of
the two runs' chosen lists per set (= 10 chains): 12 / 8 / 19, 13.0.

Not run: the third families run, families with 10 chains, grids with and without `closer`: the OpenRouter workspace
budget ran out on Oct 16 at ~02:20 (403 until Oct 19). A partial grids-without-`closer` run (3 of 4 sets: 4, 3, 5
certified, numpy) was removed from the store.
