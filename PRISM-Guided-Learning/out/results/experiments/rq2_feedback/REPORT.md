# RQ2a: feedback, restarts, or both

Each round either refines the rule list with the verifier's feedback or starts over from the initial prompt; the best list so far is kept either way.

Reference: **Feedback, no restarts (core)**. Certified: held-out targets certified per rule list (mean over runs, range over runs). Δ: difference of means to the reference, with a 90% bootstrap interval over runs (few runs: read it as a rough guide). Ties open: certified with the action orders left open, i.e. however ties are broken. Library: targets certified by at least one of a run's lists.

## Random grids

Targets 12-19 (each list's own training instances left out); certified of 8.

| condition | code | runs | certified | Δ vs reference | ties open | library | sets solved | rounds / set | output tokens / set |
|---|---|---|---|---|---|---|---|---|---|
| Feedback, no restarts (core) | F | 5 | 6.25 (5.50–7.00) | reference | 5.50 (5.00–6.25) | 8.0 | 10% (0%–25%) | 4.9 (4.5–5.0) | 11,315 (8,570–15,009) |
| Feedback, restart when a round gains < 0.05 | F+R gain | 3 | 5.92 (5.75–6.25) | -0.33 [-0.75, +0.08] | 4.75 (4.50–5.25) | 7.3 (7.0–8.0) | 0% | 5.0 | 5,105 (4,376–5,622) |
| Feedback, restart after 2 rounds without gain | F+R stall | 3 | 6.25 (6.00–6.50) | +0.00 [-0.42, +0.42] | 5.00 (4.75–5.50) | 7.7 (7.0–8.0) | 17% (0%–25%) | 4.7 (4.2–5.0) | 10,073 (6,541–14,468) |
| Restarts only (no feedback) | R | 3 | 5.42 (5.25–5.50) | -0.83 [-1.22, -0.45] | 4.42 (4.25–4.50) | 6.3 (6.0–7.0) | 0% | 5.0 | 3,298 (3,192–3,383) |

## Gridworld families

Targets 16-35,76-95,136-155 (each list's own training instances left out); certified of 20 in its own group.

| condition | code | runs | certified | Δ vs reference | ties open | library | sets solved | rounds / set | output tokens / set |
|---|---|---|---|---|---|---|---|---|---|
| Feedback, no restarts (core) | F | 5 | 9.67 (5.67–15.00) | reference | – | 33.8 (24.0–56.0) | 40% (33%–67%) | 3.9 (3.7–4.3) | 9,947 (7,662–15,685) |
| Feedback, restart when a round gains < 0.05 | F+R gain | 3 | 8.22 (7.33–9.33) | -1.44 [-3.91, +0.84] | 2.00 (0.00–3.00) | 29.7 (27.0–31.0) | 22% (0%–33%) | 4.1 (3.7–5.0) | 5,048 (4,747–5,584) |
| Feedback, restart after 2 rounds without gain | F+R stall | 3 | 6.78 (5.00–10.00) | -2.89 [-6.02, +0.31] | 1.78 (0.33–4.33) | 26.7 (19.0–31.0) | 33% | 4.4 (4.0–4.7) | 9,239 (8,350–9,781) |
| Restarts only (no feedback) | R | 3 | 9.78 (8.67–11.00) | +0.11 [-2.40, +2.40] | 0.22 (0.00–0.33) | 32.3 (27.0–38.0) | 22% (0%–33%) | 4.1 (3.7–5.0) | 2,941 (2,653–3,471) |

Per group (certified per rule list on its own group, mean over runs):

| condition | rooms | crossing | adjacent |
|---|---|---|---|
| Feedback, no restarts (core) | 8.4 | 4.6 | 16.0 |
| Feedback, restart when a round gains < 0.05 | 11.3 | 0.0 | 13.3 |
| Feedback, restart after 2 rounds without gain | 8.0 | 0.7 | 11.7 |
| Restarts only (no feedback) | 10.0 | 4.7 | 14.7 |

## Pac-Man (trained on horizons 14, 16, 18; 5 held out: 8, 10, 12, 20, 25)

Targets 0-7 (each list's own training instances left out); certified of 5.

| condition | code | runs | certified | Δ vs reference | ties open | library | sets solved | rounds / set | output tokens / set |
|---|---|---|---|---|---|---|---|---|---|
| Feedback, no restarts (core) | F | 3 | 3.00 | reference | – | 3.0 | 67% (0%–100%) | 4.0 (3.0–5.0) | 4,649 (3,555–6,328) |
| Feedback, restart when a round gains < 0.05 | F+R gain | 0 | – | – | – | – | – | – | – |
| Feedback, restart after 2 rounds without gain | F+R stall | 0 | – | – | – | – | – | – | – |
| Restarts only (no feedback) | R | 0 | – | – | – | – | – | – | – |

![RQ2a: feedback, restarts, or both](rq2_feedback.png)
