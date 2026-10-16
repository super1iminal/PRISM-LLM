# Fresh transfer (branch claude/fresh-transfer-5c4595): results

Certified validation members per list (shared `transfer.py`, ids 16-35 / 76-95 / 136-155, 20 per family),
3 independent runs (`final/run1..3`). The ablations reuse the same LLM answers and change only the offline steps;
lists that the ablation leaves unchanged keep their certification (`ablation/` holds the rest, each certified on
its own family's 20 members).

| variant | rooms | crossing | adjacent | mean |
|---|---|---|---|---|
| frozen method: repair + `any` resolved | 19, 19, 19 (19.00) | 20, 20, 20 (20.00) | 19, 20, 19 (19.33) | 19.44 (19.33-19.67) |
| repair, `any` left open | 16, 17, 16 (16.33) | 20, 20, 20 (20.00) | 19, 17, 19 (18.33) | 18.22 (18.00-18.33) |
| no repair, `any` resolved | 19, 9, 19 (15.67) | 9, 20, 20 (16.33) | 19, 20, 19 (19.33) | 17.11 (15.67-19.33) |
| no repair, `any` left open | 16, 8, 16 (13.33) | 2, 16, 16 (11.33) | 19, 17, 19 (18.33) | 14.33 (12.33-17.00) |
| PPO / decision trees / current LLM | 19 / 2 / 8.4 | 0 / 19 / 4.6 | 19 / 16 / 16.0 | 12.67 / 12.33 / 9.67 |

Repair changed the kept list in 4 of 9 cases (run 1 crossing, run 2 rooms and crossing, run 3 crossing).
Cost per repeat: rooms $0.45-0.51, crossing $0.49-0.59, adjacent $0.09-0.18 (Haiku via OpenRouter; 1.0-1.5M tokens
per family, mostly reasoning); the ablations cost nothing extra.

Pac-Man (`pm/trial2_strict`, `pm/run2`, `pm/run3`): 3, 4, 3 of 6 validation horizons (to beat: 4); every kept list
reaches the training optimum 0.5511 on horizons 14, 16, 18.
