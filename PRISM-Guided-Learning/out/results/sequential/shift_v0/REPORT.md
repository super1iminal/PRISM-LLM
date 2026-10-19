# Sequential adaptation study `shift_v0`

Config: `configs/sequential/shift_v0.yaml`. Seed library: `out/results/core/port_fam_history/seed_none_1` (rule lists with `any` resolved). Generation: condition `port_fam_history` on `sequential_v0.csv`. Policy: retain the current rule list if certified, else switch to the certified library list with the largest smallest margin, else generate (mode `growing`) on the instance plus the closest list's training instances, else defer. Certification: interval iteration; within `prism.exact_epsilon` of a threshold, or from the Gauss-Seidel fallback, is inconclusive. Validation instances only.

## Decisions per sequence

| mode | sequence | episodes | served | retain | switch | generate | defer | generations | generations certified | library at end | LLM $ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen | s1 | 12 | 10 | 9 | 1 | 0 | 2 | 0 | 0 | 3 | 0.0 |
| frozen | s2 | 12 | 12 | 9 | 3 | 0 | 0 | 0 | 0 | 3 | 0.0 |
| frozen | s3 | 12 | 9 | 8 | 1 | 0 | 3 | 0 | 0 | 3 | 0.0 |
| growing | s1 | 12 | 12 | 9 | 1 | 2 | 0 | 2 | 2 | 5 | 0.151 |
| growing | s2 | 12 | 12 | 9 | 3 | 0 | 0 | 0 | 0 | 3 | 0.0 |
| growing | s3 | 12 | 10 | 8 | 1 | 1 | 2 | 3 | 1 | 4 | 0.211 |

## Certification outcomes

| mode | certified | failed |
|---|---|---|
| frozen | 33 | 23 |
| growing | 36 | 26 |

Solvers used: interval iteration: 116, Gauss-Seidel 1e-12: 2

## Cost per decision (mean seconds)

| mode | decision | episodes | checks | certify_s | total_s | llm_s | prism_s |
|---|---|---|---|---|---|---|---|
| frozen | defer | 5.0 | 3.0 | 49.7 | 49.7 |  |  |
| frozen | retain | 26.0 | 1.0 | 4.0 | 4.0 |  |  |
| frozen | switch | 5.0 | 3.0 | 10.5 | 10.5 |  |  |
| growing | defer | 2.0 | 3.0 | 75.4 | 675.4 |  |  |
| growing | generate | 3.0 | 3.3 | 13.0 | 728.9 |  |  |
| growing | retain | 26.0 | 1.0 | 3.5 | 3.5 |  |  |
| growing | switch | 5.0 | 3.0 | 9.7 | 9.7 |  |  |
| growing | (generation runs) | 5.0 |  |  | 666.0 | 475.6 | 304.5 |

## Episodes

### frozen s1

| episode | instance | family | n | decision | deployed | checks | training | new_outcome | library_size | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 16 | rooms | 10 | switch | seed0 | 3 |  |  | 3 | 10.2 |
| 1 | 76 | crossing | 10 | retain | seed0 | 1 |  |  | 3 | 3.6 |
| 2 | 142 | adjacent | 11 | defer |  | 3 |  |  | 3 | 11.5 |
| 3 | 136 | adjacent | 10 | retain | seed0 | 1 |  |  | 3 | 3.6 |
| 4 | 192 | random | 12 | defer |  | 3 |  |  | 3 | 13.9 |
| 5 | 193 | random | 12 | retain | seed0 | 1 |  |  | 3 | 3.4 |
| 6 | 194 | random | 12 | retain | seed0 | 1 |  |  | 3 | 4.1 |
| 7 | 195 | random | 12 | retain | seed0 | 1 |  |  | 3 | 4.2 |
| 8 | 196 | random | 13 | retain | seed0 | 1 |  |  | 3 | 3.7 |
| 9 | 197 | random | 13 | retain | seed0 | 1 |  |  | 3 | 5.5 |
| 10 | 198 | random | 13 | retain | seed0 | 1 |  |  | 3 | 6.2 |
| 11 | 199 | random | 13 | retain | seed0 | 1 |  |  | 3 | 3.7 |

### frozen s2

| episode | instance | family | n | decision | deployed | checks | training | new_outcome | library_size | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 17 | rooms | 10 | switch | seed0 | 3 |  |  | 3 | 10.6 |
| 1 | 77 | crossing | 10 | retain | seed0 | 1 |  |  | 3 | 3.8 |
| 2 | 137 | adjacent | 10 | retain | seed0 | 1 |  |  | 3 | 3.7 |
| 3 | 181 | random | 9 | retain | seed0 | 1 |  |  | 3 | 3.4 |
| 4 | 184 | random | 10 | retain | seed0 | 1 |  |  | 3 | 3.5 |
| 5 | 187 | random | 10 | retain | seed0 | 1 |  |  | 3 | 4.0 |
| 6 | 190 | random | 11 | retain | seed0 | 1 |  |  | 3 | 4.4 |
| 7 | 21 | rooms | 11 | retain | seed0 | 1 |  |  | 3 | 3.7 |
| 8 | 182 | random | 9 | switch | seed2 | 3 |  |  | 3 | 9.4 |
| 9 | 185 | random | 10 | switch | seed0 | 3 |  |  | 3 | 10.3 |
| 10 | 188 | random | 11 | retain | seed0 | 1 |  |  | 3 | 4.1 |
| 11 | 191 | random | 11 | retain | seed0 | 1 |  |  | 3 | 3.3 |

### frozen s3

| episode | instance | family | n | decision | deployed | checks | training | new_outcome | library_size | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 80 | crossing | 11 | switch | seed0 | 3 |  |  | 3 | 11.9 |
| 1 | 140 | adjacent | 11 | retain | seed0 | 1 |  |  | 3 | 3.9 |
| 2 | 24 | rooms | 12 | retain | seed0 | 1 |  |  | 3 | 4.2 |
| 3 | 180 | random | 9 | defer |  | 3 |  |  | 3 | 11.1 |
| 4 | 183 | random | 9 | defer |  | 3 |  |  | 3 | 198.4 |
| 5 | 186 | random | 10 | retain | seed0 | 1 |  |  | 3 | 3.2 |
| 6 | 189 | random | 11 | retain | seed0 | 1 |  |  | 3 | 3.7 |
| 7 | 84 | crossing | 12 | retain | seed0 | 1 |  |  | 3 | 3.7 |
| 8 | 144 | adjacent | 12 | retain | seed0 | 1 |  |  | 3 | 4.2 |
| 9 | 28 | rooms | 13 | retain | seed0 | 1 |  |  | 3 | 4.5 |
| 10 | 192 | random | 12 | defer |  | 3 |  |  | 3 | 13.5 |
| 11 | 195 | random | 12 | retain | seed0 | 1 |  |  | 3 | 3.9 |

### growing s1

| episode | instance | family | n | decision | deployed | checks | training | new_outcome | library_size | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 16 | rooms | 10 | switch | seed0 | 3 |  |  | 3 | 10.5 |
| 1 | 76 | crossing | 10 | retain | seed0 | 1 |  |  | 3 | 3.6 |
| 2 | 142 | adjacent | 11 | generate | gen_t2 | 3 | [0, 4, 8, 142] | certified | 4 | 740.3 |
| 3 | 136 | adjacent | 10 | retain | gen_t2 | 1 |  |  | 4 | 3.3 |
| 4 | 192 | random | 12 | generate | gen_t4 | 4 | [60, 64, 68, 192] | certified | 5 | 674.6 |
| 5 | 193 | random | 12 | retain | gen_t4 | 1 |  |  | 5 | 3.1 |
| 6 | 194 | random | 12 | retain | gen_t4 | 1 |  |  | 5 | 3.6 |
| 7 | 195 | random | 12 | retain | gen_t4 | 1 |  |  | 5 | 3.6 |
| 8 | 196 | random | 13 | retain | gen_t4 | 1 |  |  | 5 | 3.2 |
| 9 | 197 | random | 13 | retain | gen_t4 | 1 |  |  | 5 | 4.4 |
| 10 | 198 | random | 13 | retain | gen_t4 | 1 |  |  | 5 | 4.6 |
| 11 | 199 | random | 13 | retain | gen_t4 | 1 |  |  | 5 | 3.4 |

### growing s2

| episode | instance | family | n | decision | deployed | checks | training | new_outcome | library_size | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 17 | rooms | 10 | switch | seed0 | 3 |  |  | 3 | 9.7 |
| 1 | 77 | crossing | 10 | retain | seed0 | 1 |  |  | 3 | 3.1 |
| 2 | 137 | adjacent | 10 | retain | seed0 | 1 |  |  | 3 | 3.2 |
| 3 | 181 | random | 9 | retain | seed0 | 1 |  |  | 3 | 3.1 |
| 4 | 184 | random | 10 | retain | seed0 | 1 |  |  | 3 | 3.3 |
| 5 | 187 | random | 10 | retain | seed0 | 1 |  |  | 3 | 3.4 |
| 6 | 190 | random | 11 | retain | seed0 | 1 |  |  | 3 | 3.6 |
| 7 | 21 | rooms | 11 | retain | seed0 | 1 |  |  | 3 | 3.3 |
| 8 | 182 | random | 9 | switch | seed2 | 3 |  |  | 3 | 8.7 |
| 9 | 185 | random | 10 | switch | seed0 | 3 |  |  | 3 | 8.9 |
| 10 | 188 | random | 11 | retain | seed0 | 1 |  |  | 3 | 3.7 |
| 11 | 191 | random | 11 | retain | seed0 | 1 |  |  | 3 | 3.0 |

### growing s3

| episode | instance | family | n | decision | deployed | checks | training | new_outcome | library_size | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 80 | crossing | 11 | switch | seed0 | 3 |  |  | 3 | 10.4 |
| 1 | 140 | adjacent | 11 | retain | seed0 | 1 |  |  | 3 | 3.4 |
| 2 | 24 | rooms | 12 | retain | seed0 | 1 |  |  | 3 | 3.7 |
| 3 | 180 | random | 9 | defer |  | 3 | [0, 4, 8, 180] | failed | 3 | 652.9 |
| 4 | 183 | random | 9 | defer |  | 3 | [0, 4, 8, 183] | failed | 3 | 698.0 |
| 5 | 186 | random | 10 | retain | seed0 | 1 |  |  | 3 | 3.1 |
| 6 | 189 | random | 11 | retain | seed0 | 1 |  |  | 3 | 3.7 |
| 7 | 84 | crossing | 12 | retain | seed0 | 1 |  |  | 3 | 3.6 |
| 8 | 144 | adjacent | 12 | retain | seed0 | 1 |  |  | 3 | 3.6 |
| 9 | 28 | rooms | 13 | retain | seed0 | 1 |  |  | 3 | 3.8 |
| 10 | 192 | random | 12 | generate | gen_t10 | 3 | [60, 64, 68, 192] | certified | 4 | 771.7 |
| 11 | 195 | random | 12 | retain | gen_t10 | 1 |  |  | 4 | 3.8 |
