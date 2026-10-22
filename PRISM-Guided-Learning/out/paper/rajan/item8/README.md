# Item 8, checkpoint 2: the joint-solvability check for UUV (evidence)

North Sea scenario of `domains/uuv/data/uuv_paper.csv` (deadline 30), rendered by the UUV domain.

- `north_sea.prism`: the model as the domain renders it.
- `north_sea_counter.prism`: a copy with two changes, so that PRISM's multi-objective LP accepts the requirements:
  1. a module `stepcount` (`t : [0..31]`, +1 on every `step`/`low`/`med`/`high` transition, saturating at 31), so
     `F<=30 "done"` becomes the reachability `F ("done" & t<=30)`;
  2. the `energy` reward moved from states to the transitions leaving them (multi-objective queries take
     transition rewards only).
- The joint query replaces `R{"energy"}<=e [ F "done" ]` with `R{"energy"}<=e [ C ]` (multi-objective queries take
  only `C`). This is equal here: "done" is reached with probability 1 under every scheduler (Pmin = 1), and no energy
  accrues in "done".

Equivalence checked (PRISM 4.10.1, explicit engine; `check_*.props`, `energy.props`):

| Query | Original | Counter copy |
|---|---|---|
| Pmax done by 30 | 0.8180961312225429 | 0.8180961312225429 |
| Pmin done by 30 | 0.5164341307824618 | 0.5164341307824618 |
| R{energy}min until done | 24.784391252260622 | 24.784391252260626 |
| R{energy}max until done | 44.38762114126277 | 44.38762114126277 |
| R{energy}max, total (C) | n/a | 44.38762114126277 |

States: 5,580 (original), 95,215 (counter copy). The joint query `joint_counter.props` (sparse engine, `-lp`) had
not finished after 25+ minutes on a desktop at the freeze; its result is not known.
