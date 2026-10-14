"""Paired tests over grids for the summary pages (ablation_summary.py): one condition against its reference."""
from typing import Optional

import numpy as np
import pandas as pd


def paired(data: pd.DataFrame, condition: str, reference: str, col: str = "met",
           n_perm: int = 20000) -> Optional[dict]:
    """Mean per-grid difference in `col` (each grid's runs averaged first), 95% bootstrap CI and sign-flip
    permutation p-value; None with fewer than 3 grids in common. `data` has columns condition, sample_id, `col`."""
    a = data[data.condition == condition].groupby("sample_id")[col].mean()
    b = data[data.condition == reference].groupby("sample_id")[col].mean()
    d = (a - b).dropna().to_numpy()
    if len(d) < 3:
        return None
    rng = np.random.default_rng(0)
    boot = rng.choice(d, size=(n_perm, len(d))).mean(axis=1)
    flips = rng.choice([-1, 1], size=(n_perm, len(d)))
    p = float(((np.abs((flips * d).mean(axis=1)) >= abs(d.mean()) - 1e-12).sum() + 1) / (n_perm + 1))
    return {"diff": float(d.mean()), "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975)),
            "p": p, "n": len(d)}


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-adjusted p-values (step-down; controls the chance of any false positive among the comparisons)."""
    adjusted, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted
