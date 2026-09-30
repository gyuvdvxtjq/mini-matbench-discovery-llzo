"""Shared statistics helpers for benchmark reports.

All estimators are deterministic: bootstrap resampling uses a fixed seed, so
reported confidence intervals are reproducible across runs and machines.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

RNG_SEED = 20260927
DEFAULT_RESAMPLES = 2000
MIN_BOOTSTRAP_ROWS = 5


def bootstrap_metric(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = 0.05,
    seed: int = RNG_SEED,
) -> dict[str, Any] | None:
    """Percentile bootstrap confidence interval for a row-level metric.

    ``metric`` receives a resampled copy of ``frame`` and returns a scalar.
    Resamples whose metric is undefined (e.g. F1 with no positives) are
    dropped; if fewer than half the resamples are valid, no interval is
    reported.
    """
    n_rows = len(frame)
    if n_rows < MIN_BOOTSTRAP_ROWS:
        return None
    point = float(metric(frame))
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples)
    for draw in range(n_resamples):
        sample = frame.iloc[rng.integers(0, n_rows, size=n_rows)]
        estimates[draw] = metric(sample)
    estimates = estimates[~np.isnan(estimates)]
    if len(estimates) < n_resamples / 2:
        return None
    low, high = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "alpha": alpha,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def auroc(scores: Any, labels: Any) -> float | None:
    """Rank-based AUROC (Mann-Whitney U statistic), NaN-safe, no sklearn."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    valid = ~np.isnan(scores)
    scores, labels = scores[valid], labels[valid]
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    auc = (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def safe_spearman(left: Any, right: Any) -> float | None:
    """Spearman rank correlation, returning None when undefined."""
    value = pd.Series(left).corr(pd.Series(right), method="spearman")
    return None if pd.isna(value) else float(value)
