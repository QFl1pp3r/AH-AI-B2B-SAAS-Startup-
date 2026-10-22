"""95% bootstrap CIs for benchmark means: plain, clustered by groups, and paired differences.

One resampler covers all three. Each grouping (player, song, album...) is resampled
independently and a row's multiplicity is the product of its groups' draw counts; with
no grouping every row is its own group, which is the ordinary bootstrap. Two groupings
give the pigeonhole bootstrap used for GuitarSet (players x lead sheets).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

BOOTSTRAP = 2000
SEED = 0


def bootstrap_means(values: Sequence[float], groups: Sequence[Sequence[Any]] = (), weights: Sequence[float] | None = None,
                    n_boot: int = BOOTSTRAP, seed: int = SEED) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    base = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    rng = np.random.default_rng(seed)
    count = np.ones((n_boot, len(values)), dtype=np.int64)
    for labels in groups or [np.arange(len(values))]:
        uniq, idx = np.unique(np.asarray(labels), return_inverse=True)
        count = count * rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=n_boot)[:, idx]
    w = count * base
    total = w.sum(axis=1)
    return (w @ values)[total > 0] / total[total > 0]


def _keep(values: np.ndarray, groups: Sequence[Sequence[Any]], weights: Sequence[float] | None):
    """Drop rows with a NaN value from values, every grouping and the weights alike."""
    ok = ~np.isnan(values)
    return (values[ok], [np.asarray(g)[ok] for g in groups],
            None if weights is None else np.asarray(weights, dtype=float)[ok])


def bootstrap_ci(values: Sequence[float], groups: Sequence[Sequence[Any]] = (), weights: Sequence[float] | None = None,
                 n_boot: int = BOOTSTRAP, seed: int = SEED, digits: int = 4) -> list[float] | None:
    """Percentile 95% CI of the (weighted) mean; None when no finite values are left."""
    values, groups, weights = _keep(np.asarray(values, dtype=float), groups, weights)
    if not len(values):
        return None
    means = bootstrap_means(values, groups, weights, n_boot, seed)
    return [round(float(q), digits) for q in np.percentile(means, [2.5, 97.5])]


def mean_ci(values: Sequence[float], groups: Sequence[Sequence[Any]] = (), weights: Sequence[float] | None = None,
            n_boot: int = BOOTSTRAP, seed: int = SEED, digits: int = 4) -> dict[str, Any]:
    kept, _, w = _keep(np.asarray(values, dtype=float), groups, weights)
    mean = round(float(np.average(kept, weights=w)), digits) if len(kept) and (w is None or w.sum() > 0) else None
    return {"mean": mean, "ci95": bootstrap_ci(values, groups, weights, n_boot, seed, digits), "n": int(len(kept))}


def paired_diff(a: Sequence[float], b: Sequence[float], groups: Sequence[Sequence[Any]] = (),
                weights: Sequence[float] | None = None, n_boot: int = BOOTSTRAP, seed: int = SEED,
                digits: int = 4) -> dict[str, Any]:
    """Mean of a - b over the same tracks with a CI from resampling tracks (or their groups) as pairs.

    Resampling pairs keeps the per-track correlation between the two conditions, so the CI of
    the difference is much narrower than two separate CIs would suggest. A track missing either
    value drops out of both conditions.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    out = mean_ci(diff, groups, weights, n_boot, seed, digits)
    lo_hi = out["ci95"]
    return {"diff": out["mean"], "ci95": lo_hi, "n": out["n"],
            "significant": bool(lo_hi and (lo_hi[0] > 0 or lo_hi[1] < 0))}
