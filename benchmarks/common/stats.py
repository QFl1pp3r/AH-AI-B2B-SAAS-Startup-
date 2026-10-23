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


def wilson_ci(successes: float, n: int, z: float = 1.96, digits: int = 4) -> list[float] | None:
    """Wilson score interval for a proportion; unlike the bootstrap it stays open at 0 or n successes.

    With every one of 100 tasks right the bootstrap can only say [1.0, 1.0]; Wilson says
    [0.963, 1.0], the same as the rule of three. successes may be fractional (a sum of
    per-task rates), which treats each task as one trial: conservative when attempts correlate.
    """
    if n == 0:
        return None
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(max(0.0, center - half)), digits), round(float(min(1.0, center + half)), digits)]


def by_unit_ci(values: Sequence[float], units: Sequence[Any], digits: int = 4) -> dict[str, Any]:
    """Mean with two 95% CIs: the bootstrap over units and Wilson on per-unit means; "ci95" is the wider of the two."""
    out = mean_ci(values, groups=[units], digits=digits)
    per_unit: dict[Any, list[float]] = {}
    for value, unit in zip(values, units):
        if not np.isnan(value):
            per_unit.setdefault(unit, []).append(value)
    rates = [float(np.mean(v)) for v in per_unit.values()]
    out["ci95_bootstrap"] = out["ci95"]
    out["ci95_wilson"] = wilson_ci(sum(rates), len(rates), digits=digits)
    if out["ci95_bootstrap"] and out["ci95_wilson"]:
        out["ci95"] = [min(out["ci95_bootstrap"][0], out["ci95_wilson"][0]), max(out["ci95_bootstrap"][1], out["ci95_wilson"][1])]
    return out
