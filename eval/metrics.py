"""Metrics: how an estimate is compared to the exact Shapley values.

The primary metric is **L2Rel** (relative L2 error against the exact gold
vector), because the gains from CRN and control variates show up first in
*magnitude* accuracy. A rank-only metric can be completely blind to them.

Two traps are handled explicitly, both measured rather than assumed:

1. ``scipy.stats.spearmanr`` and ``kendalltau`` return **NaN** on constant
   input -- verified in ``repro/`` and pinned by ``tests/test_contracts.py``.
   NaN then poisons a whole ``mean`` column, so a constant input short-circuits
   to ``0.0`` here and every result is screened for non-finiteness.
   ``np.nanmean`` is deliberately **not** used to paper over this: it drops
   the bad entry and reports a healthy-looking mean for a broken run.
2. A vector with fewer than ``max(3, ceil(n/2))`` distinct values is
   degenerate even when its standard deviation is non-zero, so that check
   runs first.

Author: 晨星
"""

from __future__ import annotations

import math

import numpy as np

from core.errors import MetricUndefinedError
from core.types import BenchmarkRow

__all__ = [
    "assert_non_degenerate",
    "l2_rel",
    "rank_metrics",
    "score_row",
    "topk_jaccard",
]

_TINY = 1e-12


def l2_rel(estimate: np.ndarray, truth: np.ndarray) -> float:
    """``||estimate - truth||_2 / ||truth||_2``.

    Returns ``inf`` when the truth is all zeros: the ratio is undefined, and
    reporting a large finite number would disguise that.
    """
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    tru = np.asarray(truth, dtype=np.float64).reshape(-1)
    if est.size != tru.size:
        raise MetricUndefinedError(
            "length mismatch", n_estimate=int(est.size), n_truth=int(tru.size)
        )
    denom = float(np.linalg.norm(tru))
    if denom < _TINY:
        return float("inf")
    return float(np.linalg.norm(est - tru) / denom)


def assert_non_degenerate(values: np.ndarray, name: str) -> None:
    """Reject a degenerate vector *before* any correlation is computed.

    Two independent conditions, both required:

    1. ``n_distinct >= max(3, ceil(n/2))`` -- catches tiny-jitter degeneracy
       that no magnitude threshold can see;
    2. ``std > 0`` -- cheap pre-filter for the exactly-constant case.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = arr.size
    if n == 0:
        raise MetricUndefinedError("empty valuation vector", name=name)
    if not bool(np.all(np.isfinite(arr))):
        raise MetricUndefinedError("valuation contains NaN/Inf", name=name)
    min_distinct = max(3, math.ceil(0.5 * n))
    if np.unique(arr).size < min_distinct:
        raise MetricUndefinedError(
            "degenerate valuation vector",
            name=name,
            n_distinct=int(np.unique(arr).size),
            min_distinct=min_distinct,
            n=n,
        )
    if float(arr.std()) <= 0.0:
        raise MetricUndefinedError("constant valuation vector", name=name)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, with ties resolved by midpoint.

    Written out rather than imported so the tie convention is explicit: scipy
    is a cross-check, not the definition.
    """
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    ac = a - a.mean()
    bc = b - b.mean()
    denom = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    if denom < _TINY:
        return 0.0
    return float(np.dot(ac, bc) / denom)


def rank_metrics(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """``(spearman, kendall)``, both safe on constant input.

    Returns ``0.0`` rather than NaN for a constant input: a degenerate estimate
    has no rank agreement, and ``0.0`` is the honest "no information" answer.
    """
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    tru = np.asarray(truth, dtype=np.float64).reshape(-1)
    if est.size != tru.size:
        raise MetricUndefinedError(
            "length mismatch", n_estimate=int(est.size), n_truth=int(tru.size)
        )
    if est.size < 2:
        return 0.0, 0.0
    if np.unique(est).size < 2 or np.unique(tru).size < 2:
        return 0.0, 0.0
    spearman = _pearson(_rankdata(est), _rankdata(tru))
    kendall = _kendall_tau(est, tru)
    return spearman, kendall


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau-b via the O(n^2) concordance count (n is tiny on the gold track)."""
    n = a.size
    concordant = 0.0
    discordant = 0.0
    ties_a = 0.0
    ties_b = 0.0
    for i in range(n - 1):
        da = a[i + 1 :] - a[i]
        db = b[i + 1 :] - b[i]
        prod = da * db
        concordant += float(np.count_nonzero(prod > 0))
        discordant += float(np.count_nonzero(prod < 0))
        ties_a += float(np.count_nonzero(da == 0))
        ties_b += float(np.count_nonzero(db == 0))
    n0 = n * (n - 1) / 2.0
    denom = np.sqrt((n0 - ties_a) * (n0 - ties_b))
    if denom < _TINY:
        return 0.0
    return float((concordant - discordant) / denom)


def topk_jaccard(estimate: np.ndarray, truth: np.ndarray, frac: float = 0.3) -> float:
    """Jaccard overlap of the top ``frac`` most valuable points."""
    k = max(1, round(frac * np.asarray(truth).size))
    top_e = set(np.argsort(estimate)[-k:].tolist())
    top_t = set(np.argsort(truth)[-k:].tolist())
    union = top_e | top_t
    if not union:
        return 0.0
    return float(len(top_e & top_t) / len(union))


def score_row(
    dataset: str,
    method: str,
    seed: int,
    estimate: np.ndarray,
    truth: np.ndarray,
    n_utility_evals: int,
    wall_seconds: float,
    tier: str = "tier0",
    strict: bool = False,
    n_calls_full: int = 0,
    n_calls_cheap: int = 0,
) -> BenchmarkRow:
    """Score one (dataset, method, seed) cell.

    ``n_calls_full`` / ``n_calls_cheap`` are the two-currency budget split. They
    must be passed through rather than left at their defaults: the field
    ``n_calls_full_equiv`` is what makes "equal budget" auditable, and a row
    that silently reported ``0.0`` would make the whole budget premise vacuous
    while still looking well formed.

    ``strict=False`` records a degenerate estimate as ``skipped`` with a reason
    instead of raising, so one broken cell cannot take down a whole benchmark
    run. It is never silently reported as a number.
    """
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    metrics: dict[str, float] = {}
    try:
        assert_non_degenerate(est, f"{dataset}/{method}")
        metrics["l2_rel"] = l2_rel(est, truth)
        spearman, kendall = rank_metrics(est, truth)
        metrics["spearman"] = spearman
        metrics["kendall"] = kendall
        metrics["topk_jaccard"] = topk_jaccard(est, truth)
        metrics["skipped"] = 0.0
    except MetricUndefinedError as exc:
        if strict:
            raise
        metrics = {
            "l2_rel": float("nan"),
            "spearman": float("nan"),
            "kendall": float("nan"),
            "topk_jaccard": float("nan"),
            "skipped": 1.0,
            "skip_reason": exc.message,
        }
    return BenchmarkRow(
        dataset=dataset,
        method=method,
        seed=int(seed),
        metrics=metrics,
        n_utility_evals=int(n_utility_evals),
        wall_seconds=float(wall_seconds),
        tier=tier,
        n_calls_full=int(n_calls_full),
        n_calls_cheap=int(n_calls_cheap),
    )
