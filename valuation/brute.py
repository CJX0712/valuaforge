"""Anchor A: exact Data Shapley by brute-force enumeration.

Enumerating all ``2^N`` subsets costs exactly ``2^N`` utility evaluations
because ``v(S)`` is memoised across the ``N`` marginal differences that need
it. Beyond the configured ceiling this raises E400 rather than quietly
approximating: an "exact" gold standard that is really another estimator
cannot be used to score estimators.

At N=12 that is 4096 evaluations, which is the whole reason the gold track is
capped there.

Author: 晨星
"""

from __future__ import annotations

from math import factorial

import numpy as np

from core.errors import ensure_gold_size
from core.interfaces import UtilityFn
from core.types import Dataset, ValuationResult

__all__ = ["BruteForceShapley", "exact_shapley"]


def exact_shapley(utility: UtilityFn, n: int) -> np.ndarray:
    """Exact Shapley values by enumerating all ``2**n`` subsets.

    The weights are the classical ``|S|! (n-|S|-1)! / n!``.
    """
    total_masks = 1 << n
    v = np.empty(total_masks, dtype=np.float64)
    for mask in range(total_masks):
        subset = np.array([i for i in range(n) if (mask >> i) & 1], dtype=np.int64)
        v[mask] = utility.evaluate(subset)

    phi = np.zeros(n, dtype=np.float64)
    fact_n = factorial(n)
    for i in range(n):
        bit = 1 << i
        acc = 0.0
        for mask in range(total_masks):
            if (mask >> i) & 1:
                continue
            size = bin(mask).count("1")
            weight = factorial(size) * factorial(n - size - 1) / fact_n
            acc += weight * (v[mask | bit] - v[mask])
        phi[i] = acc
    return phi


class BruteForceShapley:
    """Exact gold standard. Registered under the name ``brute``."""

    name = "brute"
    is_exact = True
    max_n = 12

    @classmethod
    def available(cls) -> bool:
        return True

    def value(
        self,
        data: Dataset,
        utility: UtilityFn,
        budget: object = None,
        rng: np.random.Generator | None = None,
    ) -> ValuationResult:
        n = int(data.n_train)
        ensure_gold_size(n, type(self).max_n, anchor="A")
        before = utility.n_evals
        phi = exact_shapley(utility, n)
        return ValuationResult(
            values=phi,
            method=self.name,
            n_utility_evals=int(utility.n_evals - before),
            wall_seconds=0.0,
            tier="tier0",
            diagnostics={"exact": True, "n_subsets": 1 << n},
        )
