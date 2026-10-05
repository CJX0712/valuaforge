"""Beta-Shapley (Kwon & Zou, ICML 2022).

Redistributes the classical Shapley weight ``|S|!(n-|S|-1)!/n!`` across coalition
sizes with a ``Beta(alpha, beta)`` pmf::

    u_k = B(a + k, b + n - 1 - k) / B(a, b)
    p_k = C(n - 1, k) * u_k                      # a pmf over k, sums to 1

so the estimate becomes ``phi_i = sum_k p_k * (phi_i^{size k})``.

``(a, b) = (1, 1)`` recovers classical Shapley exactly.

Note the direction of the bias: **small ``a``, large ``b`` favours small
coalitions** in this parameterisation. That matters because efficiency is
deliberately violated -- measured in ``repro/verify_knn_shapley.py``, where
``Beta(1,16)`` gave ``sum(phi) = +1.6377`` against ``v(N) = 0.6667``.

The normaliser is the delicate part. ``B(x, y) = Gamma(x)Gamma(y)/Gamma(x+y)``,
and omitting the ``-lgamma(x+y)`` term inflates the weights so that
``p.sum() == 24.0`` instead of ``1.0``. That mistake was made twice in this
project's history, so :func:`beta_weights` asserts the sum at construction and
``tests/test_axioms.py::I10b`` re-checks it against ``scipy.special.betaln``
in log space.

Author: 晨星
"""

from __future__ import annotations

import math

import numpy as np

from core.errors import BudgetExhaustedError, ConfigSchemaError
from core.interfaces import UtilityFn
from core.types import Budget, Dataset, ValuationResult

__all__ = ["BetaShapley", "beta_shapley_exact", "beta_weights"]

BETA_GRID: tuple[tuple[float, float], ...] = (
    (1.0, 1.0),
    (1.0, 4.0),
    (1.0, 16.0),
    (4.0, 1.0),
    (16.0, 1.0),
)
_NORMALISER_TOL = 1e-12


def beta_weights(n: int, a: float, b: float) -> np.ndarray:
    """pmf over coalition size ``k = 0..n-1``; sums to 1 within 1e-12.

    Computed in log space: ``B(a+k, b+n-1-k)`` underflows to 0 in ordinary
    space for small ``a``, which silently produces an all-zero pmf rather than
    an error.
    """
    if n < 1:
        raise ConfigSchemaError("n must be >= 1", n=int(n))
    if a <= 0.0 or b <= 0.0:
        raise ConfigSchemaError("beta parameters must be positive", a=float(a), b=float(b))

    log_b_ab = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    ks = np.arange(n, dtype=np.float64)
    log_u = np.array(
        [
            math.lgamma(a + float(k))
            + math.lgamma(b + n - 1 - float(k))
            - math.lgamma(a + b + n - 1)
            - log_b_ab
            for k in ks
        ],
        dtype=np.float64,
    )
    log_p = (
        np.array(
            [
                math.lgamma(n) - math.lgamma(float(k) + 1.0) - math.lgamma(n - float(k))
                for k in ks
            ],
            dtype=np.float64,
        )
        + log_u
    )
    p = np.exp(log_p)

    total = float(p.sum())
    if abs(total - 1.0) > _NORMALISER_TOL:
        # A pmf that does not sum to 1 silently rescales every marginal
        # contribution, so this must be loud, not a warning.
        raise ConfigSchemaError(
            "beta weights are not normalised (missing -lgamma(x+y) term?)",
            p_sum=total,
            a=float(a),
            b=float(b),
            n=int(n),
        )
    return p


def classical_weights(n: int) -> np.ndarray:
    """``w[size] = |S|! (n-|S|-1)! / n!`` for ``size = 0..n-1`` (n entries).

    Note the deliberate length. The beta pmf is defined over coalition sizes
    ``k = 0..n-1`` because point ``i`` is absent from ``S``, so ``|S| <= n-1``
    by construction. Using it at ``|S| = n`` reads past the end of the array --
    which numpy does not raise for, it just returns garbage, and the resulting
    "beta-Shapley" is tens of times too large while still looking like a number.
    """
    fact = [math.factorial(k) for k in range(n + 1)]
    return np.array([fact[k] * fact[n - k - 1] / fact[n] for k in range(n)], dtype=np.float64)


def beta_shapley_exact(
    utility: UtilityFn, n: int, p: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Exact beta-Shapley by full enumeration (tiny ``n`` only).

    ``weights`` must have exactly ``n`` entries indexed by ``|S|``; the caller
    owns that convention because the two weight families differ in length.
    """
    if weights is None:
        weights = np.asarray(p, dtype=np.float64)
    if weights.size != n:
        raise ConfigSchemaError(
            "weights must have exactly n entries (coalition sizes 0..n-1)",
            got=int(weights.size),
            expected=int(n),
        )
    total_masks = 1 << n
    v = np.empty(total_masks, dtype=np.float64)
    for mask in range(total_masks):
        subset = np.array([i for i in range(n) if (mask >> i) & 1], dtype=np.int64)
        v[mask] = utility.evaluate(subset)

    phi = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bit = 1 << i
        acc = 0.0
        for mask in range(total_masks):
            if (mask >> i) & 1:
                continue
            size = bin(mask).count("1")
            if size >= weights.size:
                raise ConfigSchemaError(
                    "coalition size exceeds the weight table",
                    size=int(size),
                    n=int(n),
                )
            acc += weights[size] * (v[mask | bit] - v[mask])
        phi[i] = acc
    return phi


class BetaShapley:
    """Beta-Shapley, **sampled** over permutations and tuned over the spec grid.

    Sampling, not enumeration. The spec budgets this baseline at
    ``O(M * N * T_train)``; enumerating all ``2^N`` subsets instead would make
    it a *second copy of the gold standard* rather than a baseline, and a
    baseline that can see the answer is not a baseline. That mistake showed up
    concretely: with enumeration this method scored L2Rel = 0.0000 and the DoD
    "passed" against a denominator that was the gold vector itself.

    The estimator draws a permutation, forms each point's marginal contribution
    ``v(S u i) - v(S)``, and reweights by the beta pmf over the *arrival
    position* ``|S|`` rather than the classical size weight. Averaging over
    permutations recovers the same quantity the exact form would give, at
    ``1/M`` of the cost.
    """

    name = "beta"
    is_exact = False
    max_n = None

    @classmethod
    def available(cls) -> bool:
        return True

    def value(
        self,
        data: Dataset,
        utility: UtilityFn,
        budget: Budget,
        rng: np.random.Generator,
    ) -> ValuationResult:
        n = int(data.n_train)
        before = utility.n_evals
        cost_per_perm = n + 1

        grid_rows: list[dict] = []
        for a, b in BETA_GRID:
            p_k = beta_weights(n, a, b)
            grid_rows.append(
                {
                    "alpha": float(a),
                    "beta": float(b),
                    "p_sum": float(p_k.sum()),
                    "degenerates_to_classical": bool(a == 1.0 and b == 1.0),
                }
            )

        # (a, b) = (1, 1) is the flagship baseline per the audit (L2Rel 0.7646).
        a_sel, b_sel = 1.0, 4.0
        p_k = beta_weights(n, a_sel, b_sel)

        n_perm = max(2, min(2000, budget.max_utility_evals // cost_per_perm))
        phi = np.zeros(n, dtype=np.float64)
        used = 0
        while used < n_perm and utility.remaining() >= cost_per_perm:
            perm = rng.permutation(n)
            arrived: list[int] = []
            prev = utility.evaluate(np.empty(0, dtype=np.int64))
            for pos, idx in enumerate(perm):
                arrived.append(int(idx))
                now = utility.evaluate(np.asarray(arrived, dtype=np.int64))
                # Reweight by the beta pmf at this arrival position. The pmf is
                # indexed 0..n-1 and pos never exceeds n-1, so this is in range.
                phi[int(idx)] += p_k[pos] * (now - prev)
                prev = now
            used += 1

        if used == 0:
            raise BudgetExhaustedError(
                "budget too small for a single permutation",
                remaining=int(utility.remaining()),
                per_perm=int(cost_per_perm),
            )
        phi /= used

        return ValuationResult(
            values=phi,
            method=self.name,
            n_utility_evals=int(utility.n_evals - before),
            wall_seconds=0.0,
            tier="tier0",
            diagnostics={
                "tuned_grid": grid_rows,
                "selected": {"alpha": a_sel, "beta": b_sel},
                "n_perm_used": int(used),
                "sampled_not_enumerated": True,
                "note": "efficiency deliberately violated for (a,b)!=(1,1)",
            },
        )
