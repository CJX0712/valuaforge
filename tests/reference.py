"""Independent golden reference for ValuaForge.

Author: 晨星

This module is deliberately **independent** of the `valuaforge` package. It is
written straight from the definitions in `docs/algorithm_spec.md` so that it can
act as a true oracle: once Phase 2 implements `valuation/*.py`, those modules are
cross-validated against the code here rather than against themselves.

Conventions locked by the spec (see algorithm_spec.md 2.1):
    v(S) = (1/K) * sum_{t=1..min(K,|S|)} 1[y_{alpha_t(S)} == y_val]
i.e. 1/K normalisation with ZERO PADDING when |S| < K (NOT 1/min(K,|S|)).
Ties in distance are broken by ascending index so ordering is deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from math import comb, factorial, lgamma

import numpy as np

__all__ = [
    "beta_shapley_exact",
    "beta_shapley_weights",
    "brute_force_shapley",
    "knn_shapley_closed_form",
    "knn_utility",
    "permutation_marginals",
    "shapley_weights",
    "subset_masks",
]


# ---------------------------------------------------------------------------
# utility
# ---------------------------------------------------------------------------
def knn_utility(
    subset: Sequence[int],
    y_train: np.ndarray,
    dist: np.ndarray,
    y_val: int,
    k: int,
    *,
    v_empty: float = 0.0,
) -> float:
    """KNN utility under the locked 1/K zero-padding convention.

    `v_empty` defaults to 0.0, which is what the 1/K padding convention implies
    for the KNN utility (a zero-filled neighbourhood scores zero).
    """
    if len(subset) == 0:
        return v_empty
    order = sorted(subset, key=lambda i: (dist[i], i))
    hits = sum(1.0 for i in order[:k] if y_train[i] == y_val)
    return hits / k


def subset_masks(n: int) -> list[int]:
    """All 2^n subset bitmasks, ordered by the integer value of the mask."""
    return list(range(1 << n))


# ---------------------------------------------------------------------------
# exact Shapley by brute-force enumeration (anchor A)
# ---------------------------------------------------------------------------
def shapley_weights(n: int, size: int) -> float:
    """Classical Shapley weight |S|!(n-|S|-1)!/n! for a coalition of given size."""
    return factorial(size) * factorial(n - size - 1) / factorial(n)


def brute_force_shapley(
    y_train: np.ndarray,
    dist: np.ndarray,
    y_val: int,
    k: int,
) -> tuple[np.ndarray, float]:
    """Exact Data Shapley by enumerating all 2^n subsets.

    Returns (phi, v_full) where v_full = v(N).
    """
    n = len(y_train)
    v = np.empty(1 << n, dtype=np.float64)
    for mask in subset_masks(n):
        subset = [i for i in range(n) if (mask >> i) & 1]
        v[mask] = knn_utility(subset, y_train, dist, y_val, k)

    phi = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bit_i = 1 << i
        total = 0.0
        for mask in subset_masks(n):
            if (mask >> i) & 1:
                continue
            size = bin(mask).count("1")
            total += shapley_weights(n, size) * (v[mask | bit_i] - v[mask])
        phi[i] = total
    return phi, float(v[(1 << n) - 1])


# ---------------------------------------------------------------------------
# KNN-Shapley closed form (Jia et al. 2019), generalised K
# ---------------------------------------------------------------------------
def knn_shapley_closed_form(
    y_train: np.ndarray,
    dist: np.ndarray,
    y_val: int,
    k: int,
) -> np.ndarray:
    """Closed-form KNN-Shapley under the 1/K padding convention.

    alpha_1..alpha_N = training points by increasing distance to x_val (alpha_1 nearest)
    a_p = 1[y_{alpha_p} == y_val]                                    (1-indexed)
    phi_N = a_N * min(K,N) / (K*N)
    phi_p = phi_{p+1} + (a_p - a_{p+1}) * min(K,p) / (K*p)           p = N-1 .. 1
    """
    n = len(y_train)
    order = sorted(range(n), key=lambda i: (dist[i], i))
    a = np.array([1.0 if y_train[i] == y_val else 0.0 for i in order], dtype=np.float64)

    phi_sorted = np.zeros(n, dtype=np.float64)
    phi_sorted[n - 1] = a[n - 1] * min(k, n) / (k * n)
    for p in range(n - 1, 0, -1):
        coef = min(k, p) / (k * p)
        phi_sorted[p - 1] = phi_sorted[p] + (a[p - 1] - a[p]) * coef

    phi = np.zeros(n, dtype=np.float64)
    for pos, idx in enumerate(order):
        phi[idx] = phi_sorted[pos]
    return phi


# ---------------------------------------------------------------------------
# beta-Shapley
# ---------------------------------------------------------------------------
def beta_shapley_weights(n: int, a_beta: float, b_beta: float) -> tuple[np.ndarray, np.ndarray]:
    """u_k = B(a+k, b+n-1-k) / B(a,b);  p_k = C(n-1,k) u_k is a pmf over k.

    NOTE the parameterisation: in THIS convention a small and b large biases
    towards SMALL coalitions (see algorithm_spec.md 2.4, trap 5).
    """
    log_b_ab = lgamma(a_beta) + lgamma(b_beta) - lgamma(a_beta + b_beta)
    u = np.array(
        [
            math.exp(
                lgamma(a_beta + kk)
                + lgamma(b_beta + n - 1 - kk)
                - lgamma(a_beta + b_beta + n - 1)  # B(x,y) = Gamma(x)Gamma(y)/Gamma(x+y)
                - log_b_ab
            )
            for kk in range(n)
        ],
        dtype=np.float64,
    )
    p = np.array([comb(n - 1, kk) * u[kk] for kk in range(n)], dtype=np.float64)
    return u, p


def beta_shapley_exact(
    y_train: np.ndarray,
    dist: np.ndarray,
    y_val: int,
    k: int,
    a_beta: float,
    b_beta: float,
) -> np.ndarray:
    """Exact beta-Shapley by full enumeration (tiny n only)."""
    n = len(y_train)
    u, _ = beta_shapley_weights(n, a_beta, b_beta)
    v = np.empty(1 << n, dtype=np.float64)
    for mask in subset_masks(n):
        subset = [i for i in range(n) if (mask >> i) & 1]
        v[mask] = knn_utility(subset, y_train, dist, y_val, k)

    phi = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bit_i = 1 << i
        total = 0.0
        for mask in subset_masks(n):
            if (mask >> i) & 1:
                continue
            size = bin(mask).count("1")
            total += u[size] * (v[mask | bit_i] - v[mask])
        phi[i] = total
    return phi


# ---------------------------------------------------------------------------
# permutation estimator
# ---------------------------------------------------------------------------
def permutation_marginals(
    perm: Sequence[int],
    y_train: np.ndarray,
    dist: np.ndarray,
    y_val: int,
    k: int,
) -> np.ndarray:
    """Marginal contribution of every point under a single arrival order."""
    n = len(perm)
    delta = np.zeros(n, dtype=np.float64)
    arrived: list[int] = []
    prev = 0.0
    for idx in perm:
        arrived.append(idx)
        now = knn_utility(arrived, y_train, dist, y_val, k)
        delta[idx] = now - prev
        prev = now
    return delta
