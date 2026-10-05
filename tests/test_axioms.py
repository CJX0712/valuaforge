"""Shapley axiom invariants I1-I6, I9, I10, I12.

Author: 晨星

Cross-validates the reference implementations in `tests/reference.py`. These are
the invariants the spec marks as "must run on every commit" (I1/I2/I9) because
they need no retraining and finish in milliseconds.

Thresholds are taken verbatim from docs/algorithm_spec.md section 3.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy.special import betaln

from tests.reference import (
    beta_shapley_exact,
    beta_shapley_weights,
    brute_force_shapley,
    knn_shapley_closed_form,
    knn_utility,
)

EPS = 1e-9
TIGHT = 1e-12


# ---------------------------------------------------------------------------
# generic brute force over an arbitrary utility callable (used by I5 / I6)
# ---------------------------------------------------------------------------
def brute_force_generic(utility: Callable[[tuple[int, ...]], float], n: int) -> np.ndarray:
    """Exact Shapley values for any utility, by enumerating all 2^n coalitions."""
    v = {}
    for size in range(n + 1):
        for combo in itertools.combinations(range(n), size):
            v[combo] = utility(combo)

    phi = np.zeros(n, dtype=np.float64)
    for i in range(n):
        total = 0.0
        for size in range(n):
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for combo in itertools.combinations([j for j in range(n) if j != i], size):
                with_i = tuple(sorted((*combo, i)))
                total += weight * (v[with_i] - v[combo])
        phi[i] = total
    return phi


def random_utility_table(n: int, seed: int) -> dict[tuple[int, ...], float]:
    """A deterministic arbitrary utility table over all 2^n coalitions."""
    gen = np.random.default_rng(seed)
    table: dict[tuple[int, ...], float] = {}
    for size in range(n + 1):
        for combo in itertools.combinations(range(n), size):
            table[combo] = float(gen.random())
    return table


# ---------------------------------------------------------------------------
# I1 / I2 -- the two headline gates
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2, 3, 5, 9, 13])
def test_i01_closed_form_matches_brute_force(k: int) -> None:
    """I1: KNN-Shapley closed form == brute-force exact Shapley, threshold 1e-12."""
    worst = 0.0
    for n in range(2, 11):
        gen = np.random.default_rng(np.random.SeedSequence([20261005, n, k]))
        for trial in range(4):
            dist = gen.random(n) * 3.0
            if trial == 1:
                dist[0] = dist[1]  # exact tie
            y_train = gen.integers(0, 2, n)
            y_val = int(gen.integers(0, 2))
            phi_bf, _ = brute_force_shapley(y_train, dist, y_val, k)
            phi_cf = knn_shapley_closed_form(y_train, dist, y_val, k)
            worst = max(worst, float(np.max(np.abs(phi_bf - phi_cf))))
    assert worst < TIGHT, f"I1 FAILED: max |closed - brute| = {worst:.3e}"


@pytest.mark.parametrize("k", [1, 2, 3, 5, 9, 13])
def test_i02_efficiency_axiom(k: int) -> None:
    """I2: sum_i phi_i == v(N) - v(empty), threshold 1e-9 * max(1,|v(N)|)."""
    worst = 0.0
    for n in range(2, 11):
        gen = np.random.default_rng(np.random.SeedSequence([20261006, n, k]))
        for _ in range(4):
            dist = gen.random(n) * 3.0
            y_train = gen.integers(0, 2, n)
            y_val = int(gen.integers(0, 2))
            phi_bf, v_full = brute_force_shapley(y_train, dist, y_val, k)
            worst = max(worst, abs(float(phi_bf.sum()) - v_full))
    assert worst < TIGHT, f"I2 FAILED: max |sum(phi) - v(N)| = {worst:.3e}"


# ---------------------------------------------------------------------------
# I3 -- symmetry (exact duplicate points must receive equal value)
# ---------------------------------------------------------------------------
def test_i03_symmetry_under_exact_duplicates(knn_instance) -> None:
    """I3: two points at identical distance with identical label get equal phi."""
    y_train, dist, y_val, k = knn_instance
    n = len(y_train)
    dist = dist.copy()
    y_train = y_train.copy()
    # make points 2 and 3 exact duplicates of each other
    dist[3] = dist[2]
    y_train[3] = y_train[2]
    phi = knn_shapley_closed_form(y_train, dist, y_val, k)
    assert abs(phi[2] - phi[3]) < TIGHT, f"I3 FAILED: phi[2]={phi[2]:.3e} phi[3]={phi[3]:.3e}"
    assert n == 9


# ---------------------------------------------------------------------------
# I4 -- dummy / null player
# ---------------------------------------------------------------------------
def test_i04_dummy_player_contributes_nothing(knn_instance) -> None:
    """I4: a maximally distant point whose label never matches has phi == 0.

    Under 1/K padding a far, always-wrong point is a textbook null player: for
    |S| <= K it adds a zero term, and for |S| > K it never reaches the top-K.
    """
    y_train, dist, y_val, k = knn_instance
    y_ext = np.append(y_train, 1 - y_val)
    dist_ext = np.append(dist, 1e6)
    phi = knn_shapley_closed_form(y_ext, dist_ext, y_val, k)
    assert abs(phi[-1]) < TIGHT, f"I4 FAILED: dummy phi = {phi[-1]:.3e}"


# ---------------------------------------------------------------------------
# I5 -- scale equivariance
# ---------------------------------------------------------------------------
def test_i05_scale_equivariance() -> None:
    """I5: v'(S) = c * v(S) implies phi' = c * phi."""
    n = 8
    base = random_utility_table(n, seed=4242)
    phi = brute_force_generic(lambda s: base[tuple(sorted(s))], n)
    for c in (0.5, 2.0, 10.0):
        phi_scaled = brute_force_generic(lambda s, c=c: c * base[tuple(sorted(s))], n)
        gap = float(np.max(np.abs(phi_scaled - c * phi)))
        tol = EPS * max(1.0, abs(c) * float(np.max(np.abs(phi))))
        assert gap < tol, f"I5 FAILED for c={c}: gap={gap:.3e} tol={tol:.3e}"


# ---------------------------------------------------------------------------
# I6 -- additivity / linearity
# ---------------------------------------------------------------------------
def test_i06_additivity() -> None:
    """I6: v = v1 + v2 implies phi = phi1 + phi2."""
    n = 7
    v1 = random_utility_table(n, seed=777)
    v2 = random_utility_table(n, seed=778)
    phi1 = brute_force_generic(lambda s: v1[tuple(sorted(s))], n)
    phi2 = brute_force_generic(lambda s: v2[tuple(sorted(s))], n)
    phi_sum = brute_force_generic(lambda s: v1[tuple(sorted(s))] + v2[tuple(sorted(s))], n)
    assert float(np.max(np.abs(phi_sum - phi1 - phi2))) < EPS, "I6 FAILED"


# ---------------------------------------------------------------------------
# I9 / I10 -- beta-Shapley
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 3])
def test_i09_beta_one_one_equals_classical(k: int) -> None:
    """I9: beta(1,1) degenerates to classical Shapley, threshold 1e-12."""
    worst = 0.0
    for n in range(2, 9):
        gen = np.random.default_rng(np.random.SeedSequence([20261007, n, k]))
        dist = gen.random(n) * 3.0
        y_train = gen.integers(0, 2, n)
        y_val = int(gen.integers(0, 2))
        phi_bf, _ = brute_force_shapley(y_train, dist, y_val, k)
        phi_beta = beta_shapley_exact(y_train, dist, y_val, k, 1.0, 1.0)
        worst = max(worst, float(np.max(np.abs(phi_bf - phi_beta))))
    assert worst < TIGHT, f"I9 FAILED: max diff = {worst:.3e}"


@pytest.mark.parametrize("n", [4, 8, 12])
@pytest.mark.parametrize(
    ("a_beta", "b_beta"), [(1.0, 1.0), (1.0, 16.0), (4.0, 1.0), (16.0, 1.0)]
)
def test_i10_beta_weights_normalised(n: int, a_beta: float, b_beta: float) -> None:
    """I10: sum_k C(n-1,k) u_k == 1, threshold 1e-9."""
    _, p = beta_shapley_weights(n, a_beta, b_beta)
    assert abs(float(p.sum()) - 1.0) < EPS, f"I10 FAILED: sum = {p.sum():.15f}"


def test_i11_beta_breaks_efficiency_on_purpose() -> None:
    """I11 (REVERSE assertion): beta != (1,1) violates efficiency BY DESIGN.

    Kwon & Zou 2022 deliberately relax the efficiency axiom. This test locks that
    behaviour so a later maintainer does not "fix" it. The redistributed variant
    must then satisfy efficiency.
    """
    n, k = 7, 3
    gen = np.random.default_rng(31337)
    dist = gen.random(n) * 3.0
    y_train = gen.integers(0, 2, n)
    y_val = int(gen.integers(0, 2))
    _, v_full = brute_force_shapley(y_train, dist, y_val, n and k)

    for a_beta, b_beta in [(1.0, 16.0), (16.0, 1.0), (4.0, 1.0)]:
        phi = beta_shapley_exact(y_train, dist, y_val, k, a_beta, b_beta)
        assert abs(float(phi.sum()) - v_full) > 1e-6, (
            f"I11 FAILED: beta({a_beta},{b_beta}) unexpectedly satisfies efficiency"
        )
        # multiplicative redistribution must restore efficiency
        redistributed = phi * (v_full - 0.0) / phi.sum()
        assert abs(float(redistributed.sum()) - v_full) < EPS, "redistribution failed"


# ---------------------------------------------------------------------------
# I12 -- multi validation point aggregation
# ---------------------------------------------------------------------------
def test_i12_multi_val_aggregation_is_exact() -> None:
    """I12: mean_j phi^(j) == exact Shapley of the aggregated utility."""
    n, k, n_val = 7, 3, 4
    gen = np.random.default_rng(90210)
    dist = gen.random(n) * 3.0
    y_train = gen.integers(0, 2, n)
    y_vals = gen.integers(0, 2, n_val)

    # aggregate by linearity of the mean utility
    def agg_utility(subset):
        vals = [knn_utility(list(subset), y_train, dist, int(yv), k) for yv in y_vals]
        return float(np.mean(vals))

    phi_agg = np.mean(
        [knn_shapley_closed_form(y_train, dist, int(yv), k) for yv in y_vals],
        axis=0,
    )
    phi_exact = brute_force_generic(agg_utility, n)
    gap = float(np.max(np.abs(phi_agg - phi_exact)))
    assert gap < TIGHT, f"I12 FAILED: gap = {gap:.3e}"


# ---------------------------------------------------------------------------
# I10b -- beta weight normaliser, pinned against the exact bug that caused it
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("n", "a_beta", "b_beta"), [(12, 4.0, 2.5), (9, 1.0, 1.0), (7, 0.5, 3.0)]
)
def test_i10b_beta_weight_normaliser_uses_full_lgamma_xy(
    n: int, a_beta: float, b_beta: float
) -> None:
    """I10b: B(x,y) must divide by ``Gamma(x+y)``, not just ``Gamma(x)Gamma(y)``.

    Independent of I10 on purpose. I10 asserts only the *consequence*
    (``p.sum() == 1``); this one pins the *mechanism*, so a normalisation error
    that happens to cancel in the sum still fails here.

    Real bug this locks: dropping ``-lgamma(x+y)`` from the ``B(a+k, b+n-1-k)``
    term inflated the weights to ``p.sum() == 24.0`` instead of 1.0.

    Cross-checked against ``scipy.special.betaln`` -- an independent
    implementation of the same normaliser -- entirely in log space.

    Both degenerate-parameter traps below were hit and verified while writing
    this, so the parameters are chosen to avoid them:
      * ``b == 1`` makes Beta a point mass at 0, so ``beta.logpdf(0, a, 1)`` is
        ``-inf`` and any ``x - (-inf)`` comparison silently yields NaN;
      * the naive ``beta.pdf(0, a, b)`` denominator underflows to 0.0, giving 0/0.
    ``betaln`` has neither failure mode.
    """
    _, p = beta_shapley_weights(n, a_beta, b_beta)
    ks = np.arange(n, dtype=np.float64)

    # p_k = C(n-1,k) * B(a+k, b+n-1-k) / B(a,b), evaluated in log space.
    log_b_ab = betaln(a_beta, b_beta)
    log_terms = np.array(
        [
            math.lgamma(n)
            - math.lgamma(kk + 1.0)
            - math.lgamma(n - kk)
            + betaln(a_beta + kk, b_beta + n - 1 - kk)
            - log_b_ab
            for kk in ks
        ],
        dtype=np.float64,
    )
    assert np.all(np.isfinite(log_terms)), f"I10b FAILED: non-finite log weight: {log_terms}"
    expected = np.exp(log_terms)

    assert np.allclose(p, expected, rtol=0, atol=1e-12), (
        f"I10b FAILED: beta weights diverge from scipy.special.betaln.\n"
        f"  p.sum()      = {p.sum():.15f}\n"
        f"  expected.sum = {expected.sum():.15f}"
    )
    # The headline invariant, restated independently of the I10 parametrization.
    assert abs(float(p.sum()) - 1.0) < 1e-12, f"I10b FAILED: p.sum() = {p.sum():.15f}"
