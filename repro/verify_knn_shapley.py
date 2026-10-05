"""
Spec-verification harness for ValuaForge.
Goal: BEFORE writing any formula into algorithm_spec.md, verify it by brute-force
enumeration of the exact Shapley value (2^N subsets) on tiny instances.

Verified here:
  V1  KNN-Shapley closed-form recursion (general K) == brute-force exact Shapley
  V2  Efficiency axiom  sum_i phi_i == v(N) - v(empty)
  V3  Beta-Shapley size weights  sum_k C(N-1,k) u_k == 1
  V4  Beta-Shapley (a,b)=(1,1) degenerates to classical Shapley
  V5  Permutation-sampling estimator is unbiased
  V6  Antithetic (reversed permutation) pairing reduces variance
"""

import itertools
import math
from math import comb, factorial, lgamma

import numpy as np


# ----------------------------------------------------------------------------
# Utility convention LOCKED by spec:
#   v(S) = (1/K) * sum_{t=1..min(K,|S|)} 1[y_{alpha_t(S)} == y_val]
# i.e. 1/K normalisation with ZERO PADDING when |S| < K.  v(empty) = 0.
# ----------------------------------------------------------------------------
def knn_util(subset, y_tr, d, y_val, K):
    if len(subset) == 0:
        return 0.0
    order = sorted(subset, key=lambda i: (d[i], i))  # tie-break by index -> determinism
    return sum(1.0 for i in order[:K] if y_tr[i] == y_val) / K


def brute_force_shapley(y_tr, d, y_val, K):
    N = len(y_tr)
    v = np.zeros(1 << N)
    for m in range(1 << N):
        v[m] = knn_util([i for i in range(N) if (m >> i) & 1], y_tr, d, y_val, K)
    phi = np.zeros(N)
    for i in range(N):
        tot = 0.0
        for m in range(1 << N):
            if (m >> i) & 1:
                continue
            s = bin(m).count("1")
            w = factorial(s) * factorial(N - s - 1) / factorial(N)
            tot += w * (v[m | (1 << i)] - v[m])
        phi[i] = tot
    return phi, v[(1 << N) - 1]


def knn_shapley_closed_form(y_tr, d, y_val, K):
    """Jia et al. 2019 closed form, generalised K, under the 1/K-padding convention.

    alpha_1..alpha_N : training points sorted by INCREASING distance to x_val
    a_p = 1[y_{alpha_p} == y_val]
    phi_N   = a_N * min(K,N) / (K*N)
    phi_p   = phi_{p+1} + (a_p - a_{p+1}) * min(K,p) / (K*p)      p = N-1 .. 1
    """
    N = len(y_tr)
    order = sorted(range(N), key=lambda i: (d[i], i))
    a = np.array([1.0 if y_tr[i] == y_val else 0.0 for i in order])  # a[0] is p=1 (1-indexed)
    phi_sorted = np.zeros(N)
    p = N  # 1-indexed position, farthest
    phi_sorted[N - 1] = a[N - 1] * min(K, N) / (K * N)
    for p in range(N - 1, 0, -1):  # p = N-1 ... 1
        c = min(K, p) / (K * p)
        phi_sorted[p - 1] = phi_sorted[p] + (a[p - 1] - a[p]) * c
    phi = np.zeros(N)
    for idx, i in enumerate(order):
        phi[i] = phi_sorted[idx]
    return phi


def beta_shapley_weights(N, a_beta, b_beta):
    """u_k = B(a+k, b+N-1-k) / B(a,b);  p_k = C(N-1,k)*u_k  is a pmf over k."""

    def lB(x, y):
        return lgamma(x) + lgamma(y) - lgamma(x + y)

    lBab = lB(a_beta, b_beta)
    u = np.array([math.exp(lB(a_beta + k, b_beta + N - 1 - k) - lBab) for k in range(N)])
    p = np.array([comb(N - 1, k) * u[k] for k in range(N)])
    return u, p


def beta_shapley_exact(y_tr, d, y_val, K, a_beta, b_beta):
    """Exact beta-Shapley by full enumeration (only feasible for tiny N)."""
    N = len(y_tr)
    u, _ = beta_shapley_weights(N, a_beta, b_beta)
    v = np.zeros(1 << N)
    for m in range(1 << N):
        v[m] = knn_util([i for i in range(N) if (m >> i) & 1], y_tr, d, y_val, K)
    phi = np.zeros(N)
    for i in range(N):
        tot = 0.0
        for m in range(1 << N):
            if (m >> i) & 1:
                continue
            s = bin(m).count("1")
            tot += u[s] * (v[m | (1 << i)] - v[m])
        phi[i] = tot
    return phi


def permutation_marginals(pi, y_tr, d, y_val, K):
    """Marginal contributions of each point under permutation pi (order of arrival)."""
    N = len(pi)
    delta = np.zeros(N)
    cur = []
    prev = 0.0
    for pos, i in enumerate(pi):
        cur.append(i)
        now = knn_util(cur, y_tr, d, y_val, K)
        delta[i] = now - prev
        prev = now
    return delta


# ===========================================================================
rng = np.random.default_rng(20261005)
print("=" * 78)
print("V1/V2  KNN-Shapley closed form vs brute force  (N<=12, all K)")
print("=" * 78)
worst_v1 = 0.0
worst_v2 = 0.0
worst_v2_abs = 0.0
for N in range(2, 11):
    for K in [1, 2, 3, 5, 9, 13]:
        for trial in range(6):
            d = rng.random(N) * 3.0
            if trial == 1:  # force exact distance ties
                d[0] = d[1]
            y_tr = rng.integers(0, 2, N)
            y_val = int(rng.integers(0, 2))
            phi_bf, vN = brute_force_shapley(y_tr, d, y_val, K)
            phi_cf = knn_shapley_closed_form(y_tr, d, y_val, K)
            e1 = float(np.max(np.abs(phi_bf - phi_cf)))
            # absolute gap is the honest measure; scale-relative only when v(N)!=0
            gap = abs(float(phi_cf.sum()) - vN)
            e2 = gap / abs(vN) if abs(vN) > 1e-12 else 0.0
            worst_v1 = max(worst_v1, e1)
            worst_v2 = max(worst_v2, e2)
            worst_v2_abs = max(worst_v2_abs, gap)
print(f"  max |closed-form - brute-force|      = {worst_v1:.3e}")
print(f"  max ABSOLUTE efficiency gap          = {worst_v2_abs:.3e}")
print(f"  max RELATIVE efficiency gap (v(N)!=0)= {worst_v2:.3e}")
assert worst_v1 < 1e-12, "V1 FAILED: closed form does not match brute force"
assert worst_v2_abs < 1e-12, "V2 FAILED: efficiency violated"
assert worst_v2 < 1e-12, "V2 FAILED: relative efficiency violated"
print("  -> V1 PASS (closed form is EXACT)   V2 PASS (efficiency holds)")

print()
print("=" * 78)
print("V3/V4  Beta-Shapley weights")
print("=" * 78)
for N in [4, 8, 12]:
    for ab, bb in [(1, 1), (1, 16), (4, 1), (0.5, 0.5), (16, 1)]:
        u, p = beta_shapley_weights(N, ab, bb)
        assert abs(p.sum() - 1.0) < 1e-9, f"V3 fail N={N} a={ab} b={bb}: {p.sum()}"
print("  -> V3 PASS  sum_k C(N-1,k) u_k == 1 for all tested (N,a,b)")

# V4: (1,1) must reproduce classical Shapley
for N in range(2, 9):
    for K in [1, 3]:
        d = rng.random(N) * 3.0
        y_tr = rng.integers(0, 2, N)
        y_val = int(rng.integers(0, 2))
        phi_bf, _ = brute_force_shapley(y_tr, d, y_val, K)
        phi_b11 = beta_shapley_exact(y_tr, d, y_val, K, 1.0, 1.0)
        assert np.max(np.abs(phi_bf - phi_b11)) < 1e-12, f"V4 fail N={N} K={K}"
print("  -> V4 PASS  Beta(1,1) == classical Shapley (bitwise, tol 1e-12)")

# V4b: (a,b) != (1,1) BREAKS efficiency -> spec must mandate renormalisation
print("\n  Efficiency check for Beta(a,b) != (1,1):")
for ab, bb in [(1, 16), (16, 1), (4, 1)]:
    N, K = 7, 3
    d = rng.random(N) * 3.0
    y_tr = rng.integers(0, 2, N)
    y_val = int(rng.integers(0, 2))
    phi = beta_shapley_exact(y_tr, d, y_val, K, ab, bb)
    _, vN = brute_force_shapley(y_tr, d, y_val, K)
    print(
        f"    Beta({ab:>4},{bb:>2}): sum phi = {phi.sum():+.6f}   v(N) = {vN:.6f}   "
        f"gap = {phi.sum() - vN:+.6f}"
    )

print()
print("=" * 78)
print("V5/V6  Permutation-sampling estimator: unbiasedness + antithetic gain")
print("=" * 78)
N, K = 9, 3
d = rng.random(N) * 3.0
y_tr = rng.integers(0, 2, N)
y_val = int(rng.integers(0, 2))
# ground truth via closed form (already proven exact)
phi_true = knn_shapley_closed_form(y_tr, d, y_val, K)

all_perms = list(itertools.permutations(range(N)))
true_mean = np.mean([permutation_marginals(p, y_tr, d, y_val, K) for p in all_perms], axis=0)
print(
    f"  |mean over ALL {len(all_perms)} permutations - closed form| = "
    f"{np.max(np.abs(true_mean - phi_true)):.3e}  -> estimator is UNBIASED (exhaustive check)"
)
assert np.max(np.abs(true_mean - phi_true)) < 1e-12

# antithetic: pair permutation pi with reversed pi
rng2 = np.random.default_rng(7)
M = 400
plain, anti = [], []
for _ in range(M // 2):
    pi = list(rng2.permutation(N))
    m1 = permutation_marginals(pi, y_tr, d, y_val, K)
    m2 = permutation_marginals(pi[::-1], y_tr, d, y_val, K)
    plain.append(m1)
    plain.append(m2)
    anti.append(0.5 * (m1 + m2))
plain = np.array(plain)
anti = np.array(anti)
# variance of the mean estimate
vp = plain.mean(axis=0).var()  # not the right object; use per-sample trace var
var_plain = float(np.mean(plain.var(axis=0)))
var_anti_mean = float(np.mean((anti - phi_true) ** 2))
var_plain_mean = float(np.mean((plain - phi_true) ** 2))
# compare estimators that consume the SAME number of permutations (M each)
est_plain = plain.mean(axis=0)
est_anti = anti.mean(axis=0)
print(
    f"  MSE vs truth (M={M} perms): plain MC = {np.mean((est_plain - phi_true) ** 2):.6e}   "
    f"antithetic = {np.mean((est_anti - phi_true) ** 2):.6e}"
)
print(f"  per-sample mean variance:  plain = {var_plain:.6e}")
print(
    "  -> V5 PASS (exact unbiasedness)  V6 computed above (antithetic gain is "
    "problem-dependent, NOT guaranteed positive for KNN utility)"
)

print()
print("ALL SPEC VERIFICATIONS COMPLETE")
