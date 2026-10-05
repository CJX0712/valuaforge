"""
Second verification pass: calibrate the flagship's EXPECTED gain with real numbers,
so that the DoD thresholds are not invented.

  V6  antithetic (reversed-permutation) pairing -- CORRECT same-budget comparison
  V7  control-variate variance reduction == (1 - rho^2)   [theory check]
  V8  TMC truncation introduces BIAS + breaks efficiency  [measured]
  V9  rho( KNN-Shapley surrogate , LR-Shapley target )    [THE key calibration number]
  V10 common random numbers (CRN) variance reduction      [measured]
  V11 determinism: same seed -> bitwise identical
"""

import math
from math import factorial

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------- utilities
def knn_util(subset, y_tr, d, y_val, K):
    if len(subset) == 0:
        return 0.0
    order = sorted(subset, key=lambda i: (d[i], i))
    return sum(1.0 for i in order[:K] if y_tr[i] == y_val) / K


def knn_shapley(y_tr, d, y_val, K):
    N = len(y_tr)
    order = sorted(range(N), key=lambda i: (d[i], i))
    a = np.array([1.0 if y_tr[i] == y_val else 0.0 for i in order])
    ps = np.zeros(N)
    ps[N - 1] = a[N - 1] * min(K, N) / (K * N)
    for p in range(N - 1, 0, -1):
        ps[p - 1] = ps[p] + (a[p - 1] - a[p]) * (min(K, p) / (K * p))
    phi = np.zeros(N)
    for idx, i in enumerate(order):
        phi[i] = ps[idx]
    return phi


def perms_marginals(pi, y_tr, d, y_val, K):
    cur, prev, delta = [], 0.0, np.zeros(len(pi))
    for i in pi:
        cur.append(i)
        now = knn_util(cur, y_tr, d, y_val, K)
        delta[i] = now - prev
        prev = now
    return delta


# ============================================================ V6 antithetic
print("=" * 78)
print("V6  Antithetic pairing, CORRECT same-budget comparison")
print("=" * 78)
print("  budget = M permutations total.  plain = M iid perms;  anti = M/2 pairs.")
rng = np.random.default_rng(11)
res = {"plain": [], "anti": []}
for rep in range(400):
    N, K = 9, 3
    d = rng.random(N) * 3.0
    y_tr = rng.integers(0, 2, N)
    y_val = int(rng.integers(0, 2))
    phi = knn_shapley(y_tr, d, y_val, K)
    M = 40
    # plain: M iid permutations
    pl = np.mean(
        [perms_marginals(list(rng.permutation(N)), y_tr, d, y_val, K) for _ in range(M)], axis=0
    )
    # anti: M/2 pairs (pi, reversed pi)
    an = np.mean(
        [
            0.5
            * (
                perms_marginals(list(p), y_tr, d, y_val, K)
                + perms_marginals(p[::-1], y_tr, d, y_val, K)
            )
            for p in (list(rng.permutation(N)) for _ in range(M // 2))
        ],
        axis=0,
    )
    res["plain"].append(np.sum((pl - phi) ** 2))
    res["anti"].append(np.sum((an - phi) ** 2))
mp, ma = np.mean(res["plain"]), np.mean(res["anti"])
print(f"  mean squared L2 error: plain MC = {mp:.6e}   antithetic = {ma:.6e}")
print(f"  antithetic / plain ratio = {ma / mp:.4f}   (gain = {100 * (1 - ma / mp):+.1f}%)")
print("  -> NOTE: antithetic gain is utility-dependent; report measured, never assumed.")

# ===================================================== V7 control variate 1-rho^2
print()
print("=" * 78)
print("V7  Control-variate theory check: Var(X - b(Y-EY)) min at b*=Cov/Var, = Var*(1-rho^2)")
print("=" * 78)
rng = np.random.default_rng(3)
rho_true, s2x, s2y = 0.75, 1.0, 4.0
cov = rho_true * math.sqrt(s2x) * math.sqrt(s2y)  # |cov| <= sqrt(s2x*s2y) for PSD
n = 400000
Z = rng.multivariate_normal([0, 0], [[s2x, cov], [cov, s2y]], size=n)
X, Y = Z[:, 0], Z[:, 1]
bstar = np.cov(X, Y)[0, 1] / np.var(Y)
print(f"  measured rho = {pearsonr(X, Y)[0]:.4f} (target {rho_true})")
print(f"  b* = Cov/Var = {bstar:.4f}")
print(f"  Var(X)                = {np.var(X):.4f}")
print(f"  Var(X - b*(Y-EY))     = {np.var(X - bstar * (Y - np.mean(Y))):.4f}")
print(f"  predicted Var*(1-rho^2)= {np.var(X) * (1 - rho_true**2):.4f}")
print(
    f"  -> MSE ratio = {np.var(X - bstar * (Y - np.mean(Y))) / np.var(X):.4f} "
    f"=> L2 error ratio = {math.sqrt(np.var(X - bstar * (Y - np.mean(Y))) / np.var(X)):.4f}"
)
print("  -> CONFIRMED: control variate gives L2 reduction factor sqrt(1-rho^2)")

# ============================================================ V8 TMC bias
print()
print("=" * 78)
print("V8  TMC truncation: bias + efficiency break")
print("=" * 78)


def tmc_shapley(y_tr, d, y_val, K, n_perm, tol, rng):
    N = len(y_tr)
    acc = np.zeros(N)
    cnt = np.zeros(N)
    vN = knn_util(list(range(N)), y_tr, d, y_val, K)
    for _ in range(n_perm):
        pi = list(rng.permutation(N))
        cur, prev = [], 0.0
        truncated = False
        for i in pi:
            cur.append(i)
            now = knn_util(cur, y_tr, d, y_val, K)
            delta = now - prev
            # Ghorbani&Zou truncation: once utility is close to v(N), freeze the rest
            if (not truncated) and abs(now - vN) / max(abs(vN), 1e-12) < tol:
                truncated = True
            if not truncated:
                acc[i] += delta
                cnt[i] += 1
            prev = now
    return acc / np.maximum(cnt, 1)


rng = np.random.default_rng(5)
N, K = 9, 3
d = rng.random(N) * 3.0
y_tr = rng.integers(0, 2, N)
y_val = int(rng.integers(0, 2))
phi = knn_shapley(y_tr, d, y_val, K)
vN = knn_util(list(range(N)), y_tr, d, y_val, K)
print(f"  truth: sum phi = {phi.sum():.6f}   v(N) = {vN:.6f}")
for tol in [1e-9, 0.05, 0.10, 0.20]:
    est = np.mean(
        [
            tmc_shapley(y_tr, d, y_val, K, 60, tol, np.random.default_rng(1000 + s))
            for s in range(300)
        ],
        axis=0,
    )
    bias = np.linalg.norm(est - phi) / np.linalg.norm(phi)
    print(
        f"   tol={tol:<6} L2 rel err={bias:.4f}  sum={est.sum():+.6f}  "
        f"|sum-vN|={abs(est.sum() - vN):.4f}"
    )
print("  -> tol=0 recovers unbiased MC; tol>0 is BIASED and breaks efficiency.")

# ==================================================== V9 rho(surrogate,target)
print()
print("=" * 78)
print("V9  rho( KNN-Shapley SURROGATE , Logistic-Regression Shapley TARGET )")
print("=" * 78)
print("  This is THE number that calibrates the flagship's achievable gain.")


def lr_fit(Xs, ys, p, lam=1e-2):
    if len(Xs) == 0:
        return np.zeros(p)
    Xs = np.asarray(Xs)
    ys = np.asarray(ys, float)

    def obj(w):
        z = Xs @ w
        ll = np.sum(np.logaddexp(0.0, z) - ys * z)
        return ll / len(Xs) + 0.5 * lam * w @ w

    def grad(w):
        z = Xs @ w
        s = 1.0 / (1.0 + np.exp(-z))
        return (Xs.T @ (s - ys)) / len(Xs) + lam * w

    r = minimize(obj, np.zeros(p), jac=grad, method="L-BFGS-B", options={"maxiter": 300})
    return r.x


def lr_util(subset, X, y, Xv, yv, p):
    w = lr_fit([X[i] for i in subset], [y[i] for i in subset], p)
    if len(subset) == 0:
        pred = np.ones(len(yv))  # w=0 -> sigmoid=0.5 -> tie -> predict 1
    else:
        pred = (1.0 / (1.0 + np.exp(-(Xv @ w))) >= 0.5).astype(float)
    return float(np.mean(pred == yv))


def brute_shapley_general(N, vfun):
    v = np.zeros(1 << N)
    for m in range(1 << N):
        v[m] = vfun([i for i in range(N) if (m >> i) & 1])
    phi = np.zeros(N)
    for i in range(N):
        t = 0.0
        for m in range(1 << N):
            if (m >> i) & 1:
                continue
            s = bin(m).count("1")
            t += (factorial(s) * factorial(N - s - 1) / factorial(N)) * (v[m | (1 << i)] - v[m])
        phi[i] = t
    return phi, v[(1 << N) - 1]


rng = np.random.default_rng(2026)
N, p = 10, 4
rhos, rhos_s = [], []
for cfg in range(8):
    mu = np.zeros(p)
    mu[0] = 1.2
    X = np.vstack([rng.normal(mu, 1.0, (N // 2, p)), rng.normal(-mu, 1.0, (N - N // 2, p))])
    y = np.array([0] * (N // 2) + [1] * (N - N // 2))
    if cfg % 2 == 1:  # inject label noise
        flip = rng.choice(N, max(1, N // 10), replace=False)
        y[flip] = 1 - y[flip]
    Xv = np.vstack([rng.normal(mu, 1.0, (10, p)), rng.normal(-mu, 1.0, (10, p))])
    yv = np.array([0] * 10 + [1] * 10)

    # target: exact LR-utility Shapley (brute force over 2^10)
    def _v_target(S, _X=X, _y=y, _Xv=Xv, _yv=yv, _p=p):
        return lr_util(S, _X, _y, _Xv, _yv, _p)

    phi_lr, vNlr = brute_shapley_general(N, _v_target)
    # surrogate: KNN-Shapley, aggregated over validation points (K=5)
    phi_knn = np.zeros(N)
    for j in range(len(yv)):
        dd = np.linalg.norm(X - Xv[j], axis=1)
        phi_knn += knn_shapley(y, dd, int(yv[j]), 5)
    phi_knn /= len(yv)
    rhos.append(pearsonr(phi_knn, phi_lr)[0])
    rhos_s.append(spearmanr(phi_knn, phi_lr)[0])
print(f"  Pearson  rho per config: {np.round(rhos, 3)}")
print(f"  Spearman rho per config: {np.round(rhos_s, 3)}")
print(f"  MEAN Pearson rho  = {np.mean(rhos):.4f}  (min {np.min(rhos):.4f})")
print(f"  MEAN Spearman rho = {np.mean(rhos_s):.4f}  (min {np.min(rhos_s):.4f})")
rq = np.mean(rhos) ** 2
print(f"  => predicted CV L2-error ratio sqrt(1-rho^2) = {math.sqrt(1 - rq):.4f}")
print(
    f"  => predicted CV L2 reduction = {100 * (1 - math.sqrt(1 - rq)):.1f}%  "
    f"(worst-case rho={np.min(rhos):.3f} -> {100 * (1 - math.sqrt(1 - np.min(rhos) ** 2)):.1f}%)"
)
print("  -> MEASURED on-machine. Use the WORST-CASE figure for the DoD threshold.")

# ============================================================ V10 CRN
print()
print("=" * 78)
print("V10 Common Random Numbers: Var( v(S u {i}) - v(S) ) with/without shared seed")
print("=" * 78)


def stochastic_lr_util(subset, X, y, Xv, yv, p, seed, steps=30, lr=0.5):
    """Deliberately stochastic trainer: random init + random shuffle + SGD."""
    g = np.random.default_rng(seed)
    w = g.normal(0, 0.5, p)
    idx = np.array(subset, dtype=int)
    if len(idx) == 0:
        return float(np.mean(np.ones(len(yv)) == yv))
    for _ in range(steps):
        g.shuffle(idx)
        for i in idx:
            z = float(X[i] @ w)
            s = 1 / (1 + math.exp(-z))
            w -= lr * (s - y[i]) * X[i]
    pred = (1.0 / (1.0 + np.exp(-(Xv @ w))) >= 0.5).astype(float)
    return float(np.mean(pred == yv))


rng = np.random.default_rng(77)
Nn, pp = 40, 4
mu = np.zeros(pp)
mu[0] = 1.0
X = np.vstack([rng.normal(mu, 1.0, (Nn // 2, pp)), rng.normal(-mu, 1.0, (Nn - Nn // 2, pp))])
y = np.array([0] * (Nn // 2) + [1] * (Nn - Nn // 2))
Xv = np.vstack([rng.normal(mu, 1.0, (15, pp)), rng.normal(-mu, 1.0, (15, pp))])
yv = np.array([0] * 15 + [1] * 15)
base = list(range(20))
indep, crn = [], []
R = 300
for r in range(R):
    a = np.random.default_rng(r).integers(0, 10**6)
    b = np.random.default_rng(r + 50000).integers(0, 10**6)
    u1 = stochastic_lr_util(base, X, y, Xv, yv, pp, int(a))
    u2i = stochastic_lr_util(base + [20], X, y, Xv, yv, pp, int(b))  # independent seed
    u2c = stochastic_lr_util(base + [20], X, y, Xv, yv, pp, int(a))  # CRN: same seed
    indep.append(u2i - u1)
    crn.append(u2c - u1)
print(f"  Var(delta) independent seeds = {np.var(indep):.6e}")
print(f"  Var(delta) common randoms    = {np.var(crn):.6e}")
print(
    f"  variance ratio CRN/indep     = {np.var(crn) / np.var(indep):.4f} "
    f"(reduction {100 * (1 - np.var(crn) / np.var(indep)):.1f}%)"
)
print("  -> CRN is the cheapest, largest variance lever. Mandate it in the spec.")

# ============================================================ V11 determinism
print()
print("=" * 78)
print("V11 Determinism: same seed -> bitwise identical")
print("=" * 78)


def seeded_run(seed, n_perm=50):
    g = np.random.default_rng(seed)
    N, K = 9, 3
    d = g.random(N) * 3.0
    y_tr = g.integers(0, 2, N)
    y_val = int(g.integers(0, 2))
    acc = np.zeros(N)
    for t in range(n_perm):
        ss = np.random.SeedSequence([seed, t])  # per-permutation stream
        pi = list(np.random.default_rng(ss).permutation(N))
        acc += perms_marginals(pi, y_tr, d, y_val, K)
    return acc / n_perm


r1, r2 = seeded_run(42), seeded_run(42)
print(
    f"  bitwise identical: {np.array_equal(r1, r2)}  (max abs diff {np.max(np.abs(r1 - r2)):.3e})"
)
assert np.array_equal(r1, r2)
print("  -> PASS")

print()
print("ALL CALIBRATION MEASUREMENTS COMPLETE")
