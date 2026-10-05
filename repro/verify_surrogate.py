"""
DECISIVE calibration: which cheap surrogate actually yields a large, reliable
control-variate gain?  And does CRN work when the randomness is structurally aligned?

V9b  rho(Shapley of CHEAP surrogate utility , Shapley of TARGET utility) for several
     candidate surrogates, measured by brute force at N=10.
V10b CRN done PROPERLY (fixed init + fixed master update stream, subset by filtering).
V8b  TMC truncation with burn-in (so truncation cannot fire on |S| < burn_in).
"""

import math
from math import factorial

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr

NCFG = 6


# ------------------------------------------------------------------ LR utilities
def lr_solve(Xs, ys, p, lam=1e-2, iters=300):
    if len(Xs) == 0:
        return np.zeros(p)
    Xs = np.asarray(Xs)
    ys = np.asarray(ys, float)

    def obj(w):
        z = Xs @ w
        return np.sum(np.logaddexp(0.0, z) - ys * z) / len(Xs) + 0.5 * lam * w @ w

    def grad(w):
        z = Xs @ w
        s = 1.0 / (1.0 + np.exp(-z))
        return (Xs.T @ (s - ys)) / len(Xs) + lam * w

    return minimize(obj, np.zeros(p), jac=grad, method="L-BFGS-B", options={"maxiter": iters}).x


def lr_sgd(Xs, ys, p, steps, lr=0.5, lam=0.0, w0=None):
    """CHEAP fidelity: truncated SGD from a FIXED init (deterministic)."""
    w = np.zeros(p) if w0 is None else w0.copy()
    if len(Xs) == 0:
        return w
    Xs = np.asarray(Xs)
    ys = np.asarray(ys, float)
    for t in range(steps):
        i = t % len(Xs)  # deterministic cyclic order
        z = float(Xs[i] @ w)
        s = 1.0 / (1.0 + np.exp(-z))
        w -= lr * ((s - ys[i]) * Xs[i] + lam * w)
    return w


def acc_of(w, Xv, yv, empty_ok=True):
    if np.allclose(w, 0.0) and empty_ok:
        return float(np.mean(np.ones(len(yv)) == yv))  # w=0 -> tie -> predict 1
    pred = (1.0 / (1.0 + np.exp(-(Xv @ w))) >= 0.5).astype(float)
    return float(np.mean(pred == yv))


def brute_shapley(N, vfun):
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
    return phi


# =========================================================================
print("=" * 78)
print("V9b  rho(cheap-surrogate Shapley , target Shapley)  -- the DoD calibration")
print("=" * 78)
rng = np.random.default_rng(20261005)
N, p = 10, 4
NV = 20
rows = {k: [] for k in ["target", "sgd3", "sgd10", "sgd30", "halfval", "knn5"]}

for cfg in range(NCFG):
    mu = np.zeros(p)
    mu[0] = 1.2
    X = np.vstack([rng.normal(mu, 1.0, (N // 2, p)), rng.normal(-mu, 1.0, (N - N // 2, p))])
    y = np.array([0] * (N // 2) + [1] * (N - N // 2))
    if cfg % 2 == 1:
        flip = rng.choice(N, max(1, N // 10), replace=False)
        y[flip] = 1 - y[flip]
    Xv = np.vstack([rng.normal(mu, 1.0, (NV // 2, p)), rng.normal(-mu, 1.0, (NV - NV // 2, p))])
    yv = np.array([0] * (NV // 2) + [1] * (NV - NV // 2))

    # TARGET: full-fidelity LR utility
    def v_target(S, _X=X, _y=y, _p=p, _Xv=Xv, _yv=yv):
        return acc_of(lr_solve([_X[i] for i in S], [_y[i] for i in S], _p), _Xv, _yv)

    phi_t = brute_shapley(N, v_target)

    # CHEAP A: truncated SGD (same model class, few steps) -- fidelity ladder
    for steps in (3, 10, 30):

        def f(S, _st=steps, _X=X, _y=y, _p=p, _Xv=Xv, _yv=yv):
            return acc_of(lr_sgd([_X[i] for i in S], [_y[i] for i in S], _p, _st), _Xv, _yv)

        rows[f"sgd{steps}"].append(pearsonr(brute_shapley(N, f), phi_t)[0])

    # CHEAP B: same LR, but scored on HALF the validation set
    hv = list(range(0, NV, 2))

    def f(S, _X=X, _y=y, _p=p, _Xv=Xv, _yv=yv, _hv=hv):
        return acc_of(lr_solve([_X[i] for i in S], [_y[i] for i in S], _p), _Xv[_hv], _yv[_hv])

    rows["halfval"].append(pearsonr(brute_shapley(N, f), phi_t)[0])

    # CHEAP C: KNN-Shapley (different model CLASS) -- the cross-class surrogate
    phi_knn = np.zeros(N)
    for j in range(NV):
        d = np.linalg.norm(X - Xv[j], axis=1)
        order = sorted(range(N), key=lambda i: (d[i], i))
        K = 5
        a = np.array([1.0 if y[i] == yv[j] else 0.0 for i in order])
        ps = np.zeros(N)
        ps[N - 1] = a[N - 1] * min(K, N) / (K * N)
        for q in range(N - 1, 0, -1):
            ps[q - 1] = ps[q] + (a[q - 1] - a[q]) * (min(K, q) / (K * q))
        for idx, i in enumerate(order):
            phi_knn[i] += ps[idx]
    rows["knn5"].append(pearsonr(phi_knn / NV, phi_t)[0])

print(
    f"  {'surrogate':<10} {'mean rho':>9} {'min rho':>9} {'sqrt(1-r^2)@min':>15} "
    f"{'L2 cut @min':>12}"
)
print("  " + "-" * 60)
for k in ["sgd3", "sgd10", "sgd30", "halfval", "knn5"]:
    r = np.array(rows[k])
    rmin = float(np.min(r))
    rmean = float(np.mean(r))
    ratio = math.sqrt(max(0.0, 1 - rmin**2))
    print(f"  {k:<10} {rmean:>9.4f} {rmin:>9.4f} {ratio:>15.4f} {100 * (1 - ratio):>11.1f}%")
print()
print("  Interpretation: L2 cut @min = the CV gain you can still claim on the WORST config.")
print("  Same-model-class surrogates (sgd*) should dominate the cross-class one (knn5).")

# =========================================================================
print()
print("=" * 78)
print("V10b CRN done properly: fixed init + fixed MASTER update stream, filter by subset")
print("=" * 78)


def train_stream(S, X, y, p, master, w0, lr=0.3, nstep=60):
    """Consume the SAME master stream; skip indices not in S.  This is structural CRN."""
    Sset = set(S)
    w = w0.copy()
    used = 0
    for i in master:
        if i in Sset and used < nstep:
            z = float(X[i] @ w)
            s = 1.0 / (1.0 + np.exp(-z))
            w -= lr * (s - y[i]) * X[i]
            used += 1
        if used >= nstep:
            break
    return w


rng = np.random.default_rng(4242)
Nf, pf = 60, 4
mu = np.zeros(pf)
mu[0] = 1.0
X = np.vstack([rng.normal(mu, 1.0, (Nf // 2, pf)), rng.normal(-mu, 1.0, (Nf - Nf // 2, pf))])
y = np.array([0] * (Nf // 2) + [1] * (Nf - Nf // 2)).astype(float)
Xv = np.vstack([rng.normal(mu, 1.0, (15, pf)), rng.normal(-mu, 1.0, (15, pf))])
yv = np.array([0] * 15 + [1] * 15)

base = list(range(25))
target = base + [25]
R = 400
indep, crn = [], []
for r in range(R):
    ga = np.random.default_rng(r)
    w0a, master_a = ga.normal(0, 0.5, pf), ga.integers(0, Nf, 400)
    gb = np.random.default_rng(r + 999983)
    w0b, master_b = gb.normal(0, 0.5, pf), gb.integers(0, Nf, 400)
    u_base = acc_of(train_stream(base, X, y, pf, master_a, w0a), Xv, yv, empty_ok=False)
    # independent: different init AND different stream
    u_i = acc_of(train_stream(target, X, y, pf, master_b, w0b), Xv, yv, empty_ok=False)
    # CRN: same init AND same stream
    u_c = acc_of(train_stream(target, X, y, pf, master_a, w0a), Xv, yv, empty_ok=False)
    indep.append(u_i - u_base)
    crn.append(u_c - u_base)
vi, vc = np.var(indep), np.var(crn)
print(f"  Var(delta) independent = {vi:.6e}")
print(f"  Var(delta) CRN         = {vc:.6e}")
print(f"  ratio = {vc / vi:.4f}  -> variance reduction {100 * (1 - vc / vi):.1f}%")
print("  -> If this is large, CRN is a first-class lever; if small, report it honestly.")

# =========================================================================
print()
print("=" * 78)
print("V8b TMC truncation WITH burn-in (truncation may not fire before |S| >= burn_in)")
print("=" * 78)


def knn_util(S, y, d, yv, K):
    if not S:
        return 0.0
    o = sorted(S, key=lambda i: (d[i], i))
    return sum(1.0 for i in o[:K] if y[i] == yv) / K


def tmc(N, y, d, yv, K, n_perm, tol, burn_in, g):
    vN = knn_util(list(range(N)), y, d, yv, K)
    acc = np.zeros(N)
    cnt = np.zeros(N)
    for _ in range(n_perm):
        pi = list(g.permutation(N))
        cur, prev, stop = [], 0.0, False
        for pos, i in enumerate(pi):
            cur.append(i)
            now = knn_util(cur, y, d, yv, K)
            if len(cur) >= burn_in and abs(now - vN) / max(abs(vN), 1e-12) < tol:
                stop = True
            if not stop:
                acc[i] += now - prev
                cnt[i] += 1
            prev = now
    return acc / np.maximum(cnt, 1)


g0 = np.random.default_rng(9)
Nn, Kk = 9, 3
d = g0.random(Nn) * 3.0
ytr = g0.integers(0, 2, Nn)
yval = int(g0.integers(0, 2))
# truth via closed form (already proven exact in pass 1)
order = sorted(range(Nn), key=lambda i: (d[i], i))
a = np.array([1.0 if ytr[i] == yval else 0.0 for i in order])
ps = np.zeros(Nn)
ps[Nn - 1] = a[Nn - 1] * min(Kk, Nn) / (Kk * Nn)
for q in range(Nn - 1, 0, -1):
    ps[q - 1] = ps[q] + (a[q - 1] - a[q]) * (min(Kk, q) / (Kk * q))
phi = np.zeros(Nn)
for idx, i in enumerate(order):
    phi[i] = ps[idx]
vN = knn_util(list(range(Nn)), ytr, d, yval, Kk)
print(
    f"  truth sum phi = {phi.sum():.6f}   v(N) = {vN:.6f}   ||phi|| = {np.linalg.norm(phi):.6f}"
)
for burn in [1, 4, 999]:
    for tol in [0.05, 0.20]:
        est = np.mean(
            [
                tmc(Nn, ytr, d, yval, Kk, 80, tol, burn, np.random.default_rng(7000 + s))
                for s in range(250)
            ],
            axis=0,
        )
        rel = np.linalg.norm(est - phi) / np.linalg.norm(phi)
        print(
            f"   burn_in={burn:<4} tol={tol:<5} L2 rel err={rel:.4f}  "
            f"sum={est.sum():+.5f}  |sum-vN|={abs(est.sum() - vN):.5f}"
        )
print("  -> burn_in=999 == no truncation (unbiased baseline). Compare against it.")

print()
print("DECISIVE CALIBRATION COMPLETE")
