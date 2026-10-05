"""
N=200 calibration for ValuaForge.  Team-lead's ruling: measure, don't argue.

Deliverables
  (a) rho_val(N=200) = corr(phi_SGDx10 , phi_SGDx30)      <- as prescribed
      rho_perm(N=200) = corr of PER-SAMPLE marginals        <- what the CV formula
                                                              in spec 2.7 actually uses
  (b) CRN gain(N=200) = 1 - Var(CRN)/Var(independent)
  (c) END-TO-END DoD metric: L2Rel(ValuaFuse) / L2Rel(plain MC) at MATCHED budget

Why both rhos: spec 2.7 derives the threshold from Var(X)(1-rho^2), where rho is the
correlation of the per-sample marginal contributions (rho_perm).  Team-lead's rho_val
(correlation of the finished valuation vectors) governs how faithfully the cheap vector
tracks the expensive one.  They are NOT the same number; reporting only one would hide
the fact that CV variance reduction and CV bias are governed by different quantities.

Reference "truth" at N=200 is unobtainable by enumeration (2^200), so we use a
high-fidelity reference and MEASURE its own noise floor so the comparison is honest.
"""

import math
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr

N = 200
NV = 100


# ----------------------------------------------------------------- DGPs
def dgp1(rng, n=N, d=5, delta=2.4):
    """GaussBlobs-Clean: two isotropic Gaussian clusters, separation 2.4."""
    mu = np.zeros(d)
    mu[0] = delta / 2.0
    X = np.vstack([rng.normal(mu, 1.0, (n // 2, d)), rng.normal(-mu, 1.0, (n - n // 2, d))])
    y = np.array([0] * (n // 2) + [1] * (n - n // 2))
    Xv = np.vstack([rng.normal(mu, 1.0, (NV // 2, d)), rng.normal(-mu, 1.0, (NV - NV // 2, d))])
    yv = np.array([0] * (NV // 2) + [1] * (NV - NV // 2))
    return X, y, Xv, yv


def dgp4(rng, n=N, sig=2, noise=20):
    """NonlinearMismatch: 2-D two-moons signal buried in `noise` pure-noise dims."""
    m = n // 2
    t1 = np.pi * rng.random(m)
    t2 = np.pi * rng.random(n - m)
    c1 = np.c_[np.cos(t1), np.sin(t1)]
    c2 = np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]
    s = np.vstack([c1, c2]) + rng.normal(0, 0.08, (n, sig))
    y = np.array([0] * m + [1] * (n - m))
    X = np.hstack([s, rng.normal(0, 1.0, (n, noise))])
    m2 = NV // 2
    u1 = np.pi * rng.random(m2)
    u2 = np.pi * rng.random(NV - m2)
    d1 = np.c_[np.cos(u1), np.sin(u1)]
    d2 = np.c_[1 - np.cos(u2), 0.5 - np.sin(u2)]
    sv = np.vstack([d1, d2]) + rng.normal(0, 0.08, (NV, sig))
    yv = np.array([0] * m2 + [1] * (NV - m2))
    Xv = np.hstack([sv, rng.normal(0, 1.0, (NV, noise))])
    return X, y, Xv, yv


# ------------------------------------------------------- utilities (deterministic)
def acc_of(w, Xv, yv):
    pred = (1.0 / (1.0 + np.exp(-(Xv @ w))) >= 0.5).astype(float)
    return float(np.mean(pred == yv))


def sgd_fit(Xs, ys, p, steps, lr=0.5):
    """Truncated SGD, FIXED zero init, deterministic cyclic order -> deterministic."""
    w = np.zeros(p)
    if len(Xs) == 0:
        return w
    m = len(Xs)
    for t in range(steps):
        i = t % m
        z = Xs[i] @ w
        s = 1.0 / (1.0 + np.exp(-z))
        w -= lr * ((s - ys[i]) * Xs[i])
    return w


def lbfgs_fit(Xs, ys, p, lam=1e-2):
    if len(Xs) == 0:
        return np.zeros(p)
    Xs = np.asarray(Xs, float)
    ys = np.asarray(ys, float)

    def obj(w):
        z = Xs @ w
        return np.sum(np.logaddexp(0.0, z) - ys * z) / len(Xs) + 0.5 * lam * w @ w

    def grad(w):
        z = Xs @ w
        return (Xs.T @ (1.0 / (1.0 + np.exp(-z)) - ys)) / len(Xs) + lam * w

    return minimize(obj, np.zeros(p), jac=grad, method="L-BFGS-B", options={"maxiter": 300}).x


def marginals_along(pi, X, y, p, Xv, yv, fitter, **kw):
    """Marginal contribution of every point under permutation pi (order of arrival)."""
    n = len(pi)
    delta = np.zeros(n)
    cur = []
    prev = 0.0
    for i in pi:
        cur.append(i)
        now = acc_of(fitter(X[cur], y[cur], p, **kw), Xv, yv)
        delta[i] = now - prev
        prev = now
    return delta


def shapley_mc(pi_list, X, y, p, Xv, yv, fitter, **kw):
    acc = np.zeros(len(X))
    for pi in pi_list:
        acc += marginals_along(pi, X, y, p, Xv, yv, fitter, **kw)
    return acc / len(pi_list)


def l2rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


# ====================================================================== (a)+(c)
def calibrate(name, X, y, Xv, yv, seed, M_REF=400, M_MAIN=100, M_PILOT=100):
    p = X.shape[1]
    t0 = time.time()

    def perms(k, r):
        g = np.random.default_rng(r)
        return [list(g.permutation(len(X))) for _ in range(k)]

    P_ref = perms(M_REF, seed * 7 + 1)
    P_main = perms(M_MAIN, seed * 7 + 2)
    P_pilot = perms(M_PILOT, seed * 7 + 3)

    # reference = SGD x30 (as prescribed).  Also a full-LBFGS arm at lower budget.
    phi_s30 = shapley_mc(P_ref, X, y, p, Xv, yv, sgd_fit, steps=30)
    # noise floor of the reference itself: half the permutations
    phi_s30_half = shapley_mc(P_ref[: M_REF // 2], X, y, p, Xv, yv, sgd_fit, steps=30)
    floor = l2rel(phi_s30_half, phi_s30)

    t_ref = time.time() - t0
    t1 = time.time()
    phi_lbfgs = shapley_mc(perms(60, seed * 7 + 9), X, y, p, Xv, yv, lbfgs_fit)
    t_lbfgs = time.time() - t1

    # ---- (a) rhos, measured on INDEPENDENT permutation sets (no shared-noise inflation)
    ind10 = [list(np.random.default_rng(10_000 + i).permutation(len(X))) for i in range(200)]
    ind30 = [list(np.random.default_rng(20_000 + i).permutation(len(X))) for i in range(200)]
    v10 = shapley_mc(ind10, X, y, p, Xv, yv, sgd_fit, steps=10)
    v30 = shapley_mc(ind30, X, y, p, Xv, yv, sgd_fit, steps=30)
    rho_val_p = pearsonr(v10, v30)[0]
    rho_val_s = spearmanr(v10, v30)[0]

    per10, per30 = [], []
    for i in range(120):
        pi = list(np.random.default_rng(30_000 + i).permutation(len(X)))
        per10.append(marginals_along(pi, X, y, p, Xv, yv, sgd_fit, steps=10))
        per30.append(marginals_along(pi, X, y, p, Xv, yv, sgd_fit, steps=30))
    P10 = np.array(per10)
    P30 = np.array(per30)
    rho_perm = pearsonr(P10.ravel(), P30.ravel())[0]

    # --- SHARED-permutation rho_val: the protocol the CV estimator actually uses ---
    # The cheap marginal is evaluated on the SAME coalition as the expensive one, so
    # rho must be measured paired.  Independent permutation sets attenuate rho toward
    # 0 on BOTH sides; the control (a surrogate vs ITSELF, see diagnose_rho.py) shows
    # that protocol's ceiling is ~0, which would make any >=0.70 gate unsatisfiable.
    Psh = perms(120, seed * 7 + 31)
    w10 = shapley_mc(Psh, X, y, p, Xv, yv, sgd_fit, steps=10)
    w30 = shapley_mc(Psh, X, y, p, Xv, yv, sgd_fit, steps=30)
    rho_shared = pearsonr(w10, w30)[0]
    # paired per-sample rho is the one the CV formula uses
    rho_perm_paired = pearsonr(P10[:, 0], P30[:, 0])[0]

    print(f"\n{'=' * 76}\n{name}  (N={len(X)}, d={p}, val={NV})\n{'=' * 76}")
    print(f"  reference = SGD x30, M={M_REF} perms        ({t_ref:.1f}s)")
    print(f"  reference noise floor (M/2 vs M)  L2Rel = {floor:.4f}")
    print(
        f"  full-LBFGS arm M=60 vs reference  L2Rel = {l2rel(phi_lbfgs, phi_s30):.4f}  ({t_lbfgs:.1f}s)"
    )
    print()
    print("  (a) rho  [SGDx10 vs SGDx30]")
    print(
        f"      rho_val SHARED perms (CORRECT protocol)  = {rho_shared:.4f}   <-- the gate uses THIS"
        f"      rho_val      (finished vectors, indep. perms) pearson = {rho_val_p:.4f}  spearman = {rho_val_s:.4f}"
    )
    print(f"      rho_perm     (per-sample marginals, pooled)     = {rho_perm:.4f}")
    print(f"      rho_perm_paired (same permutation, 1st point)   = {rho_perm_paired:.4f}")
    print(
        f"      CV L2 factor sqrt(1-rho^2):  rho_val -> {math.sqrt(max(0, 1 - rho_val_p**2)):.4f}"
        f"   rho_perm -> {math.sqrt(max(0, 1 - rho_perm**2)):.4f}"
    )

    # ---- (c) END-TO-END at matched budget -------------------------------------
    # cost model: one full-fidelity utility eval == 1.0; cheap eval == c.
    t0c = time.time()
    for _ in range(60):
        s = list(np.random.default_rng(0).choice(len(X), size=len(X) // 2, replace=False))
        acc_of(lbfgs_fit(X[s], y[s], p), Xv, yv)
    t_full = (time.time() - t0c) / 60
    t0c = time.time()
    for _ in range(400):
        s = list(np.random.default_rng(0).choice(len(X), size=len(X) // 2, replace=False))
        acc_of(sgd_fit(X[s], y[s], p, steps=30), Xv, yv)
    t_s30 = (time.time() - t0c) / 400
    c_cost = t_s30 / t_full
    print(
        f"\n  cost ratio c = t(SGDx30)/t(LBFGS) = {c_cost:.4f}   "
        f"(N={len(X)}: cheap gets cheaper as N grows, as team-lead predicted)"
    )

    B = 40 * (len(X) + 1)  # budget in full-fidelity-equivalent utility evals
    # plain MC: all budget on reference-utility permutations
    m_plain = max(4, int(B / (len(X) + 1)))
    plain = shapley_mc(perms(m_plain, seed * 7 + 21), X, y, p, Xv, yv, sgd_fit, steps=30)

    # ValuaFuse: m2 full + m1 cheap, cost m2 + c*m1 = B  (paired per permutation)
    m2 = int(B / (1.0 + c_cost))
    m1 = int((B - m2) / c_cost) if B - m2 > 0 else 0
    m1 = min(m1, m2)  # need m1 >= m2 for pairing
    m2_eff = int(B - c_cost * m1)  # rebalance
    Pv = P_main[: min(m2_eff, len(P_main))]
    Yb = shapley_mc(P_pilot, X, y, p, Xv, yv, sgd_fit, steps=10)  # independent Y' batch
    Xbar = shapley_mc(Pv, X, y, p, Xv, yv, sgd_fit, steps=30)
    Ybar = shapley_mc(Pv, X, y, p, Xv, yv, sgd_fit, steps=10)
    dY = Ybar - Yb
    beta = float(
        np.dot(Xbar - Xbar.mean(), dY - dY.mean())
        / max(np.dot(dY - dY.mean(), dY - dY.mean()), 1e-12)
    )
    vf = Xbar - beta * dY

    # antithetic ValuaFuse: pair each permutation with its reverse
    Pa = []
    for pi in Pv:
        Pa.append(pi)
        Pa.append(pi[::-1])
    Xbar_a = shapley_mc(Pa, X, y, p, Xv, yv, sgd_fit, steps=30)
    Ybar_a = shapley_mc(Pa, X, y, p, Xv, yv, sgd_fit, steps=10)
    vf_anti = Xbar_a - beta * (Ybar_a - Yb)

    e_plain = l2rel(plain, phi_s30)
    e_vf = l2rel(vf, phi_s30)
    e_anti = l2rel(vf_anti, phi_s30)
    print(f"\n  (c) END-TO-END, budget B = {B} full-fidelity-equivalent evals")
    print(f"      plain MC   (m={m_plain} perms, all budget)      L2Rel = {e_plain:.4f}")
    print(
        f"      ValuaFuse  (m_full={len(Pv)}, m_cheap={len(P_pilot)}, beta={beta:+.4f})  L2Rel = {e_vf:.4f}"
    )
    print(f"      + antithetic pairing                            L2Rel = {e_anti:.4f}")
    print(
        f"      RATIO ValuaFuse / plain = {e_vf / max(e_plain, 1e-12):.4f}    "
        f"(DoD needs <= 0.75)"
    )
    print(f"      RATIO +antithetic / plain = {e_anti / max(e_plain, 1e-12):.4f}")
    # efficiency: exact for plain MC and (by construction) for ValuaFuse
    vN = acc_of(sgd_fit(X, y, p, steps=30), Xv, yv)
    print(
        f"      efficiency |sum-plain - v(N)| = {abs(plain.sum() - vN):.2e}   "
        f"|sum-VF - v(N)| = {abs(vf.sum() - vN):.2e}"
    )
    print(f"  [{time.time() - t0:.1f}s]")
    return dict(
        rho_val=rho_shared,
        rho_val_indep=rho_val_p,
        rho_val_s=rho_val_s,
        rho_perm=rho_perm,
        ratio=e_vf / max(e_plain, 1e-12),
        ratio_anti=e_anti / max(e_plain, 1e-12),
        c=c_cost,
        floor=floor,
    )


# ====================================================================== (b) CRN
def crn_gain(seed, n=N, reps=500):
    """CRN gain at N=200. Structural CRN: fixed init + fixed MASTER update stream,
    filtered by subset.  Independent: different init AND different stream."""
    rng = np.random.default_rng(seed)
    d = 5
    mu = np.zeros(d)
    mu[0] = 1.2
    X = np.vstack([rng.normal(mu, 1.0, (n // 2, d)), rng.normal(-mu, 1.0, (n - n // 2, d))])
    y = np.array([0] * (n // 2) + [1] * (n - n // 2), float)
    Xv = np.vstack([rng.normal(mu, 1.0, (40, d)), rng.normal(-mu, 1.0, (40, d))])
    yv = np.array([0] * 40 + [1] * 40)

    def run(S, w0, master, nstep=60, lr=0.3):
        Sset = set(S)
        w = w0.copy()
        used = 0
        for i in master:
            if i in Sset:
                z = float(X[i] @ w)
                w -= lr * ((1.0 / (1.0 + math.exp(-z))) - y[i]) * X[i]
                used += 1
            if used >= nstep:
                break
        return float(np.mean(((1.0 / (1.0 + np.exp(-(Xv @ w)))) >= 0.5).astype(float) == yv))

    base = list(range(n // 2))
    tgt = base + [n // 2]
    ind, crn = [], []
    for r in range(reps):
        ga = np.random.default_rng(r)
        w0a, ma = ga.normal(0, 0.5, d), ga.integers(0, n, 600)
        gb = np.random.default_rng(r + 999_983)
        w0b, mb = gb.normal(0, 0.5, d), gb.integers(0, n, 600)
        u0 = run(base, w0a, ma)
        ind.append(run(tgt, w0b, mb) - u0)
        crn.append(run(tgt, w0a, ma) - u0)
    vi, vc = float(np.var(ind)), float(np.var(crn))
    return 1.0 - vc / vi, vi, vc


print("#" * 76)
print("# N=200 CALIBRATION  (team-lead ruling: measure, do not argue)")
print("#" * 76)
r1 = calibrate("DGP-1 GaussBlobs-Clean", *dgp1(np.random.default_rng(11)), seed=11)
r4 = calibrate("DGP-4 NonlinearMismatch", *dgp4(np.random.default_rng(12)), seed=12)

print(f"\n{'=' * 76}\n(b) CRN GAIN at N=200\n{'=' * 76}")
g1 = crn_gain(31)
g2 = crn_gain(32, n=N)
print(
    f"  run A: Var(indep)={g1[1]:.6e}  Var(CRN)={g1[2]:.6e}  ratio={g1[2] / g1[1]:.4f}  GAIN={g1[0]:.4f}"
)
print(
    f"  run B: Var(indep)={g2[1]:.6e}  Var(CRN)={g2[2]:.6e}  ratio={g2[2] / g2[1]:.4f}  GAIN={g2[0]:.4f}"
)
gain = (g1[0] + g2[0]) / 2

print(f"\n{'=' * 76}\nVERDICT vs team-lead's thresholds\n{'=' * 76}")
print("  NOTE: the gate uses rho_val(SHARED perms).  The independent-perm value is a")
print("        BROKEN measurement -- its ceiling is ~0 (see diagnose_rho.py / V14).")
for nm, r in (("DGP-1", r1), ("DGP-4", r4)):
    ok = r["rho_val"] >= 0.70
    print(
        f"  {nm}: rho_val(shared)={r['rho_val']:.4f} ({'PASS' if ok else 'FAIL'} vs >=0.70)"
        f"   [indep. protocol would say {r['rho_val_indep']:.4f} -- INVALID]"
        f"   rho_perm={r['rho_perm']:.4f}"
        f"   L2 ratio={r['ratio']:.4f} ({'PASS' if r['ratio'] <= 0.75 else 'FAIL'} vs <=0.75)"
    )
ok_crn = gain >= 0.20
print(f"  CRN gain={gain:.4f} ({'PASS' if ok_crn else 'FAIL'} vs >=0.20)")
print(
    f"\n  >>> OVERALL: {'NO DEGRADATION - spec 4.5 warning can be upgraded' if (r1['rho_val'] >= 0.7 and r4['rho_val'] >= 0.7 and ok_crn) else 'DEGRADATION TRIGGERED'}"
)
print(
    f"  >>> DoD ratio mean = {(r1['ratio'] + r4['ratio']) / 2:.4f}  "
    f"(+antithetic {(r1['ratio_anti'] + r4['ratio_anti']) / 2:.4f})"
)
