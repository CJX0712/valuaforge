"""
Re-run the N=200 calibration with the CORRECTED ValuaFuse.

Why this file exists
--------------------
`calibrate_n200.py` computed the control-variate coefficient IN SAMPLE:

    Xbar = shapley_mc(Pv, ...); Ybar = shapley_mc(Pv, ...); Yb = shapley_mc(P_pilot, ...)
    beta = cov(Xbar - mean, dY - mean) / var(dY)        # <-- fitted on the same Pv
    vf   = Xbar - beta * dY

That is exactly the optimism the two-independent-batch form was introduced to remove,
and it is also the estimator that the cross-validation later found to be mis-sliceable.
The 0.53 / 0.50 ratios reported from that script are therefore WITHDRAWN pending this
re-run.

Two CV forms are compared here, because the cross-validation showed they are NOT
equivalent for a STOCHASTIC surrogate:

  form A  two-independent-batch difference : x - beta * (Ybar - Yb)      (what I had)
  form B  per-permutation mean-centring    : x - beta * (y - mean(y))    (systems-eng's)

Form B satisfies sum_i (y_i - mean(y)) = 0 for EVERY permutation regardless of whether
the surrogate is stochastic, so it preserves efficiency unconditionally. Form A only
does so for a deterministic surrogate.

Run: python repro/recalibrate_n200.py     (~4 min)
"""

import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / "repro" / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sur = load("sur", "calibrate_n200.py")


def per_perm(v, n, P):
    out = np.zeros((len(P), n))
    for r, pi in enumerate(P):
        cur, prev = [], 0.0
        for i in pi:
            cur.append(i)
            now = v(cur)
            out[r, i] = now - prev
            prev = now
    return out


def cv_estimate(Dfull, Dcheap, Dpilot, form):
    """form 'A' = two-batch difference ; form 'B' = per-permutation mean-centring."""
    M, n = Dfull.shape
    if form == "B":
        yc = Dcheap - Dcheap.mean(axis=1, keepdims=True)
        den = float(np.sum(yc * yc))
        if den <= 1e-14:
            raise ValueError("degenerate: Var(y)=0 (surrogate has no sampling variance)")
        b = float(np.sum((Dfull - Dfull.mean()) * yc) / den)
        return Dfull - b * yc, b
    dY = Dcheap - Dpilot
    half = M // 2
    out = np.zeros((M, n))
    betas = []
    for fit, app in (
        (np.arange(half, M), np.arange(0, half)),
        (np.arange(0, half), np.arange(half, M)),
    ):
        df = dY[fit]
        ctr = df - df.mean()
        den = float(np.sum(ctr * ctr))
        if den <= 1e-14:
            raise ValueError(f"degenerate Var(dY)={den!r} over {len(fit)} perms")
        xf = Dfull[fit]
        b = float(np.sum((xf - xf.mean()) * ctr) / den)
        betas.append(b)
        out[app] = Dfull[app] - b * dY[app]
    return out, float(np.mean(betas))


def run(DGP, n=200, seed=11, budget=None):
    rng = np.random.default_rng(seed)
    if DGP == 1:
        X, y, Xv, yv = sur.dgp1(rng, n=n)
    else:
        X, y, Xv, yv = sur.dgp4(rng, n=n)
    p = X.shape[1]
    nv = len(yv)
    c_cost = 0.05  # measured SGDx30 / LBFGS-style full utility
    per_perm_equiv = 2 * (n + 1) * (1 + c_cost)
    B = budget or int(per_perm_equiv * 100)
    n_perm = max(4, int(B / per_perm_equiv))

    full = lambda S: sur.sgd_fit(X[list(S)], y[list(S)], p, steps=30)
    cheap = lambda S: sur.sgd_fit(X[list(S)], y[list(S)], p, steps=10)
    vfull = lambda S: sur.acc_of(full(S), Xv, yv)
    vcheap = lambda S: sur.acc_of(cheap(S), Xv, yv)

    P1 = [list(rng.permutation(n)) for _ in range(n_perm)]
    P2 = [list(rng.permutation(n)) for _ in range(n_perm)]
    ref = [list(rng.permutation(n)) for _ in range(4 * n_perm)]  # high-fidelity reference

    Dref = per_perm(vfull, n, ref)
    phi_ref = Dref.mean(axis=0)
    Df = per_perm(vfull, n, P1)
    Dc = per_perm(vcheap, n, P1)
    Dp = per_perm(vcheap, n, P2)

    l2 = lambda a: float(np.linalg.norm(a - phi_ref) / max(np.linalg.norm(phi_ref), 1e-12))
    plain = Df.mean(axis=0)
    A, bA = cv_estimate(Df, Dc, Dp, "A")
    Bv, bB = cv_estimate(Df, Dc, Dp, "B")

    vN = vfull(list(range(n)))
    v0 = vfull([])
    print(f"\nDGP-{DGP}  n={n} d={p} val={nv}   budget={B} equiv-units, {n_perm} perms")
    print(f"  reference = SGDx30 @ {4 * n_perm} perms ; c(cost ratio) = {c_cost}")
    print(
        f"  {'estimator':<26} {'L2Rel':>9} {'ratio vs plain':>15} {'beta':>9} {'efficiency':>12}"
    )
    for nm, phi, b in (
        ("plain MC (no CV)", plain, 0.0),
        ("ValuaFuse form A (2-batch)", A.mean(axis=0), bA),
        ("ValuaFuse form B (per-perm)", Bv.mean(axis=0), bB),
    ):
        e = float(abs(phi.sum() - (vN - v0)))
        print(f"  {nm:<26} {l2(phi):>9.4f} {l2(phi) / l2(plain):>15.4f} {b:>+9.4f} {e:>12.2e}")
    return dict(
        dgp=DGP, n=n, l2_plain=l2(plain), l2_A=l2(A.mean(axis=0)), l2_B=l2(Bv.mean(axis=0))
    )


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 84)
    print("N=200 RE-CALIBRATION with the corrected, cross-fitted estimator")
    print("(the previous 0.53/0.50 used an IN-SAMPLE beta and is withdrawn)")
    print("=" * 84)
    r1 = run(1, seed=11)
    r4 = run(4, seed=12)
    print("\n" + "=" * 84)
    print("VERDICT vs DoD 0.75")
    print("=" * 84)
    for r in (r1, r4):
        for tag, key in (("form A", "l2_A"), ("form B", "l2_B")):
            ratio = r[key] / r["l2_plain"]
            print(
                f"  DGP-{r['dgp']}  {tag}: ratio={ratio:.4f}  "
                f"{'PASS' if ratio <= 0.75 else 'FAIL'}"
            )
    print(f"\n  total {time.time() - t0:.0f}s")
