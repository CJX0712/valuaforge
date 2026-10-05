"""End-to-end demo. Runs the real benchmark and prints the DoD table.

Not a mock: this executes the same code path as ``python -m cli benchmark``,
including the exact gold standard (2^n enumeration) and the double-run
determinism check. If the flagship misses the gate, the demo says so and exits
non-zero -- a demo that always exits 0 is decoration.

Runtime: well under the 60 s budget at n=10, 1 seed, budget 1024.

Author: 晨星
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Running this file by path puts ``examples/`` on sys.path, not the repo root,
# so the flat top-level packages (``core``, ``data``, ...) would be unimportable.
# CI runs the demo as a script, so the root has to be added here.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import contextlib

from core.errors import ValuaError
from core.seed import SeedBank
from core.types import Budget
from data.dgp import generate
from eval.report import aggregate, dod_verdict, format_dod_table
from pipeline.benchmark import METHODS, run_benchmark
from valuation.brute import exact_shapley
from valuation.utility import make_utility

__all__ = ["main"]

# Measured on this track (gaussian_mixture, n_train=10, d=6, seed=7, 200 calls,
# time.perf_counter around `utility.evaluate`): knn 67.21 us/call, the hardcoded
# surrogate sgd(10 steps) 185.94 us/call. Ratio 2.77x in WALL CLOCK -- and
# exactly 1.00x in the budget currency the DoD actually uses. Constants here
# because a demo should not silently re-measure and drift; the audit script
# repro/audit_scale_and_budget.py measures per track.
_SURROGATE_COST_RATIO = 2.77


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(ValueError, OSError):
                stream.reconfigure(encoding="utf-8")


def _determinism_demo() -> None:
    """Show that the same seed reproduces the same stream, across processes' rules."""
    print("1. determinism")
    bank = SeedBank(7)
    a = bank.stream("val/valuafuse/gaussian_mixture/7").standard_normal(4)
    b = bank.stream("val/valuafuse/gaussian_mixture/7").standard_normal(4)
    c = bank.stream("val/tmc/gaussian_mixture/7").standard_normal(4)
    print(f"   same stream twice bitwise equal : {a.tobytes() == b.tobytes()}")
    print(f"   different stream differs        : {a.tobytes() != c.tobytes()}")
    print(
        f"   entropy of val/valuafuse/...   : {bank.entropy_for('val/valuafuse/gaussian_mixture/7')}"
    )


def _gold_demo() -> None:
    """Show the gold standard and that the budget counter is real."""
    print()
    print("2. gold standard (anchor A, exact by enumeration)")
    data = generate("gaussian_mixture", n=30, d=6, seed=7, n_train=10)
    oracle = make_utility("knn", data, Budget(1 << 16), seed=7, k=5)
    gold = exact_shapley(oracle, data.n_train)
    print(
        f"   n_train={data.n_train}  utility evals={oracle.n_evals}  (== 2^n = {2**data.n_train})"
    )
    print(f"   gold[:5] = {[round(float(v), 4) for v in gold[:5]]}")
    print(
        f"   efficiency |sum(phi) - v(N)| = {abs(float(gold.sum()) - oracle.evaluate(__import__('numpy').arange(data.n_train))):.2e}"
    )

    print()
    print("3. budget guard is real, not estimated")
    capped = make_utility("knn", data, Budget(20), seed=7, k=5)
    try:
        for _ in range(100):
            capped.evaluate([0, 1, 2])
    except ValuaError as exc:
        print(
            f"   stopped at {capped.n_evals} evals with {exc.code} (cap 20) -- no silent truncation"
        )


def main() -> int:
    _utf8()
    t0 = time.perf_counter()

    _determinism_demo()
    _gold_demo()

    print()
    print("4. benchmark (n=10, exact gold, 1 seed) -- DO NOT read the ratio; see step 5")
    rows, meta = run_benchmark(
        datasets=("gaussian_mixture", "label_noise"),
        seeds=(7,),
        n=10,
        n_val=20,
        budget=2048,
        methods=METHODS,
    )
    agg = aggregate(rows)
    verdict = dod_verdict(agg)
    print(format_dod_table(agg, verdict))

    fuse = [r for r in rows if r.method == "valuafuse"]
    if fuse:
        m = fuse[0].metrics
        print()
        print(f"   F1 control variate active : {bool(m.get('f1_active'))}")
        print(f"   beta_hat                  : {m.get('beta_hat', float('nan')):+.4f}")
        print(f"   rho_CV                    : {m.get('rho_CV', float('nan')):.4f}")

    failures = meta.get("failures", [])
    print()
    print(f"   failed cells: {len(failures)}")
    for line in failures[:3]:
        print(f"     {line[:100]}")

    print()
    print(f"demo finished in {time.perf_counter() - t0:.1f}s")

    # --- I27: is the flagship's premise even true on THIS track? -------------
    # The budget currency is `utility.evaluate` call count, and I21 books the
    # surrogate and the target identically: one call costs one unit either way.
    # So the flagship only pays off where cost(surrogate) < cost(target), and
    # that ratio must be MEASURED per track -- it is not a constant. On this
    # KNN track it is 1.00 (measured 67.21 vs 185.94 us/call), meaning there is
    # nothing to save: spending budget on the surrogate buys an equally
    # expensive and strictly noisier sample. See docs/algorithm_spec.md 4.7.
    #
    # Reporting PASS here would be an accounting artefact -- it compares two
    # different currencies. The number is printed because hiding it would be
    # worse, but it is explicitly marked VOID.
    print()
    print("5. I27 premise check on this track")
    print("   budget currency       : utility.evaluate calls (target AND surrogate cost 1)")
    print(
        f"   measured cost ratio c : {_SURROGATE_COST_RATIO:.2f}x   (sgd 185.94 / knn 67.21 us/call)"
    )
    if _SURROGATE_COST_RATIO >= 1.0:
        print("   VERDICT: VOID -- c >= 1 in budget units, the surrogate is not cheaper.")
        print("            The PASS ratio printed in step 4 is an accounting artefact,")
        print("            NOT evidence of performance. See docs/algorithm_spec.md 4.7.")
        print("   Audited result on the c<1 track (L-BFGS): ratio 1.1563 -> FAIL.")
        print()
        print("demo verdict: VOID (I27 premise fails on this track; no performance claim)")
        return 1

    ok = verdict.get("verdict") == "PASS" and not failures
    print(f"demo verdict: {'PASS' if ok else verdict.get('verdict')}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
