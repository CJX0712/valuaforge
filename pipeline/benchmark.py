"""End-to-end benchmark orchestration.

One pass per (dataset, seed): generate data, compute the exact gold vector once
by enumeration, then run every method **at the same budget** and score it.

Two rules make the comparison meaningful:

* **equal budget.** Every method gets the same ``Budget``. The flagship cannot
  buy accuracy with retrainings the baselines were not allowed.
* **one gold vector per (dataset, seed).** Computing it once and sharing it
  also keeps the truth identical across methods; a per-method truth would let
  a method be graded against a different yardstick.

The end-to-end L2 ratio is the only DoD gate (``eval/report.py``).

Author: 晨星
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from core.errors import ensure_deterministic
from core.registry import get_constructor, register
from core.seed import SeedBank
from core.types import CHEAP_CALL_EQUIV, BenchmarkRow, Budget
from data.dgp import generate
from eval.metrics import score_row
from eval.report import aggregate, build_report, dod_verdict, format_dod_table
from valuation.beta import BetaShapley
from valuation.brute import BruteForceShapley, exact_shapley
from valuation.loo import LeaveOneOut
from valuation.rng_baseline import RandomBaseline, ShuffledTruth
from valuation.tmc import TmcShapley
from valuation.utility import make_utility
from valuation.valuafuse import ValuaFuse

__all__ = ["METHODS", "benchmark", "run_benchmark"]

register("brute")(BruteForceShapley)
register("tmc")(TmcShapley)
register("beta")(BetaShapley)
register("valuafuse")(ValuaFuse)
register("loo")(LeaveOneOut)
register("rng_baseline")(RandomBaseline)

# The benchmark grid. Gold is excluded: it is the yardstick, not a competitor.
METHODS: tuple[str, ...] = (
    "valuafuse",
    "tmc",
    "beta",
    "loo",
    "rng_baseline",
    "shuffle_truth",
)


def _build_method(name: str, truth: np.ndarray) -> object:
    if name == "shuffle_truth":
        return ShuffledTruth(truth)
    return get_constructor(name)()


def run_benchmark(
    datasets: tuple[str, ...] = ("gaussian_mixture", "label_noise"),
    seeds: tuple[int, ...] = (7, 17, 2027),
    n: int = 10,
    d: int = 6,
    budget: int = 2048,
    methods: tuple[str, ...] = METHODS,
    n_val: int = 20,
    verbose: bool = False,
) -> tuple[list[BenchmarkRow], dict]:
    """Run the grid. Returns ``(rows, meta)``.

    ``n`` is the **Shapley problem size** (the number of training points whose
    value is estimated) and ``n_val`` the size of the disjoint validation set.
    The exact gold costs ``2^n`` evaluations, so ``n=10`` means 1024 -- cheap
    enough that the truth is *exact* rather than a noisy stand-in, which is the
    only regime in which an absolute L2Rel can be trusted.
    """
    bank = SeedBank(7)
    rows: list[BenchmarkRow] = []
    failures: list[str] = []
    meta: dict = {
        "n": int(n),
        "n_val": int(n_val),
        "d": int(d),
        "budget": int(budget),
        "seeds": list(seeds),
        "datasets": list(datasets),
        "gold": "exact enumeration (anchor A, 2^n)",
        "gold_absolute": True,
        "budget_currencies": {
            "n_calls_raw": "every utility.evaluate call, surrogate calls included",
            "n_calls_full_equiv": (
                "n_calls_full + " + str(CHEAP_CALL_EQUIV) + " * n_calls_cheap"
            ),
            "cheap_call_equiv": CHEAP_CALL_EQUIV,
            "note": (
                "raw alone overstates a surrogate-using method; equivalent alone "
                "understates its real call pressure. Both are required."
            ),
        },
    }

    for ds_name in datasets:
        for seed in seeds:
            data = generate(ds_name, n=n + n_val, d=d, seed=seed, n_train=n)

            # --- gold, computed once and shared by every method -------------
            gold_util = make_utility("knn", data, Budget(1 << 16), seed=seed, k=5)
            gold = exact_for(data, gold_util)

            for m_name in methods:
                fresh = make_utility("knn", data, Budget(budget), seed=seed, k=5)
                stream = bank.stream(f"val/{m_name}/{ds_name}/{seed}")
                method = _build_method(m_name, gold)
                t0 = time.perf_counter()
                try:
                    result = method.value(data, fresh, Budget(budget), stream)  # type: ignore[attr-defined]
                    elapsed = time.perf_counter() - t0
                    # Dual-currency budget. The flagship's surrogate calls are
                    # cheap; billing them at face value overstates its cost by
                    # ~2.8x, and billing only the equivalent understates the
                    # pressure it puts on the oracle. BenchmarkRow is frozen, so
                    # the fields go in at construction via replace().
                    n_full = int(result.diagnostics.get("n_calls_full", result.n_utility_evals))
                    n_cheap = int(result.diagnostics.get("n_calls_cheap", 0))
                    row = score_row(
                        ds_name,
                        result.method,
                        seed,
                        result.values,
                        gold,
                        n_full + n_cheap,
                        elapsed,
                        result.tier,
                    )
                    row = replace(
                        row,
                        n_calls_full=n_full,
                        n_calls_cheap=n_cheap,
                        n_calls_raw=n_full + n_cheap,
                        n_calls_full_equiv=float(n_full + CHEAP_CALL_EQUIV * n_cheap),
                    )
                    if m_name == "valuafuse":
                        row.metrics.update(
                            {
                                f"diag_{k}": float(v)
                                for k, v in result.diagnostics.items()
                                if isinstance(v, int | float | bool)
                            }
                        )
                        row.metrics["f1_active"] = float(result.diagnostics.get("f1_active", 0))
                        row.metrics["beta_hat"] = float(result.diagnostics.get("beta_hat", 0.0))
                        row.metrics["rho_CV"] = float(result.diagnostics.get("rho_CV", 0.0))
                except Exception as exc:  # one bad cell must not kill the whole grid
                    failures.append(f"{ds_name}/{m_name}/{seed}: {type(exc).__name__}: {exc}")
                    row = BenchmarkRow(
                        dataset=ds_name,
                        method=m_name,
                        seed=seed,
                        metrics={
                            "l2_rel": float("nan"),
                            "spearman": float("nan"),
                            "kendall": float("nan"),
                            "topk_jaccard": float("nan"),
                            "skipped": 1.0,
                            "skip_reason": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-800:],
                        },
                        n_utility_evals=int(fresh.n_evals),
                        wall_seconds=time.perf_counter() - t0,
                        tier="tier0",
                    )
                rows.append(row)
                if verbose:
                    print(
                        f"  {ds_name:<20}{m_name:<16}{row.metrics.get('l2_rel', float('nan')):.4f}"
                    )

    meta["failures"] = failures
    return rows, meta


def exact_for(data, utility) -> np.ndarray:
    """Exact Shapley by enumeration, sharing the utility's counter."""
    return exact_shapley(utility, int(data.n_train))


def benchmark(
    datasets: tuple[str, ...] = ("gaussian_mixture", "label_noise"),
    seeds: tuple[int, ...] = (7, 17, 2027),
    n: int = 10,
    d: int = 6,
    budget: int = 2048,
    out: str = "results",
    n_val: int = 20,
    check_determinism: bool = True,
    verbose: bool = False,
) -> dict:
    """Run, aggregate, judge, and write ``benchmark.json``.

    With ``check_determinism`` the whole grid is run **twice** and the vectors
    compared byte for byte (E500). A benchmark that cannot reproduce itself
    cannot support a performance claim, and the check is cheap at n=10.
    """
    rows, meta = run_benchmark(
        datasets=datasets,
        seeds=seeds,
        n=n,
        d=d,
        budget=budget,
        verbose=verbose,
        n_val=n_val,
    )

    determinism: dict[str, object] = {"checked": False}
    if check_determinism:
        rows2, _ = run_benchmark(
            datasets=datasets, seeds=seeds, n=n, d=d, budget=budget, n_val=n_val
        )
        index1 = {(r.dataset, r.method, r.seed): r for r in rows}
        mismatches: list[str] = []
        for r2 in rows2:
            r1 = index1.get((r2.dataset, r2.method, r2.seed))
            if r1 is None:
                mismatches.append(f"missing {r2.dataset}/{r2.method}/{r2.seed}")
                continue
            a = np.asarray([r1.metrics.get("l2_rel", np.nan)], dtype=np.float64)
            b = np.asarray([r2.metrics.get("l2_rel", np.nan)], dtype=np.float64)
            ensure_deterministic(f"{r2.dataset}/{r2.method}/l2_rel", a, b)
        determinism = {
            "checked": True,
            "bitwise_identical": not mismatches,
            "mismatches": mismatches,
        }

    agg = aggregate(rows)
    verdict = dod_verdict(agg)
    payload = {
        "meta": {**meta, "determinism": determinism},
        "aggregate": agg,
        "dod": verdict,
        "rows": [
            {
                "dataset": r.dataset,
                "method": r.method,
                "seed": r.seed,
                "metrics": {
                    k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                    for k, v in r.metrics.items()
                },
                "n_utility_evals": r.n_utility_evals,
                "n_calls_raw": r.n_calls_raw,
                "n_calls_full_equiv": r.n_calls_full_equiv,
                "n_calls_full": r.n_calls_full,
                "n_calls_cheap": r.n_calls_cheap,
                "wall_seconds": r.wall_seconds,
                "tier": r.tier,
            }
            for r in rows
        ],
    }

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # sort_keys=True: the determinism gate diffs this file, so key order must be
    # fixed rather than insertion-ordered.
    path = out_dir / "benchmark.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if verbose:
        print(format_dod_table(agg, verdict))
    payload["_path"] = str(path)
    payload["_report"] = build_report(rows, meta)
    return payload
