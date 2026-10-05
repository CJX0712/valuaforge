"""F1 / F2 / F3 ablation for ValuaFuse.

Team-lead's attribution warning: if the real ``rho_CV`` is low, then
``sqrt(1-rho^2) ~ 1`` and the control variate contributes almost nothing, so the
flagship's gain would be coming entirely from CRN (F2) and antithetic pairing
(F3). This script settles it by measurement rather than by formula.

Rows:

===========================  =========================================
plain MC                     no F1, no F2, no F3 -- the floor
F3 only                      antithetic pairing
F2 only                      common random numbers
F2 + F3                      both variance levers, no control variate
F1 forced on                 CV applied with the gate overridden
full (gated, shipped)        what actually ships
===========================  =========================================

CV and CRN are **not** treated as additive. The two gains are reported side by
side and no combined figure is computed, because a high rho means the surrogate
already explains the variance CRN would have removed, so
``1 - (1-a)(1-b)`` is at best an approximation and an additive claim would be
fiction.

Author: 晨星
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import contextlib  # noqa: E402

import numpy as np  # noqa: E402

from core.seed import SeedBank  # noqa: E402
from core.types import Budget  # noqa: E402
from data.dgp import generate  # noqa: E402
from eval.metrics import l2_rel  # noqa: E402
from valuation.brute import exact_shapley  # noqa: E402
from valuation.utility import make_utility  # noqa: E402
from valuation.valuafuse import FuseConfig, run_fused  # noqa: E402

CONFIGS: list[tuple[str, FuseConfig]] = [
    ("plain_mc", FuseConfig(use_cv=False, use_crn=False, use_antithetic=False)),
    ("f3_antithetic", FuseConfig(use_cv=False, use_crn=False, use_antithetic=True)),
    ("f2_crn", FuseConfig(use_cv=False, use_crn=True, use_antithetic=False)),
    ("f2_f3", FuseConfig(use_cv=False, use_crn=True, use_antithetic=True)),
    ("f1_forced", FuseConfig(use_cv=True, use_crn=True, use_antithetic=True)),
    ("full_gated", FuseConfig(use_cv=True, use_crn=True, use_antithetic=True)),
]
DATASETS = ("gaussian_mixture", "label_noise")
SEEDS = (7, 17, 2027)
N = 10
N_VAL = 20
BUDGET = 2048


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(ValueError, OSError):
                stream.reconfigure(encoding="utf-8")


def main() -> int:
    _utf8()
    bank = SeedBank(7)
    scores: dict[str, list[float]] = {name: [] for name, _ in CONFIGS}
    betas: dict[str, list[float]] = {name: [] for name, _ in CONFIGS}
    rhos: dict[str, list[float]] = {name: [] for name, _ in CONFIGS}
    gate_on: dict[str, int] = {name: 0 for name, _ in CONFIGS}

    for ds in DATASETS:
        for seed in SEEDS:
            data = generate(ds, n=N + N_VAL, d=6, seed=seed, n_train=N)
            gold_util = make_utility("knn", data, Budget(1 << 16), seed=seed, k=5)
            gold = exact_shapley(gold_util, data.n_train)

            for name, cfg in CONFIGS:
                # "f1_forced" bypasses the sign gate so its cost can be measured
                # even when the gate would reject it.
                forced = name == "f1_forced"
                run_cfg = FuseConfig(
                    use_cv=cfg.use_cv,
                    use_crn=cfg.use_crn,
                    use_antithetic=cfg.use_antithetic,
                    rho_cv_gate=(-1.0 if forced else cfg.rho_cv_gate),
                    beta_min=(-1.0 if forced else cfg.beta_min),
                )
                oracle = make_utility("knn", data, Budget(BUDGET), seed=seed, k=5)
                stream = bank.stream(f"val/{name}/{ds}/{seed}")
                phi, diag = run_fused(data, oracle, Budget(BUDGET), stream, run_cfg)
                scores[name].append(l2_rel(phi, gold))
                betas[name].append(float(diag["beta_hat"]))
                rhos[name].append(float(diag["rho_CV"]))
                gate_on[name] += int(bool(diag["f1_active"]))

    print(
        f"ablation: n={N} (exact gold), datasets={list(DATASETS)}, "
        f"seeds={list(SEEDS)}, budget={BUDGET}"
    )
    print()
    print(
        f"{'config':<14}{'L2Rel mean':>12}{'std':>9}{'beta_hat':>11}{'rho_CV':>9}{'F1 on':>7}"
    )
    print(f"{'-' * 14}{'-' * 12}{'-' * 9}{'-' * 11}{'-' * 9}{'-' * 7}")
    for name, _ in CONFIGS:
        arr = np.array(scores[name])
        print(
            f"{name:<14}{arr.mean():>12.4f}{arr.std(ddof=1) if arr.size > 1 else 0.0:>9.4f}"
            f"{np.mean(betas[name]):>+11.4f}{np.mean(rhos[name]):>9.4f}"
            f"{gate_on[name]:>7}"
        )

    base = np.array(scores["plain_mc"]).mean()
    print()
    print("variance reduction vs plain_mc (measured, NOT additive):")
    for name, _ in CONFIGS:
        if name == "plain_mc":
            continue
        m = np.array(scores[name]).mean()
        print(f"  {name:<14}{1.0 - m / base:>8.1%}")

    payload = {
        "n": N,
        "budget": BUDGET,
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "configs": {
            name: {
                "l2_rel_mean": float(np.mean(scores[name])),
                "l2_rel_std": float(np.std(scores[name], ddof=1))
                if len(scores[name]) > 1
                else 0.0,
                "beta_hat_mean": float(np.mean(betas[name])),
                "rho_CV_mean": float(np.mean(rhos[name])),
                "f1_active_cells": gate_on[name],
                "per_cell": scores[name],
            }
            for name, _ in CONFIGS
        },
        "note": "F1 and F2 overlap; no combined variance-reduction figure is claimed.",
    }
    out = Path(__file__).resolve().parent / "_logs" / "ablation_f1_f2_f3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
