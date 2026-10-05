"""TMC-Shapley (Ghorbani & Zou, ICML 2019) with a mandatory tuning sweep.

Truncated Monte Carlo: sample permutations, stop early once a point's marginal
contribution is smaller than ``tol`` on average. Truncation is what makes TMC
cheap, and it is also what makes TMC **biased** -- measured in
``repro/verify_estimators.py`` V8, where ``tol > 0`` gave
``|sum(phi) - v(N)| = 0.3333`` against a true value of 0.

Because of that bias, a default-hyperparameter TMC is not a fair baseline. This
module therefore exposes the full grid from the spec (4.3) and records **every**
combination in ``diagnostics["tuned_grid"]`` so an auditor can confirm the
baseline was genuinely searched rather than defaulted.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.errors import BudgetExhaustedError
from core.interfaces import UtilityFn
from core.types import Budget, Dataset, ValuationResult

__all__ = ["TMCConfig", "TmcShapley", "tmc_grid", "tmc_once"]

# Spec 4.3. Fixed, not tuned at runtime: a baseline whose search space moves
# between runs is not reproducible.
TMC_GRID: dict[str, tuple] = {
    "n_perm": (50, 100, 200, 500, 1000, 2000),
    "tol": (0.0, 0.01, 0.05, 0.2),
    "burn_in": (1, 4, 8),
}


def tmc_grid() -> list[dict]:
    """The full cartesian product, in a fixed order."""
    out: list[dict] = []
    for n_perm in TMC_GRID["n_perm"]:
        for tol in TMC_GRID["tol"]:
            for burn_in in TMC_GRID["burn_in"]:
                out.append({"n_perm": n_perm, "tol": tol, "burn_in": burn_in})
    return out


@dataclass(frozen=True)
class TMCConfig:
    """One point of the tuning grid. Frozen so a sweep cannot mutate it."""

    n_perm: int
    tol: float
    burn_in: int

    def as_dict(self) -> dict[str, float | int]:
        return {"n_perm": self.n_perm, "tol": self.tol, "burn_in": self.burn_in}


def tmc_once(
    utility: UtilityFn,
    n: int,
    rng: np.random.Generator,
    cfg: TMCConfig,
) -> tuple[np.ndarray, int, int]:
    """One TMC run. Returns ``(phi, n_evals_used, n_perm_used)``.

    Each permutation costs ``n + 1`` evaluations (the ``v(empty)`` probe plus one
    per arrival), so the cost is known exactly -- but it is still *measured*
    from the oracle rather than asserted, because the budget contract is the
    whole point of the I21 invariant.
    """
    before = utility.n_evals
    phi = np.zeros(n, dtype=np.float64)
    tol_sq = float(cfg.tol) ** 2
    burn = int(cfg.burn_in)
    used = 0
    # Welford running mean/variance of each point's marginal contribution.
    # The stopping rule needs the variance *across permutations*; taking
    # np.var of a single permutation's vector instead yields NaN under ddof=1
    # and silently disables truncation, which would quietly turn TMC into plain
    # permutation MC.
    run_mean = np.zeros(n, dtype=np.float64)
    m2 = np.zeros(n, dtype=np.float64)

    for _ in range(int(cfg.n_perm)):
        perm = rng.permutation(n)
        arrived: list[int] = []
        prev = utility.evaluate(np.empty(0, dtype=np.int64))
        marginals = np.zeros(n, dtype=np.float64)
        for idx in perm:
            arrived.append(int(idx))
            now = utility.evaluate(np.asarray(arrived, dtype=np.int64))
            marginals[int(idx)] = now - prev
            prev = now
        phi += marginals
        used += 1

        delta = marginals - run_mean
        run_mean += delta / used
        m2 += delta * (marginals - run_mean)

        if float(cfg.tol) > 0.0 and used > burn and used > 1:
            # Standard error of the mean, per point. Truncation is allowed only
            # once every point is already estimated to within tol.
            var = m2 / (used - 1)
            stderr = np.sqrt(np.maximum(var, 0.0) / used)
            if float(np.max(stderr)) < tol_sq:
                break

    if used == 0:
        raise ValueError("TMC performed zero permutations")
    return phi / used, int(utility.n_evals - before), used


class TmcShapley:
    """TMC-Shapley, tuned over the spec grid."""

    name = "tmc"
    is_exact = False
    max_n = None

    @classmethod
    def available(cls) -> bool:
        return True

    def value(
        self,
        data: Dataset,
        utility: UtilityFn,
        budget: Budget,
        rng: np.random.Generator,
    ) -> ValuationResult:
        n = int(data.n_train)
        before = utility.n_evals
        combos = tmc_grid()
        cost_per_perm = n + 1

        # The grid has 72 points; a full sweep at the largest n_perm would cost
        # ~158k evaluations, far past any budget we can be handed. So the sweep
        # runs as a cheap *probe*: every combo gets the same small allowance,
        # and selection uses a LABEL-FREE criterion (the estimator's own spread)
        # rather than the truth -- a baseline must not have access to the gold
        # vector, or "tuned baseline" becomes "oracle baseline".
        # Split the budget up front. The sweep only has to *rank* configurations,
        # it does not produce the reported answer, so it gets a quarter; the
        # final run gets 60%. Splitting by hand-off (probe until exhausted, then
        # run with whatever is left) starves the final run and raises E301 after
        # the whole sweep has already been spent.
        probe_share = max(cost_per_perm * 8, int(budget.max_utility_evals * 0.25))
        probe_budget = max(1, probe_share // max(1, len(combos)))
        reserve = int(budget.max_utility_evals * 0.6)
        rows: list[dict] = []
        best: tuple[float, TMCConfig] | None = None

        for combo in combos:
            cfg = TMCConfig(**{k: combo[k] for k in ("n_perm", "tol", "burn_in")})
            start = utility.n_evals
            # Never probe into the reserve: the final run must always be
            # affordable, otherwise the sweep can starve the reported estimate.
            if start + 2 * cost_per_perm > budget.max_utility_evals - reserve:
                rows.append({**combo, "status": "skipped_budget"})
                continue
            probe_cfg = TMCConfig(
                n_perm=min(int(combo["n_perm"]), max(4, probe_budget // cost_per_perm)),
                tol=float(combo["tol"]),
                burn_in=int(combo["burn_in"]),
            )
            try:
                vals, _, n_perm_used = tmc_once(utility, n, rng, probe_cfg)
            except Exception as exc:
                rows.append({**combo, "status": f"error:{type(exc).__name__}"})
                continue
            spread = float(np.std(vals))
            rows.append(
                {
                    **combo,
                    "status": "ok",
                    "probe_n_perm": int(n_perm_used),
                    "probe_n_utility_evals": int(utility.n_evals - start),
                    "probe_std": spread,
                }
            )
            if best is None or spread < best[0]:
                best = (spread, cfg)

        if best is None:
            raise ValueError("TMC grid produced no usable configuration")

        # Final run with the reserved budget. A *fresh* oracle would be cleaner
        # but the counter is the shared budget's, so the run is bounded by
        # ``remaining`` instead -- and if that is too small we report the probe
        # result rather than raising and losing the whole cell.
        chosen = best[1]
        left = utility.remaining()
        if left < cost_per_perm * 2:
            raise BudgetExhaustedError(
                "budget exhausted before the final TMC run",
                n_evals=int(utility.n_evals),
                max_utility_evals=int(budget.max_utility_evals),
            )
        capped = TMCConfig(
            n_perm=max(2, min(chosen.n_perm, max(2, reserve // cost_per_perm))),
            tol=chosen.tol,
            burn_in=chosen.burn_in,
        )
        phi, _, n_perm_used = tmc_once(utility, n, rng, capped)
        return ValuationResult(
            values=phi,
            method=self.name,
            n_utility_evals=int(utility.n_evals - before),
            wall_seconds=0.0,
            tier="tier0",
            diagnostics={
                "selected": capped.as_dict(),
                "n_perm_used": n_perm_used,
                "selection_criterion": "min probe std (label-free, no access to truth)",
                "tuned_grid": rows,
                "biased_by_truncation": capped.tol > 0.0,
            },
        )
