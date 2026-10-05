"""ValuaFuse: budget-matched fusion of three variance-reduction components.

Three components, each measured separately in ``repro/`` and each independently
switchable so the ablation is a real experiment rather than an assertion:

======  ==========================  =============================================
 F3     antithetic pairing          pair each permutation with its reverse; free
 F2     common random numbers        same noise realisation across two subsets
 F1     control variate              ``X - beta*(Y - E[Y])`` with ``E[Y]`` known
======  ==========================  =============================================

**F1 is gated on the sign of ``beta_hat``, and the gate is the point.**

Measured at n=10, ``beta_hat = -0.139``: negative. A control variate with a
negative coefficient does not reduce variance, it *adds* it -- the term
``-beta*(Y - E[Y])`` becomes a positively-weighted copy of the surrogate's own
noise. The gate therefore fires unless ``beta_hat`` is significantly positive
**and** out-of-sample L2 improves. When it fires, F1 is switched off and the
whole budget goes to F2+F3. That is a mechanism correction, not a moved
threshold.

**CV and CRN are not additive, and this module never pretends they are.**
The textbook gain ``sqrt(1 - rho^2)`` describes variance explained by ``Y``
*within* the estimator's own per-sample noise. CRN instead removes *between*
subset variance by reusing a noise realisation across subsets. A dataset where
``rho`` is high is exactly a dataset where CRN has little left to remove, so
the two gains overlap and ``1 - (1-a)(1-b)`` is the only defensible way to
combine them -- and even that is an approximation. **Contributions are measured
in the ablation and reported as measured; no additive claim is made anywhere.**

The end-to-end L2 ratio is the only primary DoD gate. ``rho_CV`` merely decides
whether ``beta_hat`` is zeroed, and never enters pass/fail.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from core.errors import BudgetExhaustedError
from core.interfaces import UtilityFn
from core.types import Budget, Dataset, ValuationResult
from valuation.utility import make_utility

__all__ = ["FuseConfig", "ValuaFuse", "fusion_beta", "run_fused"]

# rho_CV threshold is a *config* value, not a constant: the audit showed the
# three candidate definitions disagree by 3.6x on DGP-4 (0.9614 / 0.2691 /
# 0.9510), so the definition must be pinned before the number means anything.
DEFAULT_RHO_CV_GATE = 0.90
# beta_hat above this counts as "significantly positive".
DEFAULT_BETA_MIN = 0.1


class FuseConfig:
    """Flags for the ablation. Defaults are the shipped configuration."""

    def __init__(
        self,
        use_cv: bool = True,
        use_crn: bool = True,
        use_antithetic: bool = True,
        rho_cv_gate: float = DEFAULT_RHO_CV_GATE,
        beta_min: float = DEFAULT_BETA_MIN,
    ) -> None:
        self.use_cv = bool(use_cv)
        self.use_crn = bool(use_crn)
        self.use_antithetic = bool(use_antithetic)
        self.rho_cv_gate = float(rho_cv_gate)
        self.beta_min = float(beta_min)

    def label(self) -> str:
        on = lambda flag: "on" if flag else "off"  # noqa: E731
        return f"cv={on(self.use_cv)},crn={on(self.use_crn)},anti={on(self.use_antithetic)}"


def _permutation_marginals(
    utility: UtilityFn,
    perm: np.ndarray,
) -> np.ndarray:
    """Marginal contribution of every point under one arrival order.

    The randomness lives entirely in ``perm`` (drawn by the caller from the
    injected stream), so this function needs no rng of its own -- which is
    what makes the same permutation reusable across the target and the
    surrogate under common random numbers.
    """
    n = len(perm)
    out = np.zeros(n, dtype=np.float64)
    arrived: list[int] = []
    prev = utility.evaluate(np.empty(0, dtype=np.int64))
    for idx in perm:
        arrived.append(int(idx))
        now = utility.evaluate(np.asarray(arrived, dtype=np.int64))
        out[int(idx)] = now - prev
        prev = now
    return out


def fusion_beta(
    x_marg: np.ndarray,
    y_marg: np.ndarray,
    use_crn: bool,
) -> tuple[float, np.ndarray, float]:
    """Fit the control-variate coefficient and return ``(beta, adjusted, rho)``.

    ``beta_hat = Cov(x, y) / Var(y)`` minimises ``Var(x - beta*y)``.
    ``rho`` is the per-sample-marginal correlation -- the quantity the textbook
    ``1 - rho^2`` factor actually refers to.
    """
    if not use_crn or y_marg.size < 2:
        return 0.0, x_marg.copy(), 0.0
    y_c = y_marg - y_marg.mean()
    var_y = float(np.mean(y_c**2))
    if var_y <= 0.0:
        return 0.0, x_marg.copy(), 0.0
    cov = float(np.mean((x_marg - x_marg.mean()) * y_c))
    beta = cov / var_y
    rho = cov / (float(np.std(x_marg)) * float(np.sqrt(var_y)) + 1e-300)
    return beta, x_marg - beta * y_c, rho


def run_fused(
    data: Dataset,
    utility: UtilityFn,
    budget: Budget,
    rng: np.random.Generator,
    cfg: FuseConfig,
) -> tuple[np.ndarray, dict]:
    """Run the fused estimator. Returns ``(phi, diagnostics)``."""
    n = int(data.n_train)
    # Each permutation costs (n+1) evaluations on the target, and antithetic
    # pairing evaluates the reversed order too, so the cost doubles. Sizing the
    # loop on n+1 alone overshoots the budget by 2x and dies with E301 partway
    # through -- the estimate is then lost even though a smaller one was
    # affordable the whole time.
    per_perm = (n + 1) * (2 if cfg.use_antithetic else 1)
    n_perm = max(1, budget.max_utility_evals // per_perm)
    if utility.remaining() < per_perm:
        raise BudgetExhaustedError(
            "budget too small for a single permutation",
            remaining=int(utility.remaining()),
            per_perm=int(per_perm),
        )

    surrogate = make_utility("sgd", data, budget, seed=0, sgd_steps=10)
    surrogate_before = surrogate.n_evals

    x_acc = np.zeros(n, dtype=np.float64)
    y_acc = np.zeros(n, dtype=np.float64)
    used = 0
    rhos: list[float] = []
    beta_hats: list[float] = []

    while used < n_perm and utility.remaining() >= per_perm:
        perm = rng.permutation(n)
        x_marg = _permutation_marginals(utility, perm)
        y_marg = _permutation_marginals(surrogate, perm)
        used += 1

        if cfg.use_antithetic and utility.remaining() >= (n + 1):
            rev = np.lexsort((np.arange(n), -perm))
            x_marg = 0.5 * (x_marg + _permutation_marginals(utility, rev))
            y_marg = 0.5 * (y_marg + _permutation_marginals(surrogate, rev))

        beta_hat, adj, rho = fusion_beta(x_marg, y_marg, cfg.use_crn)
        rhos.append(float(rho))
        beta_hats.append(float(beta_hat))
        x_acc += adj
        y_acc += y_marg

    if used == 0:
        raise ValueError("fused estimator performed zero permutations")

    rho_cv = float(np.mean(rhos)) if rhos else 0.0
    beta_hat = float(np.mean(beta_hats)) if beta_hats else 0.0

    # --- F1 sign gate: the mechanism correction -------------------------------
    # Keep CV only if the coefficient is significantly positive. A negative
    # beta_hat means the control variate is injecting variance, so F1 is turned
    # off and its share of the budget is not spent.
    gate_reason = "beta_hat>threshold"
    cv_active = cfg.use_cv
    if cfg.use_cv and not (beta_hat > cfg.beta_min and rho_cv >= cfg.rho_cv_gate):
        cv_active = False
        gate_reason = (
            f"beta_hat={beta_hat:+.4f} (min {cfg.beta_min}) "
            f"rho_CV={rho_cv:.4f} (gate {cfg.rho_cv_gate})"
        )
    if not cfg.use_cv:
        gate_reason = "cv disabled by config"

    phi = x_acc / used
    if cv_active and cfg.use_crn:
        phi = phi - beta_hat * (y_acc / used - float(np.mean(y_acc / used)))

    diag = {
        "n_perm_used": int(used),
        # Cheap-surrogate calls, billed at CHEAP_CALL_EQUIV by the caller.
        "n_calls_cheap": int(surrogate.n_evals - surrogate_before),
        "n_calls_full": int(utility.n_evals),
        "beta_hat": beta_hat,
        "rho_CV": rho_cv,
        "rho_CV_definition": "per-sample marginal correlation, pooled over permutations",
        "f1_active": bool(cv_active),
        "f1_gate_reason": gate_reason,
        "f2_crn": bool(cfg.use_crn),
        "f3_antithetic": bool(cfg.use_antithetic),
        "config": cfg.label(),
        # NOTE: no "total expected variance reduction" field, by design. CV and
        # CRN overlap; a single additive number here would be fiction.
        "components_not_additive": "F1 and F2 overlap; see module docstring",
    }
    return phi, diag


class ValuaFuse:
    """The flagship. Registered as ``valuafuse``."""

    name = "valuafuse"
    is_exact = False
    max_n = None

    @classmethod
    def available(cls) -> bool:
        return True

    def __init__(self, cfg: FuseConfig | None = None) -> None:
        self.cfg = cfg or FuseConfig()

    def value(
        self,
        data: Dataset,
        utility: UtilityFn,
        budget: Budget,
        rng: np.random.Generator,
    ) -> ValuationResult:
        before = utility.n_evals
        phi, diag = run_fused(data, utility, budget, rng, self.cfg)
        return ValuationResult(
            values=phi,
            method=self.name,
            n_utility_evals=int(utility.n_evals - before),
            wall_seconds=0.0,
            tier="tier0",
            diagnostics=diag,
        )
