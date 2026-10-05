"""I26-I27: invariants that belong to the flagship alone.

These two exist because of measured failures, not because they sounded prudent.

**I26 (F1 sign gate is self-consistent).** An earlier draft of this project
reported "beta_hat came out negative, so the control variate was adding noise"
as if it were a property of the method. It was an implementation bug: the
cross-fit folded the permutation matrix into a length-n vector and then sliced
it by permutation index, producing an EMPTY second fold, hence Var(dY) = 0.
After the fix beta_hat is positive (+0.43..+0.49) across all nine (n, seed)
configurations. The test pins the gate so a future negative beta_hat cannot be
narrated as a finding without first being reported as a possible bug.

**I27 (the surrogate must actually be cheaper).** This is the expensive one.
The budget currency is the number of ``utility.evaluate`` calls, and both the
target and the surrogate cost exactly one unit per call. So the flagship's
premise -- "move budget onto a cheap surrogate" -- is not measurable in budget
units at all; it is identically 1.0x. On the KNN track the surrogate is in fact
*dearer* in wall clock (measured: sgd 185.94 us/call vs knn 67.21 us/call, 2.77x),
and ``examples/run_demo.py`` used to print PASS 0.4351 on that basis. The number
was an accounting artefact comparing two different currencies.

I27 pins the ratio so that a track with c >= 1 cannot silently produce a
performance claim.

Author: 晨星
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from core.types import Budget
from data.dgp import generate
from valuation.utility import make_utility
from valuation.valuafuse import FuseConfig, ValuaFuse, fusion_beta

# Cost per evaluate() call in microseconds, measured on this machine at
# gaussian_mixture / n_train=10 / d=6 / seed=7 over 200 calls. Recomputed by the
# test below with a 5x tolerance band rather than asserted exactly: absolute
# timings depend on the host, and a test that fails on a slow machine teaches
# nothing. The RATIO is what I27 constrains, and the ratio is far more stable
# than either absolute number.
MEASURED_US_PER_CALL = {"knn": 67.21, "sgd": 185.94, "logreg": 394.76}


def _subsets(data, n_calls: int = 200, seed: int = 0) -> list[np.ndarray]:
    """Random subsets. For logreg, only those with >= 2 classes (else E303)."""
    ytr = data.y[data.train_idx]
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    while len(out) < n_calls:
        k = int(rng.integers(2, data.n_train + 1))
        s = np.sort(rng.permutation(data.n_train)[:k])
        if np.unique(ytr[s]).size >= 2:
            out.append(s)
    return out


def _us_per_call(kind: str, data, subs: list[np.ndarray]) -> float:
    util = make_utility(kind, data, Budget(1 << 20), seed=7, k=5)
    t0 = time.perf_counter()
    for s in subs:
        util.evaluate(s)
    elapsed = time.perf_counter() - t0
    assert util.n_evals == len(subs), "meter must not undercount"
    return elapsed / util.n_evals * 1e6


class TestI26F1SignGate:
    """F1 must never be active on a non-positive beta."""

    def test_active_cv_implies_positive_beta(self) -> None:
        """cv_active => beta_hat > beta_min, read off a real end-to-end run."""
        data = generate("gaussian_mixture", n=30, d=6, seed=7, n_train=10)
        util = make_utility("knn", data, Budget(1 << 16), seed=7, k=5)
        result = ValuaFuse(FuseConfig()).value(
            data, util, Budget(4096), np.random.default_rng(7)
        )
        diag = result.diagnostics
        beta_hat = float(diag["beta_hat"])
        cfg = FuseConfig()
        if diag.get("f1_active"):
            assert beta_hat > cfg.beta_min, (
                f"F1 reported active with beta_hat={beta_hat:+.4f} <= beta_min={cfg.beta_min}"
            )
        else:
            # Being off is the normal state on this track; the reason must be
            # recorded, never silently absent.
            assert diag.get("f1_gate_reason"), "f1_active=False needs a recorded reason"

    def test_negative_beta_forces_cv_off(self) -> None:
        """A hand-made negative beta must switch F1 off, not be averaged away."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = -x  # perfectly anti-correlated
        beta, _, rho = fusion_beta(x, y, use_crn=True)
        assert beta < 0.0, "anti-correlation must yield a negative coefficient"
        assert rho < 0.0
        cfg = FuseConfig(beta_min=0.1)
        gate_open = beta > cfg.beta_min and rho >= cfg.rho_cv_gate
        assert not gate_open, "a negative beta must never open the F1 gate"

    def test_perfect_correlation_gives_beta_one(self) -> None:
        """Sanity anchor: beta_hat = 1 when x == y (variance-minimising fit)."""
        x = np.array([0.5, 1.5, 2.5, 3.5])
        beta, adjusted, _ = fusion_beta(x, x.copy(), use_crn=True)
        assert beta == pytest.approx(1.0, abs=1e-12)
        # x - 1.0*(x - mean(x)) collapses to the constant mean: zero variance.
        assert np.allclose(adjusted, adjusted[0])


class TestI27SurrogateMustBeCheaper:
    """The flagship's premise, measured rather than asserted."""

    def test_knn_track_surrogate_is_not_cheaper(self) -> None:
        """On the KNN track the hardcoded sgd surrogate is DEARER, not cheaper.

        This is the finding that voided the demo's PASS 0.4351. If a future
        change makes the surrogate genuinely cheaper on this track, this test
        should fail -- and the I27 claim in the spec should be re-measured, not
        the test deleted.
        """
        data = generate("gaussian_mixture", n=30, d=6, seed=7, n_train=10)
        subs = _subsets(data)
        knn_us = _us_per_call("knn", data, subs)
        sgd_us = _us_per_call("sgd", data, subs)
        ratio = sgd_us / knn_us

        # Sanity: the absolute numbers should be in the right ballpark, so a
        # silently-degenerate timing (e.g. 0.001 us/call) cannot pass unnoticed.
        assert 5.0 < knn_us < 5000.0, f"implausible knn cost {knn_us:.2f} us/call"
        assert 0.5 < ratio < 20.0, f"implausible cost ratio {ratio:.2f}"

        assert ratio >= 1.0, (
            f"surrogate/target = {ratio:.2f}x: if this is now < 1 the I27 premise "
            "holds on this track and the spec's VOID verdict must be re-derived"
        )

    def test_budget_currency_makes_both_cost_one_unit(self) -> None:
        """The structural half of I27: in budget units the ratio is exactly 1.

        This is why the accounting artefact was possible in the first place.
        Both oracles are metered by I21 the same way, so no measurement taken
        *in budget units* can ever distinguish them.
        """
        data = generate("gaussian_mixture", n=30, d=6, seed=7, n_train=10)
        subs = _subsets(data)
        for kind in ("knn", "sgd"):
            util = make_utility(kind, data, Budget(1 << 20), seed=7, k=5)
            for s in subs:
                util.evaluate(s)
            assert util.n_evals == len(subs), f"{kind} meter disagrees with call count"

    def test_logreg_is_the_most_expensive_oracle(self) -> None:
        """logreg is ~6x knn and ~2x sgd; it is the c<1 track the audit used."""
        data = generate("gaussian_mixture", n=30, d=6, seed=7, n_train=10)
        subs = _subsets(data)
        knn_us = _us_per_call("knn", data, subs)
        logreg_us = _us_per_call("logreg", data, subs)
        assert logreg_us > knn_us, "logreg must cost more than knn"
        # The audit's c = 0.048..0.053 was measured with an L-BFGS oracle. Our
        # logreg is cheaper than that oracle, but the point stands: only on an
        # expensive-oracle track is a cheap surrogate worth anything.
        assert logreg_us / knn_us > 2.0


class TestI27Documentation:
    """I27 must be stated in the spec, not only enforced in code."""

    @pytest.mark.parametrize(
        "doc",
        [
            "docs/algorithm_spec.md",
            "docs/architecture_spec.md",
        ],
    )
    def test_spec_documents_the_gate(self, doc: str) -> None:
        text = Path(doc).read_text(encoding="utf-8")
        assert "I27" in text, f"{doc} does not mention the I27 premise gate"
