"""Negative controls: estimators that must score badly.

A benchmark needs a floor, and it needs proof that its metric can *fail*. These
two are deliberately not trying to be good:

``rng``
    A uniformly random valuation vector. Any method that cannot beat this is
    not measuring anything.

``shuffle``
    The **true** Shapley values in random order. This one is subtler: it has
    exactly the right values, so a metric that only checks calibration will
    score it perfectly. Only a rank-sensitive metric can tell it apart from
    the truth. The audit measured it at L2Rel 1.5026, which is the concrete
    reason L2Rel alone is not trusted without a rank metric alongside.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from core.interfaces import UtilityFn
from core.types import Budget, Dataset, ValuationResult

__all__ = ["RandomBaseline", "ShuffledTruth"]


class RandomBaseline:
    """Random values in [0, 1). Registered as ``rng_baseline``."""

    name = "rng_baseline"
    is_exact = False
    max_n = None

    @classmethod
    def available(cls) -> bool:
        return True

    def value(
        self,
        data: Dataset,
        utility: UtilityFn | None = None,
        budget: Budget | None = None,
        rng: np.random.Generator | None = None,
    ) -> ValuationResult:
        n = int(data.n_train)
        gen = rng if rng is not None else np.random.default_rng(0)
        return ValuationResult(
            values=gen.random(n),
            method=self.name,
            n_utility_evals=0,
            wall_seconds=0.0,
            tier="tier0",
            diagnostics={"negative_control": True, "n_utility_evals": 0},
        )


class ShuffledTruth:
    """The true values, permuted. Registered as ``shuffle_truth``.

    Consumes no budget: it is handed the gold vector by the pipeline, so it can
    isolate "does the metric notice order?" from "is the estimate accurate?".
    """

    name = "shuffle_truth"
    is_exact = False
    max_n = None

    @classmethod
    def available(cls) -> bool:
        return True

    def __init__(self, truth: np.ndarray | None = None) -> None:
        self._truth = None if truth is None else np.asarray(truth, dtype=np.float64)

    def value(
        self,
        data: Dataset,
        utility: UtilityFn | None = None,
        budget: Budget | None = None,
        rng: np.random.Generator | None = None,
    ) -> ValuationResult:
        if self._truth is None:
            raise ValueError("ShuffledTruth requires the gold vector")
        gen = rng if rng is not None else np.random.default_rng(0)
        permuted = self._truth[gen.permutation(self._truth.size)]
        return ValuationResult(
            values=permuted,
            method=self.name,
            n_utility_evals=0,
            wall_seconds=0.0,
            tier="tier0",
            diagnostics={"negative_control": True, "perfect_calibration": True},
        )
