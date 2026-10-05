"""Structural contracts (Protocols) shared across layers.

These are duck-typed on purpose: ``valuation`` and ``eval`` must be able to
substitute their own implementations without importing anything from ``core``
beyond these types.

Field and parameter order is part of the contract. Callers pass positionally, so
reordering a Protocol's members silently changes meaning rather than raising.

Author: 晨星
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from core.types import Budget, Dataset, EvalReport, ValuationResult

__all__ = [
    "DGP",
    "Evaluator",
    "GoldStandard",
    "UtilityFn",
    "ValuationMethod",
]


@runtime_checkable
class UtilityFn(Protocol):
    """Oracle mapping a training subset to a scalar utility.

    The budget currency is *the number of times :meth:`evaluate` is called*. The
    count must be real: read ``n_evals`` before and after a run and subtract.
    Deriving the count from a formula is how benchmarks quietly overstate
    efficiency, so it is forbidden.
    """

    @property
    def n_train(self) -> int:
        """Number of trainable points the oracle accepts."""

    @property
    def n_evals(self) -> int:
        """Cumulative number of :meth:`evaluate` calls since construction."""

    def evaluate(self, subset: np.ndarray) -> float:
        """Utility of ``subset`` (ascending int64 indices); larger is better."""

    def remaining(self, budget: Budget) -> int:
        """How many evaluations are still affordable under ``budget``."""


@runtime_checkable
class ValuationMethod(Protocol):
    """A valuation algorithm.

    Implementations receive an already-derived ``rng`` and must never construct
    their own: two methods sharing a seed must not share a noise stream.
    """

    name: str

    @classmethod
    def available(cls) -> bool:
        """Whether this method can run here (tier probe, no heavy import)."""

    def value(
        self,
        data: Dataset,
        utility: UtilityFn,
        budget: Budget,
        rng: np.random.Generator,
    ) -> ValuationResult:
        """Value every training point under ``budget``."""


@runtime_checkable
class GoldStandard(Protocol):
    """Exact reference values used to score estimates."""

    name: str
    is_exact: bool
    """True when mathematically exact; False means reference-only."""

    max_n: int | None
    """Largest ``n`` this gold standard accepts; ``None`` means unbounded."""

    def values(self, data: Dataset, utility: UtilityFn) -> np.ndarray:
        """Exact values aligned with ``data.train_idx``."""


@runtime_checkable
class DGP(Protocol):
    """Synthetic data generator with a known quality structure."""

    name: str

    def generate(self, n: int, seed: int) -> Dataset:
        """Build a reproducible dataset of ``n`` rows."""


@runtime_checkable
class Evaluator(Protocol):
    """Scores one valuation output against an optional gold standard."""

    name: str

    def run(
        self,
        data: Dataset,
        values: dict[str, np.ndarray],
        gold: np.ndarray | None,
        budget: Budget,
    ) -> EvalReport:
        """Produce the evaluation report for one dataset."""
