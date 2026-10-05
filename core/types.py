"""Cross-layer data contracts (architecture spec sections 4 and 5).

This module only *declares* types plus the invariant checks that belong to the
type itself (dtype normalisation, shape agreement, train/val disjointness).
No valuation logic, no IO.

Field order is part of the contract: ``valuation`` and ``eval`` construct these
positionally, so new fields must be appended with defaults, never inserted.

Float64 for feature/label values and int64 for indices is enforced here rather
than trusted, because a stray float32 label array silently changes ranking ties.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from core.errors import (
    EmptyDatasetError,
    NonFiniteValuationError,
    ShapeMismatchError,
    UnsupportedTaskError,
    ensure_no_leakage,
)

__all__ = [
    "CHEAP_CALL_EQUIV",
    "TASK_CLASSIFICATION",
    "TASK_REGRESSION",
    "BenchmarkRow",
    "Budget",
    "Dataset",
    "EvalReport",
    "UtilitySpec",
    "ValuationResult",
]

# Cost of one cheap-surrogate utility call expressed in units of one
# full-fidelity call (spec 4.6.6, measured by repro/audit_scale_and_budget.py).
# A truncated-SGD x10 fit is ~20x cheaper than the full-fidelity oracle.
CHEAP_CALL_EQUIV = 0.05

TASK_CLASSIFICATION = "classification"
TASK_REGRESSION = "regression"
_VALID_TASKS = (TASK_CLASSIFICATION, TASK_REGRESSION)
_TIER0 = "tier0"
_TIER1 = "tier1"


@dataclass(frozen=True, eq=False)
class Dataset:
    """A training pool plus a disjoint validation pool.

    ``X`` is ``(n, d)``; ``y`` is ``(n,)`` int64 for classification and float64
    for regression. ``train_idx`` and ``val_idx`` index into the *same* row space
    and must be disjoint (E204).
    """

    X: np.ndarray
    y: np.ndarray
    name: str
    task: str
    val_idx: np.ndarray
    train_idx: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=np.float64)
        y = np.asarray(self.y)
        if X.ndim != 2:
            raise ShapeMismatchError("X must be 2-D", ndim=int(X.ndim), expected=2)
        if y.ndim != 1:
            raise ShapeMismatchError("y must be 1-D", ndim=int(y.ndim), expected=1)
        if X.shape[0] == 0:
            raise EmptyDatasetError("dataset has zero rows", name=self.name)
        if X.shape[0] != y.shape[0]:
            raise ShapeMismatchError(
                "X and y row counts differ",
                n_x=int(X.shape[0]),
                n_y=int(y.shape[0]),
            )
        if self.task not in _VALID_TASKS:
            raise UnsupportedTaskError(
                "unsupported task", task=self.task, valid=list(_VALID_TASKS)
            )

        y = y.astype(np.int64 if self.task == TASK_CLASSIFICATION else np.float64, copy=False)
        train_idx = np.asarray(self.train_idx, dtype=np.int64).reshape(-1)
        val_idx = np.asarray(self.val_idx, dtype=np.int64).reshape(-1)

        n_rows = int(X.shape[0])
        for label, idx in (("train_idx", train_idx), ("val_idx", val_idx)):
            if idx.size == 0:
                continue
            lo, hi = int(idx.min()), int(idx.max())
            if lo < 0 or hi >= n_rows:
                raise ShapeMismatchError(
                    "index out of range", field=label, n_rows=n_rows, lo=lo, hi=hi
                )

        ensure_no_leakage(train_idx, val_idx, where=f"Dataset({self.name})")

        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "train_idx", train_idx)
        object.__setattr__(self, "val_idx", val_idx)

    @property
    def n_train(self) -> int:
        return int(self.train_idx.size)

    @property
    def n_val(self) -> int:
        return int(self.val_idx.size)

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def X_train(self) -> np.ndarray:
        return self.X[self.train_idx]

    def y_train(self) -> np.ndarray:
        return self.y[self.train_idx]


@dataclass(frozen=True, eq=False)
class Budget:
    """Hard cap on utility evaluations.

    The budget currency is *the number of ``UtilityFn.evaluate`` calls*, counted
    for real by the oracle. Estimating the count from a formula is cheating.
    """

    max_utility_evals: int
    max_wall_seconds: float | None = None

    def __post_init__(self) -> None:
        if int(self.max_utility_evals) <= 0:
            raise ShapeMismatchError(
                "max_utility_evals must be positive", value=int(self.max_utility_evals)
            )
        object.__setattr__(self, "max_utility_evals", int(self.max_utility_evals))

    def remaining(self, used: int) -> int:
        return max(0, int(self.max_utility_evals) - int(used))


@dataclass(frozen=True, eq=False)
class UtilitySpec:
    """Declarative description of a utility model.

    ``params`` stays a plain dict so that JSON round-trips keep key order and the
    spec can be serialised into benchmark.json without custom encoders.
    """

    model: str
    params: dict[str, Any] = field(default_factory=dict)
    tier: str = _TIER1

    VALID_MODELS: ClassVar[tuple[str, ...]] = ("logreg", "knn", "ridge")

    def __post_init__(self) -> None:
        if self.model not in self.VALID_MODELS:
            raise UnsupportedTaskError(
                "unknown utility model", model=self.model, valid=list(self.VALID_MODELS)
            )
        if self.tier not in (_TIER0, _TIER1):
            raise UnsupportedTaskError("unknown tier", tier=self.tier)
        object.__setattr__(self, "params", dict(self.params))

    def describe(self) -> str:
        """Stable single-line description, safe to use as a registry key."""
        inner = ",".join(f"{k}={self.params[k]!r}" for k in sorted(self.params))
        return f"{self.model}({inner})"


@dataclass(frozen=True, eq=False)
class ValuationResult:
    """Output of one valuation run.

    ``values`` is aligned with ``Dataset.train_idx`` order. ``n_utility_evals``
    must be the oracle's real counter delta, not an estimate.
    """

    values: np.ndarray
    method: str
    n_utility_evals: int
    wall_seconds: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    tier: str = _TIER0

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise EmptyDatasetError("valuation vector is empty", method=self.method)
        if not bool(np.all(np.isfinite(values))):
            n_bad = int(np.count_nonzero(~np.isfinite(values)))
            raise NonFiniteValuationError(
                "valuation contains non-finite values", method=self.method, n_bad=n_bad
            )
        if self.tier not in (_TIER0, _TIER1):
            raise UnsupportedTaskError("unknown tier", tier=self.tier, method=self.method)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def n_values(self) -> int:
        return int(self.values.size)


@dataclass(frozen=True, eq=False)
class BenchmarkRow:
    """One (dataset, method, seed) cell of the benchmark grid.

    The budget is reported in **two currencies** because "equal budget" is
    ambiguous, and the ambiguity is not a rounding detail -- it decides which
    method looks cheaper:

    ``n_utility_evals``
        Renamed in meaning, not in name: the **raw** count, i.e. every
        ``utility.evaluate`` call actually made, including calls to cheap
        surrogates. See ``n_calls_raw``, which is the same number under the
        name the JSON contract uses.
    ``n_calls_raw``
        Raw call count. Kept as its own field so the JSON contract in
        ``docs/algorithm_spec.md`` 4.6.5 is satisfied by name.
    ``n_calls_full_equiv``
        Cost expressed in units of ONE full-fidelity call::

            full_equiv = n_full + CHEAP_CALL_EQUIV * n_cheap

        with ``CHEAP_CALL_EQUIV = 0.05`` (spec 4.6.6).

    Both are required. Raw alone overstates ValuaFuse by ~2.8x (measured 5160
    raw vs 1892 equivalent) because most of its extra calls are surrogate calls;
    equivalent alone understates the real call pressure it puts on the oracle.
    Either number alone tells a reader half the story, and in this repository
    half a story has already cost several wrong conclusions.
    """

    dataset: str
    method: str
    seed: int
    metrics: dict[str, float]
    n_utility_evals: int
    wall_seconds: float
    tier: str
    # --- appended with defaults; field order above is a contract ---
    n_calls_raw: int = 0
    n_calls_full_equiv: float = 0.0
    n_calls_full: int = 0
    n_calls_cheap: int = 0

    def __post_init__(self) -> None:
        # ``n_utility_evals`` is the historical name for the raw count; keep the
        # two in sync so a reader of either field sees the same number.
        raw = int(self.n_calls_raw) or int(self.n_utility_evals)
        object.__setattr__(self, "n_calls_raw", raw)
        object.__setattr__(self, "n_utility_evals", raw)
        if not self.n_calls_full_equiv:
            object.__setattr__(
                self,
                "n_calls_full_equiv",
                float(self.n_calls_full + CHEAP_CALL_EQUIV * self.n_calls_cheap),
            )

    def sort_key(self) -> tuple[str, str, int]:
        return (self.dataset, self.method, int(self.seed))


@dataclass(frozen=True, eq=False)
class EvalReport:
    """Aggregated rows plus run-level metadata."""

    rows: tuple[BenchmarkRow, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Sorted once, here, so every downstream consumer (report, json, table)
        # emits keys in the same order and stays diffable (E502).
        object.__setattr__(self, "rows", tuple(sorted(self.rows, key=lambda r: r.sort_key())))
        object.__setattr__(self, "meta", dict(self.meta))

    def methods(self) -> tuple[str, ...]:
        return tuple(sorted({row.method for row in self.rows}))

    def datasets(self) -> tuple[str, ...]:
        return tuple(sorted({row.dataset for row in self.rows}))
