"""ValuaForge unified error-code system (E100~E500).

Contract (architecture spec section 7):

- Every error code maps to one concrete exception class. The code is frozen on the
  class attribute ``code``; it is never assembled at runtime.
- Every exception carries a ``context`` dict for diagnostics. Keys are sorted when
  rendered so logs stay diffable across processes.
- Segment base classes (``ConfigError`` / ``DataError`` / ...) exist only for
  ``except`` clauses. They carry no registrable code.

Codes are also the public API: ``eval.gold`` raises E400, ``eval.metrics`` guards
NaN, and the pipeline raises E500 when two runs at the same seed disagree bitwise.

Author: 晨星
"""

from __future__ import annotations

import hashlib
from typing import Any, ClassVar

import numpy as np

__all__ = [
    "ERROR_CATALOG",
    "BudgetExhaustedError",
    "ConfigError",
    "ConfigSchemaError",
    "CurveBudgetError",
    "DataError",
    "DeterminismError",
    "DuplicateRegistrationError",
    "EmptyDatasetError",
    "EnvParseError",
    "EvalError",
    "GoldDegenerateError",
    "GoldTooLargeError",
    "KeyOrderError",
    "LeakageError",
    "MethodUnavailableError",
    "MetricUndefinedError",
    "NonFiniteValuationError",
    "NotKnnUtilityError",
    "OutputWriteError",
    "PipelineError",
    "ShapeMismatchError",
    "UnknownConfigKeyError",
    "UnknownDatasetError",
    "UnsupportedTaskError",
    "UtilityUndefinedError",
    "ValuaError",
    "ValuationError",
    "ensure_deterministic",
    "ensure_gold_size",
    "ensure_no_leakage",
    "error_class",
    "has_error_code",
]

# Authoritative code -> meaning table. Kept explicit (not derived from docstrings)
# so that ``docs`` and ``tests`` can assert the exact catalogue without parsing prose.
ERROR_CATALOG: dict[str, str] = {
    # E1xx configuration
    "E100": "unknown configuration key",
    "E101": "config schema violation (wrong type or out-of-range value)",
    "E102": "VALUA_* environment variable could not be parsed",
    "E103": "duplicate method name registered (registration-time contract)",
    # E2xx data
    "E200": "empty dataset / n == 0",
    "E201": "X and y shape mismatch, or index out of range",
    "E202": "unknown dataset name (silent fallback forbidden)",
    "E203": "unsupported task type",
    "E204": "train_idx intersects val_idx (leakage guard)",
    # E3xx valuation
    "E300": "valuation method unavailable (missing tier)",
    "E301": "utility evaluation budget exhausted",
    "E302": "valuation produced NaN or Inf",
    "E303": "utility undefined (empty subset or single class)",
    "E304": "closed-form solution requires a KNN utility",
    # E4xx evaluation
    "E400": "gold standard exceeds the allowed problem size",
    "E401": "gold standard degenerate (all values identical)",
    "E402": "metric undefined (length mismatch)",
    "E403": "downstream curve budget insufficient",
    # E5xx pipeline
    "E500": "determinism violation (same seed, runs not bitwise identical)",
    "E501": "failed to write output artifact",
    "E502": "benchmark.json key order is not stable",
}


class ValuaError(Exception):
    """Base class for every ValuaForge error.

    ``code`` is a class attribute so that ``except ValuaError as exc: exc.code``
    is enough to branch on the failure without string matching.
    """

    code: ClassVar[str] = "E000"

    def __init__(self, message: str = "", **context: Any) -> None:
        self.message = message
        self.context: dict[str, Any] = dict(context)
        super().__init__(self._render())

    def _render(self) -> str:
        head = f"[{self.code}] {self.message}" if self.message else f"[{self.code}]"
        if not self.context:
            return head
        detail = ", ".join(f"{k}={self.context[k]!r}" for k in sorted(self.context))
        return f"{head} | {detail}"

    def to_dict(self) -> dict[str, Any]:
        """Serializable form, used when an error is recorded into benchmark.json."""
        return {"code": self.code, "message": self.message, "context": dict(self.context)}

    def fingerprint(self) -> str:
        """Stable 16-hex-char digest of code + message + context."""
        payload = "|".join([self.code, self.message, repr(sorted(self.context.items()))])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --- segment base classes (for ``except`` only, no registrable code) ---


class ConfigError(ValuaError):
    """E1xx segment: configuration and environment."""


class DataError(ValuaError):
    """E2xx segment: dataset construction and integrity."""


class ValuationError(ValuaError):
    """E3xx segment: valuation algorithms and utility oracles."""


class EvalError(ValuaError):
    """E4xx segment: gold standards, metrics and downstream curves."""


class PipelineError(ValuaError):
    """E5xx segment: orchestration, determinism and artifact writing."""


# --- E1xx configuration ---


class UnknownConfigKeyError(ConfigError):
    """A config key or VALUA_* variable is not part of the frozen schema.

    E100 -- Trigger: a config key or ``VALUA_*`` variable is not in the frozen schema.
    """

    code = "E100"


class ConfigSchemaError(ConfigError):
    """A config value has the wrong type or falls outside its allowed range.

    E101 -- Trigger: value parsed but has the wrong type or falls outside its allowed range.
    """

    code = "E101"


class EnvParseError(ConfigError):
    """E102 -- a ``VALUA_*`` variable exists but its text cannot be coerced.

    Trigger: ``VALUA_BUDGET=not-a-number``. Distinct from E101, which means the
    value parsed fine but violates the schema (e.g. ``VALUA_BUDGET=0``).
    """

    code = "E102"


class DuplicateRegistrationError(ConfigError):
    """E103 -- two different callables claim the same registry name.

    Trigger: a second ``@register("tmc")`` for a different class. Raised eagerly
    at decoration/registration time rather than at lookup time, so the collision
    surfaces at import rather than mid-benchmark.

    Kept separate from E300 on purpose: E300 means "this method cannot run in this
    environment", E103 means "the registry itself is inconsistent". Folding them
    together would make a typo look like a missing tier, and the two have
    opposite fixes.
    """

    code = "E103"


# --- E2xx data ---


class EmptyDatasetError(DataError):
    """The dataset carries zero rows; nothing can be valued.

    E200 -- Trigger: the dataset or valuation vector carries zero rows.
    """

    code = "E200"


class ShapeMismatchError(DataError):
    """X and y disagree on row count, or an index array is out of bounds.

    E201 -- Trigger: X and y row counts differ, ndim is wrong, or an index is out of range.
    """

    code = "E201"


class UnknownDatasetError(DataError):
    """The requested dataset name is not registered.

    Raised instead of silently falling back to a default dataset, which would
    silently turn a typo into a valid-looking benchmark row.

    E202 -- Trigger: the requested dataset name is not registered. Silent fallback is forbidden -- a typo must not become a valid-looking benchmark row.
    """

    code = "E202"


class UnsupportedTaskError(DataError):
    """The task is neither ``classification`` nor ``regression``.

    E203 -- Trigger: task is neither ``classification`` nor ``regression``, or a tier/model string is unrecognised.
    """

    code = "E203"


class LeakageError(DataError):
    """train_idx and val_idx intersect: validation data leaked into training.

    E204 -- Trigger: ``train_idx`` intersects ``val_idx``. Validation data leaked into training.
    """

    code = "E204"


# --- E3xx valuation ---


class MethodUnavailableError(ValuationError):
    """E300 -- the method exists in the registry but cannot run here.

    Trigger: ``available()`` returned False (missing tier), or the name is not
    registered at all. This code has exactly one meaning -- "not usable now" --
    so a caller can retry after installing a tier without inspecting the message.
    Registry corruption is E103, not E300.
    """

    code = "E300"


class BudgetExhaustedError(ValuationError):
    """The utility evaluation budget is exhausted; silent truncation is forbidden.

    E301 -- Trigger: utility evaluations would exceed ``max_utility_evals``. Silent truncation is forbidden.
    """

    code = "E301"


class NonFiniteValuationError(ValuationError):
    """A valuation vector contains NaN or Inf.

    E302 -- Trigger: a valuation vector contains NaN or Inf.
    """

    code = "E302"


class UtilityUndefinedError(ValuationError):
    """The utility is undefined for this subset (empty subset or a single class).

    E303 -- Trigger: the utility is undefined for this subset (empty subset, or a single class).
    """

    code = "E303"


class NotKnnUtilityError(ValuationError):
    """The closed-form KNN-Shapley solution was requested on a non-KNN utility.

    E304 -- Trigger: the closed-form KNN-Shapley solution was requested on a non-KNN utility.
    """

    code = "E304"


# --- E4xx evaluation ---


class GoldTooLargeError(EvalError):
    """The exact gold standard was requested on a problem larger than allowed.

    Anchor A costs ``2 ** n`` utility evaluations. Silently approximating would
    turn the gold standard into just another estimate, which defeats its purpose.
    """

    code = "E400"


class GoldDegenerateError(EvalError):
    """The gold standard collapsed to a constant, so rank metrics are undefined.

    E401 -- Trigger: the gold standard collapsed to a constant, so rank metrics are undefined.
    """

    code = "E401"


class MetricUndefinedError(EvalError):
    """Inputs of unequal length make the metric undefined.

    E402 -- Trigger: inputs of unequal length make the metric undefined.
    """

    code = "E402"


class CurveBudgetError(EvalError):
    """The downstream removal curve does not have enough budget for its checkpoints.

    E403 -- Trigger: the downstream removal curve lacks budget for its checkpoints.
    """

    code = "E403"


# --- E5xx pipeline ---


class DeterminismError(PipelineError):
    """Two runs at the same seed produced different bits."""

    code = "E500"


class OutputWriteError(PipelineError):
    """An output artifact could not be written.

    E501 -- Trigger: an output artifact could not be written.
    """

    code = "E501"


class KeyOrderError(PipelineError):
    """E502 -- benchmark.json key order drifted.

    Trigger: serialising the same run twice produced different key order, so
    ``git diff --exit-code`` on the artifact is no longer a valid gate.
    """

    code = "E502"


def _discover_registry() -> dict[str, type[ValuaError]]:
    """Map code -> concrete class by walking this module's namespace once.

    A class counts as concrete only when ``code`` appears in its own ``__dict__``;
    segment bases inherit ``code`` and are therefore skipped automatically. Using
    ``__dict__`` instead of a name list means renaming a class can never silently
    drop its code from the registry.

    Guards two distinct collisions, because either one silently loses an error code:

    1. two classes claiming the same ``code``;
    2. two classes claiming the same **name**. A redefinition rebinds the name in
       ``globals()``, so the earlier class becomes unreachable -- every ``except``
       clause and ``error_class()`` lookup for it would fail at runtime with no
       import-time signal. (Hit for real: E103 was shadowed by a second
       ``DuplicateRegistrationError`` before this guard existed.)
    """
    found: dict[str, type[ValuaError]] = {}
    seen_names: dict[str, type[ValuaError]] = {}
    for name, obj in list(globals().items()):
        if not isinstance(obj, type) or not issubclass(obj, ValuaError):
            continue
        if "code" not in vars(obj):
            continue
        if name in seen_names:
            raise RuntimeError(
                f"duplicate error class name {name!r}: "
                f"{seen_names[name]} (code {seen_names[name].code}) vs {obj} "
                f"(code {obj.code}). Two definitions of one name shadow each other."
            )
        seen_names[name] = obj
        code = obj.code
        if code in found and found[code] is not obj:
            raise RuntimeError(f"duplicate error code {code}: {found[code]} vs {obj}")
        found[code] = obj
    return found


_REGISTRY: dict[str, type[ValuaError]] = _discover_registry()


def error_class(code: str) -> type[ValuaError]:
    """Return the exception class registered for ``code``.

    Raises:
        KeyError: if the code is not part of ``ERROR_CATALOG``.
    """
    try:
        return _REGISTRY[code]
    except KeyError:
        raise KeyError(f"unregistered error code: {code!r}") from None


def has_error_code(code: str) -> bool:
    """Whether ``code`` is both catalogued and bound to a concrete class."""
    return code in ERROR_CATALOG and code in _REGISTRY


# --- reusable guards (imported by data / eval / pipeline layers) ---


def ensure_no_leakage(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    where: str = "Dataset",
) -> None:
    """Fail loudly when validation indices appear in the training indices.

    Architecture spec section 10 makes this a hard guard rather than a warning:
    a leaked validation set turns every downstream number into fiction.
    """
    train = np.asarray(train_idx).reshape(-1)
    val = np.asarray(val_idx).reshape(-1)
    if train.size == 0 or val.size == 0:
        return
    overlap = np.intersect1d(train, val, assume_unique=False)
    if overlap.size == 0:
        return
    raise LeakageError(
        "train_idx intersects val_idx",
        where=where,
        n_overlap=int(overlap.size),
        sample=[int(v) for v in overlap[:8]],
    )


def ensure_gold_size(n: int, max_n: int, *, anchor: str = "A") -> None:
    """Fail loudly when the exact gold standard is asked for beyond ``max_n``.

    Anchor A costs ``2 ** n`` utility evaluations. Silently approximating would
    turn the gold standard into just another estimate, which defeats its purpose.
    """
    if n <= max_n:
        return
    raise GoldTooLargeError(
        "gold standard problem size exceeds the configured limit",
        anchor=anchor,
        n=int(n),
        max_n=int(max_n),
        utility_evals=int(2**n) if n < 1024 else -1,
    )


def ensure_deterministic(
    label: str,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> None:
    """Raise E500 unless ``candidate`` matches ``reference`` bit for bit.

    Comparison uses raw bytes rather than ``allclose`` on purpose: the
    determinism gate is about reproducibility, so a 1e-12 drift must fail.
    """
    ref = np.ascontiguousarray(reference)
    cand = np.ascontiguousarray(candidate)
    if ref.shape != cand.shape or ref.dtype != cand.dtype:
        raise DeterminismError(
            "run is not reproducible",
            label=label,
            reason="shape_or_dtype",
            ref_shape=str(ref.shape),
            cand_shape=str(cand.shape),
            ref_dtype=str(ref.dtype),
            cand_dtype=str(cand.dtype),
        )
    if ref.tobytes() != cand.tobytes():
        diff = int(np.count_nonzero(ref != cand))
        raise DeterminismError(
            "run is not reproducible",
            label=label,
            reason="bitwise_mismatch",
            n_differing=int(diff),
            n_total=int(ref.size),
        )
