"""Configuration: defaults <- ``VALUA_*`` environment <- explicit overrides.

Precedence is CLI > environment > defaults. Values are validated against a
declared schema and a violation raises E101 -- never a silent coercion, because a
silently clamped budget would produce a benchmark that claims a precision the run
never paid for.

Author: 晨星
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from core.errors import ConfigSchemaError, EnvParseError, UnknownConfigKeyError

__all__ = ["DEFAULTS", "ENV_PREFIX", "Config", "config_from_env"]

ENV_PREFIX = "VALUA_"

# Task-pairs kept short: the architecture lists eight DGPs, but the default grid
# runs three representative ones so `pytest` and the demo stay inside budget.
DEFAULTS: dict[str, Any] = {
    "seed": 7,
    "seeds": (7, 17, 2027),
    "datasets": ("gaussian_mixture", "sparse_linear", "label_noise"),
    "methods": (),
    "budget": 4096,
    "gold_max_n": 12,
    # knn, not logreg: under the pinned E303 contract a one-class coalition is
    # UNDEFINED for logreg, and every permutation walks through singletons -- so
    # logreg cannot be a permutation target at all. benchmark.py and
    # run_demo.py both hardcode knn; making it the default too means the
    # configured default and the code paths finally agree.
    "utility_model": "knn",
    "knn_k": 5,
    "beta_alpha": 1.0,
    "beta_beta": 4.0,
    "influence_damping": 1e-3,
    "oob_n_estimators": 64,
    "removal_steps": 20,
    "noise_frac": 0.15,
    "output_dir": "results",
    "n_jobs": 1,
}

#: Single source of truth for utility-oracle names. ``valuation.utility`` asserts
#: its dispatch against this, because the two lists previously disagreed in both
#: directions -- config accepted "ridge" (no factory branch -> E303) and rejected
#: "sgd" (which worked). See tests/test_utility_surface.py.
SUPPORTED_UTILITY_MODELS = ("knn", "logreg", "sgd")
_UTILITY_MODELS = SUPPORTED_UTILITY_MODELS


@dataclass(frozen=True, eq=False)
class Config:
    """Validated run configuration.

    ``methods=()`` means "every method the registry reports as available", which
    keeps the default grid from hard-coding an algorithm list that the valuation
    layer still owns.
    """

    seed: int = DEFAULTS["seed"]
    seeds: tuple[int, ...] = DEFAULTS["seeds"]
    datasets: tuple[str, ...] = DEFAULTS["datasets"]
    methods: tuple[str, ...] = DEFAULTS["methods"]
    budget: int = DEFAULTS["budget"]
    gold_max_n: int = DEFAULTS["gold_max_n"]
    utility_model: str = DEFAULTS["utility_model"]
    knn_k: int = DEFAULTS["knn_k"]
    beta_alpha: float = DEFAULTS["beta_alpha"]
    beta_beta: float = DEFAULTS["beta_beta"]
    influence_damping: float = DEFAULTS["influence_damping"]
    oob_n_estimators: int = DEFAULTS["oob_n_estimators"]
    removal_steps: int = DEFAULTS["removal_steps"]
    noise_frac: float = DEFAULTS["noise_frac"]
    output_dir: str = DEFAULTS["output_dir"]
    n_jobs: int = DEFAULTS["n_jobs"]

    FIELDS: ClassVar[tuple[str, ...]] = tuple(DEFAULTS)

    def __post_init__(self) -> None:
        for name in self.FIELDS:
            object.__setattr__(self, name, _coerce(name, getattr(self, name)))

    def replace(self, **overrides: Any) -> Config:
        """Return a validated copy with ``overrides`` applied."""
        unknown = sorted(set(overrides) - set(self.FIELDS))
        if unknown:
            raise UnknownConfigKeyError("unknown config key", keys=unknown)
        merged = {name: getattr(self, name) for name in self.FIELDS}
        merged.update(overrides)
        return Config(**merged)

    def as_dict(self) -> dict[str, Any]:
        """Plain dict in declared field order (stable JSON key order, E502)."""
        return {name: getattr(self, name) for name in self.FIELDS}


def _spec(name: str) -> tuple[type, Any, Any, Any]:
    """Return ``(kind, minimum, maximum, choices)`` for a field."""
    table: dict[str, tuple[type, Any, Any, Any]] = {
        "seed": (int, 0, 2**32 - 1, None),
        "seeds": (tuple, None, None, None),
        "datasets": (tuple, None, None, None),
        "methods": (tuple, None, None, None),
        "budget": (int, 1, 2**31 - 1, None),
        "gold_max_n": (int, 1, 22, None),
        "utility_model": (str, None, None, _UTILITY_MODELS),
        "knn_k": (int, 1, 10_000, None),
        "beta_alpha": (float, 1e-3, 1e3, None),
        "beta_beta": (float, 1e-3, 1e3, None),
        "influence_damping": (float, 1e-12, 1e3, None),
        "oob_n_estimators": (int, 1, 10_000, None),
        "removal_steps": (int, 1, 10_000, None),
        "noise_frac": (float, 0.0, 0.9, None),
        "output_dir": (str, None, None, None),
        # Windows joblib IPC crashes under concurrency (pitfall library D), so the
        # schema caps n_jobs at 1 on this platform instead of trusting the caller.
        "n_jobs": (int, 1, 1 if os.name == "nt" else 64, None),
    }
    return table[name]


def _coerce(name: str, value: Any) -> Any:
    """Normalise and range-check one field. Raises E101 on any violation."""
    kind, low, high, choices = _spec(name)
    try:
        if kind is tuple:
            out = tuple(value)
        elif kind is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigSchemaError("expected int", field=name, got=type(value).__name__)
            out = int(value)
        elif kind is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigSchemaError("expected float", field=name, got=type(value).__name__)
            out = float(value)
            if not math.isfinite(out):
                raise ConfigSchemaError("expected a finite float", field=name, got=repr(value))
        else:
            if not isinstance(value, str):
                raise ConfigSchemaError("expected str", field=name, got=type(value).__name__)
            out = value
    except ConfigSchemaError:
        raise
    except TypeError as exc:
        raise ConfigSchemaError("cannot interpret value", field=name, got=repr(value)) from exc

    if low is not None and out < low:
        raise ConfigSchemaError("value below minimum", field=name, value=out, min=low)
    if high is not None and out > high:
        raise ConfigSchemaError("value above maximum", field=name, value=out, max=high)
    if choices is not None and out not in choices:
        raise ConfigSchemaError(
            "value not allowed", field=name, value=out, allowed=list(choices)
        )
    # `methods=()` is legal and means "every method the registry reports available",
    # so an empty tuple must not be rejected the way an empty dataset list is.
    if name in {"seeds", "datasets"} and not out:
        raise ConfigSchemaError("sequence must be non-empty", field=name)
    if name == "seeds":
        for item in out:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ConfigSchemaError("seeds must contain ints", field=name, entry=repr(item))
    if name in {"datasets", "methods"}:
        for item in out:
            if not isinstance(item, str):
                raise ConfigSchemaError(
                    "expected str entry in sequence", field=name, entry=repr(item)
                )
            if not item.strip():
                raise ConfigSchemaError("blank entry in sequence", field=name, entry=repr(item))
    if name == "output_dir" and not out.strip():
        raise ConfigSchemaError("output_dir must be non-empty", field=name)
    return out


def _split(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _from_env(name: str, raw: str) -> Any:
    """Parse one ``VALUA_*`` string into the field's target type."""
    key = name[len(ENV_PREFIX) :].lower()
    kind = _spec(key)[0]
    try:
        if kind is int:
            return int(raw)
        if kind is float:
            return float(raw)
        if kind is tuple:
            parts = _split(raw)
            if key == "seeds":
                return tuple(int(p) for p in parts)
            return tuple(parts)
        return raw
    except ValueError as exc:
        raise EnvParseError(
            "cannot parse environment value",
            field=key,
            raw=raw,
            expected=kind.__name__,
        ) from exc


def config_from_env(
    env: Mapping[str, str] | None = None,
    **overrides: Any,
) -> Config:
    """Build a :class:`Config` from defaults, ``VALUA_*`` vars and explicit overrides.

    An unrecognised ``VALUA_*`` variable is an error (E100) rather than a shrug:
    a typo like ``VALUA_BUDGETT`` would otherwise silently run the default budget
    and report it as if it had been chosen.
    """
    source = os.environ if env is None else env
    known = set(Config.FIELDS)
    values = dict(DEFAULTS)

    for name in sorted(source):
        if not name.startswith(ENV_PREFIX):
            continue
        key = name[len(ENV_PREFIX) :].lower()
        if key not in known:
            raise UnknownConfigKeyError(
                "unknown VALUA_* variable",
                variable=name,
                known=sorted(ENV_PREFIX + k for k in known),
            )
        values[key] = _from_env(name, source[name])

    unknown_overrides = sorted(set(overrides) - known)
    if unknown_overrides:
        raise UnknownConfigKeyError("unknown config key", keys=unknown_overrides)
    values.update(overrides)
    return Config(**values)
