"""Method registry: name -> constructor, plus tier availability probes.

The registry exists so the CLI and the pipeline can look methods up by string
without importing ``valuation`` eagerly. Importing the whole layer just to answer
"which methods exist?" is exactly the heavy-import side effect the architecture
forbids in ``core``.

Availability probes use class-level lazy factories (pitfall library G): a probe
reading an instance attribute that only ``__init__`` sets would mark every
estimator unavailable.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.errors import DuplicateRegistrationError, MethodUnavailableError

__all__ = [
    "available_methods",
    "get_constructor",
    "is_available",
    "register",
    "registry_snapshot",
    "tier_report",
    "unregister",
]

Constructor = Callable[..., Any]

_REGISTRY: dict[str, Constructor] = {}


def register(name: str) -> Callable[[Constructor], Constructor]:
    """Class decorator registering a constructor under ``name``.

    Registration is idempotent for the same object so that a module re-import
    (notebooks, test collection) does not raise; a *different* object under an
    occupied name is a genuine bug and raises.
    """

    def _decorate(fn: Constructor) -> Constructor:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not fn:
            raise DuplicateRegistrationError(
                "duplicate method name in registry",
                name=name,
                existing=getattr(existing, "__qualname__", repr(existing)),
                incoming=getattr(fn, "__qualname__", repr(fn)),
            )
        _REGISTRY[name] = fn
        return fn

    return _decorate


def unregister(name: str) -> None:
    """Remove a registration. Used by tests to keep isolation."""
    _REGISTRY.pop(name, None)


def is_available(name: str) -> bool:
    """Whether ``name`` is registered *and* its constructor reports available.

    A registered-but-unavailable method still raises E300 on construction; this
    helper is for listing what a run can actually do right now.
    """
    fn = _REGISTRY.get(name)
    if fn is None:
        return False
    probe = getattr(fn, "available", None)
    if probe is None:
        return True
    try:
        return bool(probe())
    except Exception:
        return False


def available_methods() -> tuple[str, ...]:
    """Sorted names of usable methods. Sorted so output is diff-stable."""
    return tuple(sorted(name for name in _REGISTRY if is_available(name)))


def registry_snapshot() -> dict[str, str]:
    """name -> qualname, for diagnostics and benchmark metadata."""
    return {
        name: getattr(fn, "__qualname__", repr(fn)) for name, fn in sorted(_REGISTRY.items())
    }


def get_constructor(name: str) -> Constructor:
    """Look up a constructor by name.

    Raises:
        MethodUnavailableError: E300 if the name is unknown or reports unavailable.
    """
    fn = _REGISTRY.get(name)
    if fn is None:
        raise MethodUnavailableError(
            "unknown valuation method",
            name=name,
            known=sorted(_REGISTRY),
        )
    if not is_available(name):
        raise MethodUnavailableError("method is registered but unavailable", name=name)
    return fn


def tier_report() -> dict[str, bool]:
    """Probe which optional dependency tiers are importable.

    Uses ``find_spec`` rather than ``import`` so that probing does not pay the
    import cost or execute package side effects.
    """
    from importlib.util import find_spec

    report: dict[str, bool] = {}
    for label, module in (("numpy", "numpy"), ("scipy", "scipy"), ("sklearn", "sklearn")):
        try:
            report[label] = find_spec(module) is not None
        except (ImportError, ValueError):
            report[label] = False
    return report
