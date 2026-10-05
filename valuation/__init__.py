"""Valuation methods: what a data point is worth.

Each module here turns ``(data, utility, budget, rng)`` into a
:class:`~core.types.ValuationResult`. Two rules bind all of them:

* the ``rng`` is **injected** by the caller, never constructed internally, so
  two methods sharing a seed never share a noise stream;
* ``n_utility_evals`` is the oracle's real counter delta, measured as
  ``after - before``, never derived from a formula.

Author: 晨星
"""

from __future__ import annotations

__all__ = [
    "base",
    "beta",
    "brute",
    "loo",
    "rng_baseline",
    "tmc",
    "utility",
    "valuafuse",
]
