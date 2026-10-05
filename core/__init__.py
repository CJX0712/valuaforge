"""ValuaForge core layer.

Lowest layer in the dependency graph: nothing here imports ``data``,
``valuation``, ``eval`` or ``pipeline``. Modules use flat absolute imports
(``core.seed``), because these packages are siblings at the top level, not
nested children -- ``from ..core import x`` raises ImportError here.

Author: 晨星
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "config",
    "errors",
    "interfaces",
    "registry",
    "seed",
    "types",
]
