"""SeedBank: the single entry point for all randomness in ValuaForge.

Determinism is this repository's hard gate (architecture spec section 9, gate G5),
so the derivation rule is fixed here and nowhere else:

    rng = np.random.default_rng(np.random.SeedSequence([seed, crc32(stream_name)]))

Why ``zlib.crc32`` and **not** ``hash(str)``: CPython randomises string hashing per
process via ``PYTHONHASHSEED``. A ``hash()``-derived stream would produce different
numbers in the test process than in the benchmark process, and the failure would look
like a numerical bug rather than a seeding bug. Phase 1 measured this directly.

Stream names are namespaced by purpose (``data/...``, ``val/...``, ``eval/...``) so that
two consumers can never accidentally share a stream and correlate their noise.

Author: 晨星
"""

from __future__ import annotations

import zlib

import numpy as np

from core.errors import ConfigSchemaError

__all__ = [
    "SeedBank",
    "data_stream",
    "eval_stream",
    "stream_entropy",
    "val_stream",
]

_MAX_SEED = 2**32 - 1


def stream_entropy(stream_name: str) -> int:
    """Map a stream name to a stable 32-bit entropy value.

    ``zlib.crc32`` is specified by its algorithm, not by the interpreter, so the
    value is identical across processes, platforms and Python versions.
    """
    if not isinstance(stream_name, str):
        raise ConfigSchemaError("stream name must be str", got=type(stream_name).__name__)
    if not stream_name:
        raise ConfigSchemaError("stream name must be non-empty")
    return int(zlib.crc32(stream_name.encode("utf-8")))


class SeedBank:
    """Derives independent, reproducible generators from one root seed.

    The bank never hands out stateful shared generators: each :meth:`stream` call
    builds a fresh ``Generator`` from a ``SeedSequence`` keyed by the root seed and
    the stream name, so drawing from one stream cannot shift another.
    """

    __slots__ = ("_seed",)

    def __init__(self, seed: int) -> None:
        self._seed = self._validate(seed)

    @staticmethod
    def _validate(seed: int) -> int:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ConfigSchemaError("seed must be an integer", got=type(seed).__name__)
        value = int(seed)
        if not 0 <= value <= _MAX_SEED:
            raise ConfigSchemaError("seed out of range", seed=value, max_seed=_MAX_SEED)
        return value

    @property
    def seed(self) -> int:
        """The root seed this bank was built from."""
        return self._seed

    def set_all(self, seed: int) -> SeedBank:
        """Re-root the bank in place and return it, for fluent re-seeding.

        Mutating rather than replacing keeps every previously derived stream name
        valid; callers that hold a bank across a re-seed simply get new streams.
        """
        self._seed = self._validate(seed)
        return self

    def entropy_for(self, stream_name: str) -> list[int]:
        """The exact ``SeedSequence`` entropy for a stream. Exposed for tests."""
        return [self._seed, stream_entropy(stream_name)]

    def stream(self, stream_name: str) -> np.random.Generator:
        """Return a fresh generator dedicated to ``stream_name``.

        Two calls with the same name produce bitwise-identical output; two calls
        with different names do not.
        """
        seq = np.random.SeedSequence(self.entropy_for(stream_name))
        return np.random.default_rng(seq)

    def spawn(self, *stream_names: str) -> tuple[np.random.Generator, ...]:
        """Derive several named streams at once, in the order given."""
        return tuple(self.stream(name) for name in stream_names)

    def fingerprint(self, stream_name: str, n: int = 4) -> str:
        """A short hex digest of the stream's first ``n`` standard normals.

        Lets the determinism gate compare two runs (or two processes) with a
        string instead of shipping whole arrays around.
        """
        draws = self.stream(stream_name).standard_normal(int(n))
        return ",".join(f"{float(v):.17g}" for v in draws)

    def __repr__(self) -> str:
        return f"SeedBank(seed={self._seed})"


def data_stream(bank: SeedBank, dataset: str, seed: int) -> np.random.Generator:
    """Stream for dataset synthesis: ``data/<dataset>/<seed>``."""
    return bank.stream(f"data/{dataset}/{seed}")


def val_stream(bank: SeedBank, method: str, dataset: str, seed: int) -> np.random.Generator:
    """Stream for a valuation method: ``val/<method>/<dataset>/<seed>``."""
    return bank.stream(f"val/{method}/{dataset}/{seed}")


def eval_stream(bank: SeedBank, evaluator: str, dataset: str, seed: int) -> np.random.Generator:
    """Stream for an evaluator: ``eval/<evaluator>/<dataset>/<seed>``."""
    return bank.stream(f"eval/{evaluator}/{dataset}/{seed}")
