"""Shared fixtures and deterministic RNG derivation for ValuaForge.

Author: 晨星

Every random draw in the test-suite goes through `derive_rng`. Python's builtin
`hash()` on str is salted by PYTHONHASHSEED and therefore NOT stable across
processes, so streams are keyed by CRC32 of the UTF-8 name instead.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest

SEED = 20261005


def derive_rng(seed: int, stream: str) -> np.random.Generator:
    """Derive an independent, reproducible Generator for a named stream."""
    return np.random.default_rng(
        np.random.SeedSequence([seed, zlib.crc32(stream.encode("utf-8"))])
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return derive_rng(SEED, "tests/default")


@pytest.fixture
def knn_instance():
    """A small reproducible KNN-utility instance: (y_train, dist, y_val, k)."""
    gen = derive_rng(SEED, "tests/knn_instance")
    n, k = 9, 3
    dist = gen.random(n) * 3.0
    y_train = gen.integers(0, 2, n)
    y_val = int(gen.integers(0, 2))
    return y_train, dist, y_val, k


@pytest.fixture
def tied_instance():
    """Instance containing an exact distance tie, to exercise the (dist, index) rule."""
    gen = derive_rng(SEED, "tests/tied_instance")
    n, k = 8, 3
    dist = gen.random(n) * 3.0
    dist[0] = dist[1]  # exact tie
    dist[3] = dist[4]  # a second exact tie
    y_train = gen.integers(0, 2, n)
    y_val = int(gen.integers(0, 2))
    return y_train, dist, y_val, k
