"""Synthetic data generators with a known quality structure.

Two DGPs tonight, chosen because they fail differently:

``gaussian_mixture``
    The easy, well-separated case. A valuation method that cannot beat a
    random vector here is broken, not unlucky.
``label_noise``
    The hard case: 15% of labels are flipped, and ``meta["corruption_mask"]``
    records exactly which rows. That mask is what makes "does the estimator
    rank noisy points lower?" a measurable question rather than a vibe.

Every generator draws from an **injected** ``numpy.random.Generator`` and never
touches global RNG state, so a dataset is a pure function of ``(name, n, seed)``.

Difficulty must have a gradient. A DGP where every method either maxes out or
collapses has no discriminating power, so ``class_sep`` and ``noise_frac`` are
exposed as knobs and the defaults sit away from both extremes.

Author: 晨星
"""

from __future__ import annotations

import zlib

import numpy as np

from core.errors import UnknownDatasetError
from core.types import Dataset

__all__ = ["DGP_REGISTRY", "generate", "make_split", "stable_name_hash"]


def make_split(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_train: int,
    rng: np.random.Generator,
    meta: dict | None = None,
) -> Dataset:
    """Build a Dataset with a disjoint validation split (E204 enforced downstream).

    ``n`` is the TOTAL row count: ``n_train`` training points plus the
    validation remainder. Asking for ``n_train >= n`` is a caller bug, not
    something to silently clamp -- clamping would quietly run the gold track
    at a smaller ``n`` than the one being reported, and the exact-Shapley cost
    is exponential in ``n``, so the difference is not a rounding detail.
    """
    n = len(y)
    if n_train >= n:
        raise ValueError(
            f"n_train ({n_train}) must be < n ({n}); n is the TOTAL row count "
            "(train + validation). Generate more rows instead of clamping."
        )
    if n_train < 2:
        raise ValueError(f"n_train must be >= 2, got {n_train}")
    perm = rng.permutation(n)
    tr = np.sort(perm[:n_train])
    va = np.sort(perm[n_train:])
    return Dataset(
        X=X,
        y=y,
        name=name,
        task="classification",
        train_idx=tr,
        val_idx=va,
        meta=dict(meta or {}),
    )


def _gaussian_mixture(
    n: int, d: int, rng: np.random.Generator, class_sep: float = 1.0, noise: float = 0.5
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Two well-separated Gaussian clusters.

    ``class_sep`` is the gap between cluster centres in units of the within
    cluster spread, so the Bayes error is monotone in it and the difficulty
    knob is continuous.
    """
    half = n // 2
    mu0 = np.zeros(d)
    mu1 = np.full(d, class_sep)
    X0 = rng.normal(mu0, noise, (half, d))
    X1 = rng.normal(mu1, noise, (n - half, d))
    X = np.vstack([X0, X1])
    y = np.array([0] * half + [1] * (n - half), dtype=np.int64)
    order = rng.permutation(n)
    X, y = X[order], y[order]
    return X, y, {"class_sep": float(class_sep), "noise": float(noise)}


def _label_noise(
    n: int, d: int, rng: np.random.Generator, noise_frac: float = 0.15, class_sep: float = 1.0
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Gaussian mixture with ``noise_frac`` of labels deliberately flipped.

    The flipped positions become ``corruption_mask`` in ``meta``: the ground
    truth for "a good valuation should score these lower". Without the mask a
    noise-detection claim would be unfalsifiable.
    """
    X, y, meta = _gaussian_mixture(n, d, rng, class_sep=class_sep)
    n_flip = round(noise_frac * n)
    flip_idx = (
        rng.choice(n, n_flip, replace=False) if n_flip > 0 else np.empty(0, dtype=np.int64)
    )
    mask = np.zeros(n, dtype=bool)
    mask[flip_idx] = True
    y_noisy = y.copy()
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
    meta = {**meta, "noise_frac": float(noise_frac), "corruption_mask": mask}
    return X, y_noisy, meta


DGP_REGISTRY = {
    "gaussian_mixture": _gaussian_mixture,
    "label_noise": _label_noise,
}


def stable_name_hash(name: str) -> int:
    """Stable integer for a DGP name.

    crc32, never ``hash()``: the builtin is salted per process by
    PYTHONHASHSEED, which would make dataset generation irreproducible across
    runs -- the same reason ``core.seed`` refuses to use it.
    """
    return int(zlib.crc32(name.encode("utf-8")))


def generate(
    name: str,
    n: int = 40,
    d: int = 8,
    seed: int = 7,
    n_train: int | None = None,
    **kwargs: object,
) -> Dataset:
    """Generate dataset ``name``. Raises E202 for an unknown name.

    E202 rather than a silent fallback: a typo turning into a valid-looking
    benchmark row is the exact failure the architecture forbids.
    """
    if name not in DGP_REGISTRY:
        raise UnknownDatasetError(
            "unknown dataset name", dataset=name, known=sorted(DGP_REGISTRY)
        )
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), stable_name_hash(name)]))
    X, y, meta = DGP_REGISTRY[name](n, d, rng, **kwargs)  # type: ignore[arg-type]
    train_n = n_train if n_train is not None else max(2, n // 2)
    return make_split(name, X, y, train_n, rng, meta)
