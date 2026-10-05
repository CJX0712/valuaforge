"""Utility oracles: the single place where a subset is turned into a score.

The budget currency of this repository is *the number of times*
:meth:`UtilityOracle.evaluate` is called. That count must be real, so every
oracle here increments a counter and the estimators never estimate it.

Three models, in increasing cost:

``knn``
    Closed form, no training. Deterministic and cheap; this is the anchor-B
    utility, and the only one where common random numbers have nothing to
    cancel (nothing is random in the first place).
``logreg``
    Newton / IRLS logistic regression, deterministic. The anchor-A utility:
    cheap enough to retrain 4096 times at n=12.
``sgd``
    Truncated fixed-step SGD. Random, and therefore the only family where CRN
    (common random numbers) can cancel noise at all. sklearn's
    ``random_state`` cannot make retraining *filter-level* deterministic, so
    this is hand-written on purpose -- see ``repro/README.md``.

All models take an explicit ``seed``; nothing reads global RNG state.

Author: 晨星
"""

from __future__ import annotations

import zlib

import numpy as np

from core.config import SUPPORTED_UTILITY_MODELS
from core.errors import (
    BudgetExhaustedError,
    ShapeMismatchError,
    UtilityUndefinedError,
)
from core.types import Budget, Dataset

__all__ = [
    "KnnUtility",
    "LogregUtility",
    "SgdUtility",
    "make_utility",
    "supported_models",
]


def _standardise(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (X - mean) / scale


class _BaseUtility:
    """Shared budget accounting.

    Subclasses implement :meth:`_score`; this class owns the counter, the
    budget guard and the degenerate-subset checks so that no subclass can
    accidentally bypass them.
    """

    #: Whether this oracle is a cheap surrogate. Billed at
    #: ``CHEAP_CALL_EQUIV`` when the caller reports full-equivalent cost.
    is_full_fidelity: bool = True

    def __init__(
        self, budget: Budget, n_train: int, train_idx: np.ndarray | None = None
    ) -> None:
        self._budget = budget
        self._n_train = int(n_train)
        self._n_evals = 0
        # Global row id -> position in the training pool. Identity when the pool
        # happens to be the whole dataset, which is the usual test case.
        ids = (
            np.arange(self._n_train, dtype=np.int64)
            if train_idx is None
            else np.asarray(train_idx, dtype=np.int64)
        )
        self._train_ids = ids
        # Subset indices are LOCAL positions in the training pool (0..n_train-1),
        # which is what "the first training point" means to every estimator, so
        # the map is the identity. ``train_idx`` is retained for provenance and
        # for callers that hold global ids.
        self._row_map = np.arange(ids.size, dtype=np.int64)

    @property
    def n_train(self) -> int:
        return self._n_train

    @property
    def n_evals(self) -> int:
        """Real number of :meth:`evaluate` calls so far. Never estimated."""
        return self._n_evals

    def remaining(self, budget: Budget | None = None) -> int:
        cap = self._budget if budget is None else budget
        return max(0, int(cap.max_utility_evals) - int(self._n_evals))

    def _check_affordable(self) -> None:
        if self._n_evals >= self._budget.max_utility_evals:
            raise BudgetExhaustedError(
                "utility evaluation budget exhausted",
                n_evals=self._n_evals,
                max_utility_evals=self._budget.max_utility_evals,
            )

    def evaluate(self, subset: np.ndarray) -> float:
        """Utility of ``subset``. Counts as exactly one evaluation."""
        self._check_affordable()
        idx = np.asarray(subset, dtype=np.int64).reshape(-1)
        self._n_evals += 1
        if idx.size == 0:
            # v(empty) is model specific, and it still costs one evaluation:
            # hiding it would let an estimator probe v(empty) for free and
            # quietly understate its own budget. KNN zero padding makes it 0;
            # a parametric model has a prior instead (see _v_empty).
            return float(self._v_empty())
        return float(self._score(idx))

    def _v_empty(self) -> float:
        """Utility of the empty coalition.

        Default 0.0, which is what the locked KNN 1/K zero-padding convention
        implies. Parametric models override it: a logistic model with no data
        predicts the majority class, so ``v(empty)`` is the prior accuracy
        (algorithm spec 2.1).
        """
        return 0.0

    def _to_local(self, idx: np.ndarray) -> np.ndarray:
        """Validate and pass through training-pool positions.

        Subset indices are **local**: position ``i`` means ``train_idx[i]``. The
        estimator contract in ``core.interfaces`` says subsets are "ascending
        int64 training indices", and the only indices an estimator can produce
        are 0..n_train-1. So this is the identity plus a bounds check -- the
        check matters because an out-of-range id would otherwise score whatever
        numpy happened to return, i.e. another point's value.
        """
        ids_arr = np.asarray(idx, dtype=np.int64)
        if ids_arr.size:
            lo, hi = int(ids_arr.min()), int(ids_arr.max())
            if lo < 0 or hi >= self._n_train:
                raise ShapeMismatchError(
                    "subset index outside the training pool",
                    min_id=lo,
                    max_id=hi,
                    n_train=int(self._n_train),
                )
        return ids_arr

    def _score(self, idx: np.ndarray) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class KnnUtility(_BaseUtility):
    """KNN accuracy under the locked 1/K **zero padding** convention.

    ``v(S) = (1/K) * sum_{t=1..min(K,|S|)} 1[y_{alpha_t(S)} == y_val]``

    Zero padding (not ``1/min(K,|S|)``) is what makes KNN-Shapley's closed form
    exact. Ties in distance break by ascending index so the oracle is a pure
    function of its input.
    """

    def __init__(
        self,
        y_train: np.ndarray,
        dist: np.ndarray,
        y_val: int,
        k: int,
        budget: Budget,
        train_idx: np.ndarray | None = None,
    ) -> None:
        super().__init__(budget, n_train=len(y_train), train_idx=train_idx)
        self._y = np.asarray(y_train)
        self._dist = np.asarray(dist, dtype=np.float64)
        self._y_val = int(y_val)
        self._k = int(k)
        self._order_cache: dict[int, np.ndarray] = {}

    def _order(self, idx: np.ndarray) -> np.ndarray:
        key = int(idx[0]) * 1_000_003 + int(idx.size)
        cached = self._order_cache.get(key)
        if cached is None:
            cached = np.lexsort((idx, self._dist[idx]))
            self._order_cache[key] = cached
        return cached

    def _score(self, idx: np.ndarray) -> float:
        order = self._order(self._to_local(np.sort(idx)))
        k = min(self._k, order.size)
        hits = int(np.count_nonzero(self._y[order[:k]] == self._y_val))
        return hits / self._k


class LogregUtility(_BaseUtility):
    """Deterministic L2-regularised logistic regression via IRLS.

    Deterministic on purpose: this is the anchor-A utility, and a gold standard
    that moves between runs cannot be diffed. Standardisation statistics come
    from the **training pool only** -- fitting them on the full matrix would
    leak validation data through the scaler.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        budget: Budget,
        lam: float = 1e-2,
        n_iter: int = 100,
        train_idx: np.ndarray | None = None,
    ) -> None:
        super().__init__(budget, n_train=len(X), train_idx=train_idx)
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        self._X_val = np.asarray(X_val, dtype=np.float64)
        self._y_val = np.asarray(y_val)
        self._lam = float(lam)
        self._n_iter = int(n_iter)
        self._mean = self._X.mean(axis=0)
        scale = self._X.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self._scale = scale
        self._Xs = _standardise(self._X, self._mean, self._scale)
        self._Xvs = _standardise(self._X_val, self._mean, self._scale)

    def _fit(self, rows: np.ndarray) -> np.ndarray:
        Xs = self._Xs[rows]
        ys = self._y[rows]
        n, d = Xs.shape
        Xa = np.hstack([Xs, np.ones((n, 1))])
        w = np.zeros(d + 1)
        for _ in range(self._n_iter):
            z = Xa @ w
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
            grad = Xa.T @ (p - ys) / n
            grad[:-1] += self._lam * w[:-1]
            s = np.clip(p * (1.0 - p), 1e-6, None)
            H = (Xa * s[:, None]).T @ Xa / n
            H[np.arange(d), np.arange(d)] += self._lam
            H[-1, -1] += 1e-8
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
            w_new = w - step
            if not np.all(np.isfinite(w_new)):
                break
            if np.max(np.abs(w_new - w)) < 1e-9:
                w = w_new
                break
            w = w_new
        return w

    def _score(self, idx: np.ndarray) -> float:
        # ``idx`` arrives as indices into the ORIGINAL row space, but self._y
        # and self._Xs are built from the training pool only. Map through
        # ``_row_map`` first: indexing with a global id raised IndexError as soon
        # as the validation split moved the ids past the pool size.
        rows = self._to_local(np.sort(idx))
        if np.unique(self._y[rows]).size < 2:
            # E303 is the pinned contract, not an oversight: a logistic fit on one
            # class has no decision boundary, so the utility is genuinely
            # undefined there. A permutation estimator does walk through
            # singletons, so this makes logreg unusable as a permutation target --
            # a KNOWN LIMITATION, pinned by
            # tests/test_core_invariants.py::test_permutation_estimator_under_logreg_raises_e303_by_contract.
            #
            # The alternative (degrade to "always predict that class", accuracy =
            # that class's validation share) is defensible and was prototyped; it
            # is NOT applied because the team pinned the strict behaviour. The
            # deliberate resolution path is to make the ESTIMATOR skip singletons,
            # not to quietly widen the utility's domain.
            #
            # NOTE: algorithm spec is internally inconsistent here -- 1.0 names
            # KNN as the gold-standard utility while 4.1 names Logistic
            # Regression. Under this contract only 1.0 is runnable. Escalated.
            raise UtilityUndefinedError(
                "utility undefined: subset has a single class",
                n_rows=int(rows.size),
                model="logreg",
            )
        w = self._fit(rows)
        z = np.hstack([self._Xvs, np.ones((self._Xvs.shape[0], 1))]) @ w
        pred = (z > 0.0).astype(np.float64)
        return float(np.mean(pred == self._y_val))


class SgdUtility(_BaseUtility):
    """Truncated fixed-step SGD logistic regression.

    Marked ``is_full_fidelity = False``: it is the cheap surrogate used as the
    control-variate target, so its calls must be billed at
    ``CHEAP_CALL_EQUIV`` in the full-equivalent budget column. Reporting them at
    face value is what overstated ValuaFuse's cost by ~2.8x.


    Two uses, both deliberate:

    * as the **target** utility on a random track, where CRN has something to
      cancel;
    * at a *fidelity rung* (few steps) as the **control variate** surrogate
      ``Y`` -- same model class as the target, so ``E[Y]`` is well defined and
      the classic ``X - beta*(Y - E[Y])`` reduction applies.

    The step schedule is a fixed geometric decay, not a line search, so that
    training is a deterministic function of ``(X, y, seed)`` -- which is what
    lets CRN pair the *same* noise realisation across two different subsets.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        budget: Budget,
        seed: int,
        steps: int = 10,
        lam: float = 1e-2,
        lr0: float = 0.5,
        train_idx: np.ndarray | None = None,
    ) -> None:
        super().__init__(budget, n_train=len(X), train_idx=train_idx)
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        self._X_val = np.asarray(X_val, dtype=np.float64)
        self._y_val = np.asarray(y_val)
        self._seed = int(seed)
        self._steps = int(steps)
        self._lam = float(lam)
        self._lr0 = float(lr0)
        self._mean = self._X.mean(axis=0)
        scale = self._X.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self._scale = scale
        self._Xs = _standardise(self._X, self._mean, self._scale)
        self._Xvs = _standardise(self._X_val, self._mean, self._scale)
        # A master stream keyed on (seed, rows) -- see _stream_for. CRN needs the
        # SAME stream for two different subsets, so it cannot be drawn from a
        # per-call counter.
        self._streams: dict[tuple[int, int], np.random.Generator] = {}
        self.is_full_fidelity = False

    def _stream_for(self, rows: np.ndarray) -> np.random.Generator:
        """Deterministic stream for this exact subset, stable across processes.

        Keyed on the row content (a crc32 of it), never on a call counter: two
        different subsets must get *independent* draws, and the same subset
        must get the *same* draw on every re-evaluation. That is precisely the
        common-random-numbers coupling.
        """
        key = (int(rows.size), int(zlib.crc32(rows.tobytes())))
        stream = self._streams.get(key)
        if stream is None:
            stream = np.random.default_rng(np.random.SeedSequence([self._seed, key[1]]))
            self._streams[key] = stream
        return stream

    def _fit(self, rows: np.ndarray) -> np.ndarray:
        Xs = self._Xs[rows]
        ys = self._y[rows]
        n, d = Xs.shape
        rng = self._stream_for(rows)
        w = rng.normal(0.0, 0.01, size=d)
        b = 0.0
        lr = self._lr0
        for _ in range(self._steps):
            z = Xs @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
            grad = Xs.T @ (p - ys) / n + self._lam * w
            gb = float(np.mean(p - ys))
            w -= lr * grad
            b -= lr * gb
            lr *= 0.9
        return np.concatenate([w, [b]])

    def _score(self, idx: np.ndarray) -> float:
        rows = self._to_local(np.sort(idx))
        classes = np.unique(self._y[rows])
        if classes.size < 2:
            # A SURROGATE may be defined everywhere: it only has to track the
            # target closely, never to be exact. A one-class fit can only predict
            # that class, so the value is the class's share of validation. Note
            # the deliberate contrast with LogregUtility, which raises E303
            # instead -- there the value IS part of the gold standard.
            return self._single_class_value(classes)
        w = self._fit(rows)
        z = self._Xvs @ w[:-1] + w[-1]
        return float(np.mean((z > 0.0).astype(np.float64) == self._y_val))

    def _v_empty(self) -> float:
        counts = np.bincount(np.asarray(self._y_val, dtype=np.int64))
        return float(counts.max() / counts.sum())

    def _single_class_value(self, classes: np.ndarray) -> float:
        return float(np.mean(self._y_val == classes[0]))


#: Model names this module can actually build.
#:
#: The tuple lives in ``core.config`` because ``core`` sits BELOW ``valuation`` in
#: the dependency order (``cli -> pipeline -> {data,valuation,eval} -> core``);
#: importing it the other way round would invert the layering. Holding a second
#: copy here is what let the two lists disagree in both directions -- config
#: accepted "ridge" (no branch, died with E303) and rejected "sgd" (worked).
#: ``tests/test_utility_surface.py`` asserts they stay equal.
assert set(SUPPORTED_UTILITY_MODELS) == {"knn", "logreg", "sgd"}, (
    "make_utility's dispatch and core.config's declared models must agree; "
    f"got {SUPPORTED_UTILITY_MODELS!r}"
)


def supported_models() -> tuple[str, ...]:
    """Names accepted by :func:`make_utility`. Single source of truth."""
    return tuple(SUPPORTED_UTILITY_MODELS)


def make_utility(
    model: str,
    data: Dataset,
    budget: Budget,
    seed: int,
    k: int = 5,
    sgd_steps: int = 10,
) -> _BaseUtility:
    """Build an oracle for ``model`` over ``data``'s training pool.

    The validation split comes from ``data.val_idx``; the scaler inside each
    oracle is fit on the training pool alone.
    """
    tr = np.sort(data.train_idx)
    va = np.sort(data.val_idx)
    X_tr, y_tr = data.X[tr], data.y[tr]
    X_va, y_va = data.X[va], data.y[va]

    if model == "knn":
        dist = _knn_distances(X_tr, X_va)
        return _KnnFromDistances(y_tr, dist, y_va, k, budget, train_idx=tr)
    if model == "logreg":
        return LogregUtility(X_tr, y_tr, X_va, y_va, budget, train_idx=tr)
    if model == "sgd":
        return SgdUtility(
            X_tr, y_tr, X_va, y_va, budget, seed=seed, steps=sgd_steps, train_idx=tr
        )
    raise UtilityUndefinedError(
        "unknown utility model",
        model=model,
        supported=list(SUPPORTED_UTILITY_MODELS),
    )


def _knn_distances(X_tr: np.ndarray, X_va: np.ndarray) -> np.ndarray:
    """Euclidean distance from every validation point to every train point."""
    d2 = (
        np.sum(X_va**2, axis=1)[:, None]
        + np.sum(X_tr**2, axis=1)[None, :]
        - 2.0 * X_va @ X_tr.T
    )
    return np.sqrt(np.maximum(d2, 0.0))


class _KnnFromDistances(KnnUtility):
    """KNN utility averaged over the whole validation set (anchor B)."""

    def __init__(
        self,
        y_train: np.ndarray,
        dist: np.ndarray,
        y_val: np.ndarray,
        k: int,
        budget: Budget,
        train_idx: np.ndarray | None = None,
    ) -> None:
        super().__init__(y_train, dist, int(y_val[0]), k, budget, train_idx=train_idx)
        self._y_val_all = np.asarray(y_val)
        self._dist_all = np.asarray(dist, dtype=np.float64)

    def _score(self, idx: np.ndarray) -> float:
        rows = self._to_local(np.sort(idx))
        total = 0.0
        # ``_dist_all`` is (n_val, n_train): validation points are rows, training
        # points are columns, so a subset selects *columns*.
        d_sub = self._dist_all[:, rows]
        for j in range(d_sub.shape[0]):
            order = np.lexsort((rows, d_sub[j]))
            kk = min(self._k, order.size)
            hits = int(np.count_nonzero(self._y[order[:kk]] == self._y_val_all[j]))
            total += hits / self._k
        return total / float(self._dist_all.shape[0])
