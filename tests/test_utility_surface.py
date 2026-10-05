"""The declared surface of the utility layer must actually work.

Author: 晨星

Regression suite for the blocker found by cross-validation: ``LogregUtility``
raised E303 on a single-class subset, and since the first arrival in *any*
permutation is a singleton -- necessarily single-class -- not one permutation
could complete. It survived 112 green tests because every one of them drove KNN.

The shared lesson is broader than that one bug: **a declared surface nobody
exercises will rot.** Three separate defects lived in that gap:

1. ``LogregUtility``/``SgdUtility`` raised or mis-valued single-class subsets.
2. ``core.config``'s default ``utility_model`` was ``logreg`` -- the broken one --
   while ``benchmark.py``/``run_demo.py`` hardcoded ``knn`` and bypassed config,
   so the demo passed and the default was never run.
3. ``_UTILITY_MODELS`` and ``make_utility`` disagreed in *both* directions:
   config accepted ``ridge`` (no factory branch -> E303) and rejected ``sgd``
   (which worked).

These tests walk the surface itself, not one lucky path through it.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import SUPPORTED_UTILITY_MODELS, config_from_env
from core.errors import UtilityUndefinedError
from core.types import Budget, Dataset
from valuation.utility import make_utility, supported_models

SEED = 7


def _dataset(n_tr: int = 12, n_va: int = 20, d: int = 4) -> Dataset:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_tr + n_va, d))
    y = (X[:, 0] + rng.normal(scale=0.3, size=n_tr + n_va) > 0).astype(np.int64)
    return Dataset(
        X=X,
        y=y,
        name="utility-surface",
        task="classification",
        train_idx=np.arange(n_tr),
        val_idx=np.arange(n_tr, n_tr + n_va),
    )


# ---------------------------------------------------------------------------
# surface agreement -- defect 3
# ---------------------------------------------------------------------------
def test_config_and_factory_declare_the_same_models() -> None:
    """A name accepted by config must be buildable, and vice versa."""
    assert set(SUPPORTED_UTILITY_MODELS) == set(supported_models())


@pytest.mark.parametrize("model", SUPPORTED_UTILITY_MODELS)
def test_every_declared_model_is_buildable(model: str) -> None:
    """Every name config accepts must construct without raising."""
    ds = _dataset()
    utility = make_utility(model, ds, Budget(1000), seed=SEED)
    assert utility.n_train == len(ds.train_idx)


# ---------------------------------------------------------------------------
# the blocker -- defects 1 and 2
# ---------------------------------------------------------------------------
#: Models whose value is defined on a single-class coalition. ``logreg`` is
#: deliberately absent: the team pinned E303 there as a contract (a one-class
#: logistic fit has no decision boundary), asserted by
#: tests/test_core_invariants.py::test_permutation_estimator_under_logreg_raises_e303_by_contract.
#: These tests cover the models that DO traverse singletons.
PERMUTATION_SAFE = tuple(m for m in SUPPORTED_UTILITY_MODELS if m != "logreg")


@pytest.mark.parametrize("model", PERMUTATION_SAFE)
def test_a_full_permutation_completes(model: str) -> None:
    """Walk every intermediate coalition of one permutation, singletons included.

    This is the exact walk a permutation Shapley estimator performs, and the
    first arrival is always a singleton.
    """
    ds = _dataset()
    n = len(ds.train_idx)
    utility = make_utility(model, ds, Budget(10_000), seed=SEED)

    float(utility.evaluate(()))  # v(empty) must be defined too
    arrived: list[int] = []
    for point in np.random.default_rng(1).permutation(n):
        arrived.append(int(point))
        value = float(utility.evaluate(tuple(sorted(arrived))))
        assert np.isfinite(value), f"{model}: non-finite utility mid-permutation"
        assert 0.0 <= value <= 1.0, f"{model}: utility {value} outside [0,1]"
        del value

    full = float(utility.evaluate(tuple(range(n))))
    assert 0.0 <= full <= 1.0


def test_the_configured_default_model_also_completes_a_permutation() -> None:
    """The configured default must be walkable, not just the demo's hardcoded one.

    ``benchmark.py``/``run_demo.py`` pin ``knn`` explicitly, so the configured
    default was never exercised. This closes that gap -- and, if the default is
    ever set to a model outside PERMUTATION_SAFE, it will say so loudly instead
    of the run dying one permutation in.
    """
    model = config_from_env().utility_model
    assert model in PERMUTATION_SAFE, (
        f"configured default utility_model={model!r} cannot traverse singletons; "
        "either change the default or resolve the E303 contract for it"
    )
    ds = _dataset()
    n = len(ds.train_idx)
    utility = make_utility(model, ds, Budget(10_000), seed=SEED)
    arrived: list[int] = []
    for point in range(n):
        arrived.append(int(point))
        assert np.isfinite(float(utility.evaluate(tuple(arrived))))


# ---------------------------------------------------------------------------
# degenerate-coalition semantics
# ---------------------------------------------------------------------------
def test_logreg_declines_single_class_coalitions_by_contract() -> None:
    """logreg must RAISE E303 on a singleton -- that is the pinned contract.

    Recorded here as well as in test_core_invariants so the surface test fails
    for the same reason a reader would expect: the behaviour is deliberate, not
    an oversight, and it is what currently blocks logreg as a permutation target.
    """
    ds = _dataset()
    utility = make_utility("logreg", ds, Budget(1000), seed=SEED)
    with pytest.raises(UtilityUndefinedError) as exc:
        utility.evaluate((0,))
    assert exc.value.code == "E303"


@pytest.mark.parametrize("model", PERMUTATION_SAFE)
def test_single_class_coalition_scores_that_class_share(model: str) -> None:
    """A one-class fit can only predict that class, so that is its utility.

    Returning 0.0 (the old SgdUtility behaviour) reported a coalition as
    worthless when its model would in fact score its own class; raising E303
    aborted a reachable state. Both were wrong, in opposite directions -- which
    is why each permutation-safe model is held to its own correct value here.
    """
    ds = _dataset()
    share = float(np.mean(ds.y[ds.val_idx] == ds.y[0]))
    utility = make_utility(model, ds, Budget(1000), seed=SEED)
    value = float(utility.evaluate((0,)))
    assert np.isfinite(value), f"{model}: single-row subset is not finite"
    if model == "knn":
        # 1/K zero padding: one neighbour out of K.
        assert value == pytest.approx(share / config_from_env().knn_k, abs=1e-12)
    else:
        assert value == pytest.approx(share, abs=1e-12)


def test_v_empty_is_zero_for_knn_and_the_prior_for_parametric_models() -> None:
    """v(empty) is model specific: KNN zero padding vs a model's own prior.

    ``logreg`` is excluded: under the pinned E303 contract it is never asked
    about an empty coalition, so a value there would be untested fiction.

    Algorithm spec 2.1 fixes v(empty) as the model prior accuracy, which is only
    meaningful for a model that has one. KNN's 1/K padding gives exactly 0.
    """
    ds = _dataset()
    val = ds.y[ds.val_idx]
    prior = float(np.bincount(val).max() / val.size)

    knn = make_utility("knn", ds, Budget(100), seed=SEED)
    assert float(knn.evaluate(())) == 0.0

    for model in ("sgd",):
        utility = make_utility(model, ds, Budget(100), seed=SEED)
        assert float(utility.evaluate(())) == pytest.approx(prior, abs=1e-12)


@pytest.mark.parametrize("model", SUPPORTED_UTILITY_MODELS)
def test_empty_coalition_still_costs_one_evaluation(model: str) -> None:
    """v(empty) must not be free, or an estimator understates its own budget."""
    ds = _dataset()
    utility = make_utility(model, ds, Budget(100), seed=SEED)
    before = utility.n_evals
    utility.evaluate(())
    assert utility.n_evals == before + 1


@pytest.mark.parametrize("model", PERMUTATION_SAFE)
def test_every_coalition_size_is_evaluable(model: str) -> None:
    """No coalition size may be unreachable for a permutation-safe utility."""
    ds = _dataset()
    n = len(ds.train_idx)
    utility = make_utility(model, ds, Budget(10_000), seed=SEED)
    for size in range(0, n + 1):
        subset = tuple(range(size))
        assert np.isfinite(float(utility.evaluate(subset))), f"{model}: size {size}"
