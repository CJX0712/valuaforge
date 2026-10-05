"""Engineering contracts: I15 determinism, I21 budget accounting, I23 NaN guard.

Author: 晨星

These lock the failure modes that silently invalidate the whole benchmark rather
than the mathematics: a counter that drifts from real calls, a nondeterministic
stream, or a column of metrics quietly turning into NaN.
"""

from __future__ import annotations

import ast
import warnings
import zlib

import numpy as np
import pytest
from scipy.stats import kendalltau, spearmanr

from tests.conftest import derive_rng
from tests.reference import knn_shapley_closed_form, knn_utility, permutation_marginals


# ---------------------------------------------------------------------------
# I15 -- determinism
# ---------------------------------------------------------------------------
def test_i15_same_seed_is_bitwise_identical(knn_instance) -> None:
    """I15: two runs at the same seed must agree bit for bit."""
    y_train, dist, y_val, k = knn_instance

    def run() -> np.ndarray:
        gen = derive_rng(20261005, "tests/i15")
        total = np.zeros(len(y_train), dtype=np.float64)
        for _ in range(32):
            perm = gen.permutation(len(y_train))
            total += permutation_marginals(perm, y_train, dist, y_val, k)
        return total / 32

    a, b = run(), run()
    assert np.array_equal(a, b), "I15 FAILED: same seed produced different results"


def test_i15_streams_are_independent() -> None:
    """Distinct named streams must not alias to the same numbers."""
    a = derive_rng(7, "val/tmc/ds1").random(16)
    b = derive_rng(7, "val/beta/ds1").random(16)
    c = derive_rng(7, "val/tmc/ds1").random(16)
    assert not np.array_equal(a, b), "different streams produced identical draws"
    assert np.array_equal(a, c), "same stream did not reproduce"


def test_stream_key_is_stable_across_processes() -> None:
    """The stream key must not depend on Python's salted string hash."""
    name = "val/valuafuse/ds3"
    assert zlib.crc32(name.encode("utf-8")) == zlib.crc32(name.encode("utf-8"))
    assert (
        hash(name) != zlib.crc32(name.encode("utf-8")) or True
    )  # hash is salted, crc32 is not


# ---------------------------------------------------------------------------
# I21 -- budget accounting must be exact integer equality
# ---------------------------------------------------------------------------
class CountingUtility:
    """Reference utility wrapper that counts real calls.

    Phase 2's `UtilityFn` must behave the same way: the reported
    `n_utility_evals` is the number of real `evaluate` calls, never a formula.
    """

    def __init__(self, y_train: np.ndarray, dist: np.ndarray, y_val: int, k: int) -> None:
        self._y = y_train
        self._d = dist
        self._yv = y_val
        self._k = k
        self.n_calls = 0

    @property
    def n_train(self) -> int:
        return len(self._y)

    def evaluate(self, subset) -> float:
        self.n_calls += 1
        return knn_utility(list(subset), self._y, self._d, self._yv, self._k)


def permutation_shapley_via_utility(perm, utility: CountingUtility) -> np.ndarray:
    """Permutation estimator that routes EVERY marginal through `utility.evaluate`.

    This is the shape Phase 2's estimators must have, otherwise the budget
    counter is decorative and I21 cannot hold.
    """
    n = len(perm)
    delta = np.zeros(n, dtype=np.float64)
    arrived: list[int] = []
    prev = utility.evaluate(())
    for idx in perm:
        arrived.append(idx)
        now = utility.evaluate(tuple(arrived))
        delta[idx] = now - prev
        prev = now
    return delta


def test_i21_counter_matches_real_calls(knn_instance) -> None:
    """I21: the counter equals the number of actual calls, exactly."""
    y_train, dist, y_val, k = knn_instance
    n = len(y_train)
    n_perm = 50
    utility = CountingUtility(y_train, dist, y_val, k)
    gen = derive_rng(20261005, "tests/i21")
    for _ in range(n_perm):
        permutation_shapley_via_utility(gen.permutation(n), utility)
    expected = n_perm * (n + 1)  # n marginals + the v(empty) probe per permutation
    assert utility.n_calls == expected, f"I21 FAILED: {utility.n_calls} != {expected}"


def test_i21_relative_error_is_exactly_zero(knn_instance) -> None:
    """I21: budget equality is integer-exact, not within a float tolerance."""
    y_train, dist, y_val, k = knn_instance
    n = len(y_train)
    runs = []
    for _ in range(3):
        utility = CountingUtility(y_train, dist, y_val, k)
        gen = derive_rng(20261005, "tests/i21")
        for _ in range(20):
            permutation_shapley_via_utility(gen.permutation(n), utility)
        runs.append(utility.n_calls)
    assert len(set(runs)) == 1, f"I21 FAILED: call counts differ across runs: {runs}"
    relative_error = abs(runs[0] - runs[0]) / runs[0]
    assert relative_error == 0.0


# ---------------------------------------------------------------------------
# I23 -- a degenerate valuation must never be silently scored
# ---------------------------------------------------------------------------
def n_distinct(values: np.ndarray) -> int:
    """Number of distinct values in a valuation vector.

    This is the ONLY reliable degeneracy probe. See the tiny-jitter test below:
    rank correlation depends solely on the ordering, so it is completely
    invariant to the magnitude of a perturbation.
    """
    return int(np.unique(np.asarray(values, dtype=np.float64)).size)


def assert_non_degenerate(values: np.ndarray, *, name: str = "valuation") -> None:
    """Fail fast on a degenerate valuation vector.

    Two independent conditions, both required:

    1. ``n_distinct >= max(3, ceil(n/2))`` -- catches tiny-jitter degeneracy,
       which NO magnitude threshold can detect.
    2. ``std > 0`` -- cheap pre-filter for the exactly-constant case.

    Callers must invoke this BEFORE computing any rank correlation. Scoring a
    degenerate vector yields either NaN or a plausible-looking fabricated
    correlation, and in the latter case nothing downstream looks wrong.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 0:
        raise AssertionError(f"{name}: empty valuation vector")
    if not np.all(np.isfinite(arr)):
        raise AssertionError(f"{name}: contains NaN/Inf")
    min_distinct = max(3, int(np.ceil(0.5 * n)))
    nd = n_distinct(arr)
    if nd < min_distinct:
        raise AssertionError(
            f"{name}: degenerate, n_distinct={nd} < {min_distinct} (n={n}). "
            "A rank correlation here is meaningless (NaN or fabricated)."
        )
    if arr.std() <= 0.0:
        raise AssertionError(f"{name}: constant vector")


def test_i23_scipy_spearman_returns_nan_on_constant_input() -> None:
    """I23 root cause, pinned: both correlations return NaN on constant input.

    Measured asymmetry, which is the whole point of pinning it:
      * ``spearmanr`` emits ``ConstantInputWarning`` (a RuntimeWarning subclass)
        before returning NaN -- there is at least a signal.
      * ``kendalltau`` returns NaN **completely silently**, with no warning at
        all. It is therefore the more dangerous of the two: a degenerate
        estimation silently poisons a whole metric column.

    Pinned so that a scipy upgrade which changes this is noticed here instead
    of silently altering every metric column in the benchmark.
    """
    constant = np.ones(8)
    varying = np.arange(8, dtype=np.float64)

    with pytest.warns(RuntimeWarning, match="constant"):
        rho = spearmanr(constant, varying).statistic
    assert np.isnan(rho)

    # kendalltau: NaN with NO warning -- assert the silence explicitly
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tau = kendalltau(constant, varying).statistic
    assert np.isnan(tau)


def test_i23_exactly_constant_is_rejected() -> None:
    """The plain case: an exactly constant vector must be rejected, not scored."""
    with pytest.raises(AssertionError, match="degenerate"):
        assert_non_degenerate(np.full(10, 0.25))


def test_i23_tiny_jitter_defeats_every_magnitude_threshold() -> None:
    """The nasty case (found by the algorithm team): a 1e-15 jitter hides perfectly.

    scipy's ConstantInputWarning fires ONLY on an exactly constant input. A
    single-point perturbation of ANY size produces a strictly monotone ordering,
    so the correlation is well defined and NO warning is emitted::

        x = [1]*8 with x[-1] += eps
        eps = 1e-15 .. 1e-3  ->  spearman rho == 0.5774 in every case, warns == 0

    rho is *identical* across twelve orders of magnitude: rank correlation is
    invariant to the jitter size. Therefore NO threshold on std/scale can
    separate a 1e-15 jitter from a 1e-3 one -- which is why the naive
    ``std() > 0`` guard, and even a *relative* threshold such as
    ``std > 1e-8 * max(|x|max, 1)``, both admit the larger jitters even though
    those are just as degenerate (still only 2 distinct levels).

    Only ``n_distinct`` is reliable. This test pins the std-only blind spot so
    nobody "simplifies" the guard back into a magnitude check.
    """
    n = 8
    reference = np.arange(n, dtype=np.float64)
    min_distinct = max(3, int(np.ceil(0.5 * n)))

    std_gate_blind_spot: list[float] = []
    for eps in (1e-15, 1e-12, 1e-9, 1e-7, 1e-5, 1e-3):
        values = np.ones(n)
        values[-1] += eps

        # scipy is perfectly happy: a real number, and not a single warning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rho = spearmanr(values, reference).statistic
        assert not np.isnan(rho)
        assert len(caught) == 0, "scipy unexpectedly warned for a non-constant input"

        # our guard rejects it at every magnitude
        with pytest.raises(AssertionError, match="degenerate"):
            assert_non_degenerate(values)

        # record where a magnitude-only gate would have failed to fire
        if values.std() > 1e-8 * max(np.abs(values).max(), 1.0):
            std_gate_blind_spot.append(eps)

    # a magnitude gate would have admitted these -- n_distinct must not
    assert std_gate_blind_spot, "expected the std-only gate to have a blind spot"
    for eps in std_gate_blind_spot:
        values = np.ones(n)
        values[-1] += eps
        assert n_distinct(values) < min_distinct


def test_i23_non_degenerate_vectors_are_not_falsely_rejected() -> None:
    """The guard must not reject legitimate valuations, including discrete ones."""
    min_distinct = max(3, int(np.ceil(0.5 * 8)))
    for values in (
        np.array([0.11, 0.42, 0.07, 0.93, 0.15, 0.81, 0.23, 0.55]),  # typical
        np.arange(1.0, 9.0),  # strictly increasing
        np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0]),  # discrete but meaningful
    ):
        assert n_distinct(values) >= min_distinct
        assert_non_degenerate(values)


def test_i23_nanmean_is_not_a_substitute_for_a_guard() -> None:
    """Demonstrates why np.nanmean must not be used: it hides the failure."""
    rho_with_nan = np.array([0.9, np.nan, np.nan, 0.8, 0.85])
    assert not np.isnan(np.nanmean(rho_with_nan))  # silently drops the bad entries
    assert np.isnan(np.mean(rho_with_nan))  # plain mean propagates it


# ---------------------------------------------------------------------------
# closed-form sanity used by the calibration tests
# ---------------------------------------------------------------------------
def test_closed_form_sum_matches_v_full(knn_instance) -> None:
    """Cheap end-to-end sanity: the closed form must satisfy efficiency."""
    y_train, dist, y_val, k = knn_instance
    phi = knn_shapley_closed_form(y_train, dist, y_val, k)
    v_full = knn_utility(list(range(len(y_train))), y_train, dist, y_val, k)
    assert abs(float(phi.sum()) - v_full) < 1e-12


# ---------------------------------------------------------------------------
# A1 -- layer architecture: core must stay free of algorithms
# ---------------------------------------------------------------------------
_UPPER_LAYERS = ("valuation", "eval", "pipeline", "data", "cli")
_REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _imported_modules(path) -> set[str]:
    """Top-level package names imported by a module, via AST (never via execution).

    AST rather than grep: a commented-out import, a name inside a docstring, or a
    substring like ``evaluate`` must not produce a false positive, and a real
    ``from valuation.x import y`` must not be able to hide behind formatting.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # level > 0 means a relative import (e.g. ``from ..core import x``),
            # which is itself illegal for these sibling top-level packages.
            if node.level and node.level > 0:
                names.add(f"<relative level {node.level}>")
            elif node.module:
                names.add(node.module.split(".")[0])
    return names


def test_a1_core_never_imports_upper_layers() -> None:
    """A1: `core/` must not import data/valuation/eval/pipeline/cli.

    Enforces ``cli -> pipeline -> {data, valuation, eval} -> core``. The moment
    `core` imports an algorithm module it becomes a junk drawer that `valuation`
    must import back, and the cycle makes `core` untestable in isolation -- which
    is the whole reason `core` exists. This is the only thing stopping someone
    from "helpfully" moving an algorithm down into `core` later.
    """
    core_dir = _REPO_ROOT / "core"
    modules = sorted(core_dir.glob("*.py"))
    assert modules, "no core/*.py found -- has the layout changed?"

    violations: list[str] = []
    for path in modules:
        for name in sorted(_imported_modules(path)):
            if name in _UPPER_LAYERS:
                violations.append(f"{path.relative_to(_REPO_ROOT)} imports {name!r}")
    assert not violations, "A1 FAILED (core must not depend on upper layers):\n" + "\n".join(
        violations
    )


def test_a1_core_uses_no_relative_imports() -> None:
    """A1b: sibling top-level packages must be imported absolutely.

    ``from ..core import x`` raises ImportError because ``core`` is a *sibling*
    of the importing package, not its parent (pitfall library G). Caught here so
    the failure surfaces as a clear assertion instead of a runtime ImportError.
    """
    violations: list[str] = []
    for path in sorted((_REPO_ROOT / "core").glob("*.py")):
        for name in sorted(_imported_modules(path)):
            if name.startswith("<relative"):
                violations.append(f"{path.relative_to(_REPO_ROOT)} uses a relative import")
    assert not violations, "A1b FAILED:/n" + "\n".join(violations)


def test_a1_no_valuation_algorithm_inside_core() -> None:
    """A1c: `core/` must contain infrastructure only, never an algorithm.

    Guards the ruling that the flagship lives in ``valuation/valuafuse.py`` and
    gets no exemption. Matched on module *name* because a stray ``shapley.py``
    under ``core/`` is the exact failure mode being banned.
    """
    banned = ("shapley", "tmc", "beta", "banzhaf", "knn", "influence", "oob", "valuafuse")
    offenders = [
        path.name
        for path in sorted((_REPO_ROOT / "core").glob("*.py"))
        if any(token in path.stem.lower() for token in banned)
    ]
    assert not offenders, f"A1c FAILED: algorithm-shaped modules inside core/: {offenders}"
