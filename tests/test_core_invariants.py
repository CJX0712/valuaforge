"""Invariants for the ``core`` layer.

Scope: the contracts that the rest of the system relies on being true no matter
what the algorithm layer does -- deterministic seeding, registry round-trips,
config schema enforcement, and the error-code catalogue.

Author: 晨星
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import zlib
from math import comb

import numpy as np
import pytest

from core.config import ENV_PREFIX, Config, config_from_env
from core.errors import (
    ERROR_CATALOG,
    ConfigSchemaError,
    DeterminismError,
    DuplicateRegistrationError,
    EnvParseError,
    GoldTooLargeError,
    LeakageError,
    MethodUnavailableError,
    ShapeMismatchError,
    UnknownConfigKeyError,
    UtilityUndefinedError,
    ValuaError,
    ensure_deterministic,
    ensure_gold_size,
    ensure_no_leakage,
    error_class,
    has_error_code,
)
from core.registry import (
    available_methods,
    get_constructor,
    is_available,
    register,
    registry_snapshot,
    tier_report,
    unregister,
)
from core.seed import SeedBank, stream_entropy
from core.types import Budget, Dataset, UtilitySpec, ValuationResult
from data.dgp import generate
from eval.report import dod_verdict
from valuation.beta import beta_shapley_exact, beta_weights, classical_weights
from valuation.brute import exact_shapley
from valuation.loo import LeaveOneOut
from valuation.tmc import TMCConfig, tmc_once
from valuation.utility import make_utility

# ---------------------------------------------------------------------------
# SeedBank determinism -- three separate guarantees
# ---------------------------------------------------------------------------


def test_same_stream_twice_is_bitwise_identical() -> None:
    """Deriving the same stream name twice must give bit-identical draws.

    This is the property everything else rests on: if one call perturbed global
    state, re-running a benchmark would not reproduce it.
    """
    bank = SeedBank(7)
    first = bank.stream("val/tmc/digits/7").standard_normal(256)
    second = bank.stream("val/tmc/digits/7").standard_normal(256)
    assert first.tobytes() == second.tobytes()
    assert bank.fingerprint("val/tmc/digits/7") == bank.fingerprint("val/tmc/digits/7")


def test_different_streams_are_independent() -> None:
    """Distinct names must yield distinct streams; identical ones must not collide."""
    bank = SeedBank(7)
    names = [
        "data/digits/7",
        "val/tmc/digits/7",
        "val/beta/digits/7",
        "eval/metrics/digits/7",
    ]
    draws = {name: bank.stream(name).standard_normal(64) for name in names}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            assert draws[left].tobytes() != draws[right].tobytes(), f"{left} == {right}"
    # A stream must not be a shifted copy of another: correlation must be ~0.
    corr = float(np.corrcoef(draws[names[0]], draws[names[1]])[0, 1])
    assert abs(corr) < 0.4


def test_stream_is_stable_across_processes() -> None:
    """Stream derivation must survive a fresh interpreter.

    This is the regression guard for PYTHONHASHSEED: with ``hash(str)`` this
    assertion fails intermittently depending on the interpreter's hash seed,
    which is precisely the class of bug that makes a "deterministic" benchmark
    irreproducible on someone else's machine.
    """
    program = (
        "import numpy as np, zlib\n"
        "from core.seed import SeedBank\n"
        "bank = SeedBank(7)\n"
        "g = bank.stream('val/tmc/digits/7')\n"
        "print(','.join(f'{v:.17g}' for v in g.standard_normal(8)))\n"
    )
    runs = []
    for hash_seed in ("0", "1", "12345"):
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            # Inherit the full parent environment so the child interpreter can
            # launch on every OS (an empty PATH broke this on Linux runners). Only
            # PYTHONPATH (repo root, for `core`) and PYTHONHASHSEED (the variable
            # under test) are overridden; the seed override wins because it is set
            # after the spread.
            env={
                **os.environ,
                "PYTHONPATH": str(pathlib.Path(__file__).resolve().parents[1]),
                "PYTHONHASHSEED": hash_seed,
            },
        )
        runs.append(proc.stdout.strip())

    assert len(set(runs)) == 1, f"stream derivation varies by PYTHONHASHSEED: {runs}"
    expected = ",".join(
        f"{v:.17g}" for v in SeedBank(7).stream("val/tmc/digits/7").standard_normal(8)
    )
    assert runs[0] == expected


def test_stream_entropy_is_crc32_not_python_hash() -> None:
    """The derivation must key on crc32, which is specified, not on salted hash()."""
    name = "val/valuafuse/moons/2027"
    assert stream_entropy(name) == zlib.crc32(name.encode("utf-8"))
    assert SeedBank(7).entropy_for(name) == [7, zlib.crc32(name.encode("utf-8"))]


def test_seed_bank_rejects_invalid_seeds() -> None:
    """Out-of-range or non-integer seeds must fail loudly, not wrap around."""
    for bad in (-1, 2**32, "7", 7.5, True):
        with pytest.raises(ConfigSchemaError) as exc:
            SeedBank(bad)  # type: ignore[arg-type]
        assert exc.value.code == "E101"


def test_set_all_reroots_in_place() -> None:
    """``set_all`` must re-root the same bank and keep stream names valid."""
    bank = SeedBank(7)
    before = bank.fingerprint("val/tmc/digits/7")
    bank.set_all(17)
    assert bank.seed == 17
    assert bank.fingerprint("val/tmc/digits/7") != before
    assert bank.fingerprint("val/tmc/digits/7") == SeedBank(17).fingerprint("val/tmc/digits/7")


def test_empty_stream_name_is_rejected() -> None:
    """An empty stream name would silently collide across call sites."""
    with pytest.raises(ConfigSchemaError):
        SeedBank(7).stream("")


# ---------------------------------------------------------------------------
# Registry round-trip
# ---------------------------------------------------------------------------


def test_registry_round_trip() -> None:
    """register -> lookup -> construct must return the same callable."""

    @register("dummy_method")
    class Dummy:
        name = "dummy_method"

        @classmethod
        def available(cls) -> bool:
            return True

    try:
        assert is_available("dummy_method")
        assert get_constructor("dummy_method") is Dummy
        assert "dummy_method" in available_methods()
        assert registry_snapshot()["dummy_method"].endswith("Dummy")
        assert Dummy().name == "dummy_method"
    finally:
        unregister("dummy_method")
    assert not is_available("dummy_method")
    assert "dummy_method" not in available_methods()


def test_registry_reports_unavailable_method() -> None:
    """A method whose ``available()`` is False must be treated as unusable (E300)."""

    @register("unavailable_method")
    class Unavailable:
        name = "unavailable_method"

        @classmethod
        def available(cls) -> bool:
            return False

    try:
        assert not is_available("unavailable_method")
        assert "unavailable_method" not in available_methods()
        with pytest.raises(ValuaError) as exc:
            get_constructor("unavailable_method")
        assert exc.value.code == "E300"
    finally:
        unregister("unavailable_method")


def test_registry_rejects_unknown_name() -> None:
    """Unknown names raise E300 and list what is actually registered."""
    with pytest.raises(ValuaError) as exc:
        get_constructor("no_such_method")
    assert exc.value.code == "E300"


def test_registry_rejects_duplicate_name() -> None:
    """Re-registering a *different* object under one name is a bug, not a no-op.

    The guard lives in ``register``, so ``pytest.raises`` must wrap the
    **decoration itself**. Wrapping only the later lookup can never observe the
    error, because the registry is already poisoned by then.
    """

    @register("dup_method")
    class First:
        @classmethod
        def available(cls) -> bool:
            return True

    class Second:
        @classmethod
        def available(cls) -> bool:
            return True

    try:
        with pytest.raises(ValuaError) as exc:
            register("dup_method")(Second)
        assert exc.value.code == "E103"

        # Re-registering the *same* object stays idempotent (module re-import).
        register("dup_method")(First)
        assert get_constructor("dup_method") is First
    finally:
        unregister("dup_method")
    assert First.__name__ == "First" and Second.__name__ == "Second"


def test_available_methods_is_sorted() -> None:
    """Listing order must be stable so CLI tables and JSON diff cleanly."""

    def _available(cls) -> bool:
        return True

    for name in ("zeta_c", "alpha_c"):
        register(name)(type(name, (), {"available": classmethod(_available)}))
    try:
        listed = available_methods()
        assert listed == tuple(sorted(listed))
        assert {"alpha_c", "zeta_c"} <= set(listed)
    finally:
        unregister("zeta_c")
        unregister("alpha_c")


def test_tier_report_detects_installed_tiers() -> None:
    """Tier probes must answer without importing the packages."""
    report = tier_report()
    assert set(report) == {"numpy", "scipy", "sklearn"}
    assert all(isinstance(v, bool) for v in report.values())
    assert report["numpy"] is True


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


def test_default_config_is_valid() -> None:
    """The shipped defaults must construct cleanly, including ``methods=()``."""
    cfg = Config()
    assert cfg.seed == 7
    assert cfg.methods == ()
    assert cfg.n_jobs == 1
    assert list(cfg.as_dict()) == list(Config.FIELDS)


def test_env_overrides_defaults() -> None:
    """VALUA_* must override defaults and coerce to the declared type."""
    cfg = config_from_env(
        {
            f"{ENV_PREFIX}SEED": "17",
            f"{ENV_PREFIX}BUDGET": "512",
            f"{ENV_PREFIX}NOISE_FRAC": "0.25",
            f"{ENV_PREFIX}DATASETS": "moons, digits",
            f"{ENV_PREFIX}METHODS": "tmc",
        }
    )
    assert cfg.seed == 17
    assert cfg.budget == 512
    assert cfg.noise_frac == 0.25
    assert cfg.datasets == ("moons", "digits")
    assert cfg.methods == ("tmc",)
    assert isinstance(cfg.noise_frac, float)


def test_explicit_overrides_beat_env() -> None:
    """Precedence is CLI > VALUA_* > defaults."""
    cfg = config_from_env({f"{ENV_PREFIX}BUDGET": "512"}, budget=64)
    assert cfg.budget == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("budget", 0),  # below minimum
        ("budget", -5),
        ("gold_max_n", 0),
        ("knn_k", 0),
        ("beta_alpha", 0.0),  # below minimum
        ("noise_frac", 1.5),  # above maximum
        ("n_jobs", 8),  # capped on Windows
        ("utility_model", "not_a_model"),  # not in choices
        ("seed", -1),
        ("seed", 2**32),
        ("seeds", ()),  # empty sequence
        ("datasets", ("",)),  # blank entry
        ("output_dir", "   "),
    ],
)
def test_out_of_range_config_raises_e101(field: str, value: object) -> None:
    """Out-of-range or ill-typed values must raise E101, never coerce silently."""
    with pytest.raises(ConfigSchemaError) as exc:
        Config().replace(**{field: value})
    assert exc.value.code == "E101"
    assert exc.value.context.get("field") == field


def test_unknown_config_key_raises_e100() -> None:
    """An unknown key must fail rather than be ignored."""
    with pytest.raises(UnknownConfigKeyError) as exc:
        Config().replace(budgett=10)
    assert exc.value.code == "E100"
    with pytest.raises(UnknownConfigKeyError):
        config_from_env({f"{ENV_PREFIX}BUDGETT": "10"})


def test_unparsable_env_value_raises_e102() -> None:
    """A malformed VALUA_* value must name the field and the target type."""
    with pytest.raises(EnvParseError) as exc:
        config_from_env({f"{ENV_PREFIX}BUDGET": "not-a-number"})
    assert exc.value.code == "E102"
    assert exc.value.context["field"] == "budget"


def test_bool_is_not_accepted_as_int() -> None:
    """``True`` is an int subclass in Python; config must not silently take it."""
    with pytest.raises(ConfigSchemaError):
        Config().replace(budget=True)


# ---------------------------------------------------------------------------
# Error-code catalogue and guards
# ---------------------------------------------------------------------------


def test_every_catalogued_code_is_usable() -> None:
    """Each catalogued code must resolve to a class that can be raised and rendered."""
    for code, meaning in ERROR_CATALOG.items():
        assert has_error_code(code), f"{code} ({meaning}) has no registered class"
        cls = error_class(code)
        assert issubclass(cls, ValuaError)
        assert cls.code == code
        err = cls("synthetic", probe=1)
        assert code in str(err)
        assert err.to_dict()["code"] == code
        assert len(err.fingerprint()) == 16


def test_unknown_error_code_lookup_fails() -> None:
    """Asking for a code outside the catalogue must fail loudly."""
    assert not has_error_code("E999")
    with pytest.raises(KeyError):
        error_class("E999")


def test_error_context_is_sorted_for_stable_logs() -> None:
    """Context rendering must not depend on kwargs insertion order."""
    first = ValuaError("m", b=2, a=1)
    second = ValuaError("m", a=1, b=2)
    assert str(first) == str(second)
    assert first.fingerprint() == second.fingerprint()


def test_leakage_guard_catches_intersection() -> None:
    """train/val overlap must raise E204, never a warning."""
    with pytest.raises(LeakageError) as exc:
        ensure_no_leakage(np.array([0, 1, 2, 3]), np.array([2, 3, 4]))
    assert exc.value.code == "E204"
    assert exc.value.context["n_overlap"] == 2
    # Disjoint, or empty, is fine.
    ensure_no_leakage(np.array([0, 1]), np.array([2, 3]))
    ensure_no_leakage(np.array([], dtype=np.int64), np.array([2, 3]))


def test_dataset_enforces_leakage_guard() -> None:
    """The Dataset type itself must reject overlapping splits."""
    X = np.zeros((4, 2), dtype=np.float64)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    with pytest.raises(LeakageError):
        Dataset(X, y, "leaky", "classification", np.array([0, 1]), np.array([1, 2]))
    ok = Dataset(X, y, "clean", "classification", np.array([2, 3]), np.array([0, 1]))
    assert ok.n_train == 2 and ok.n_val == 2 and ok.n_features == 2


def test_gold_size_guard_raises_e400() -> None:
    """Asking for an exact gold standard beyond max_n must raise, not approximate."""
    ensure_gold_size(12, 12)
    with pytest.raises(GoldTooLargeError) as exc:
        ensure_gold_size(13, 12)
    assert exc.value.code == "E400"
    assert exc.value.context["n"] == 13


def test_determinism_guard_is_bitwise() -> None:
    """E500 must fire on any bit drift, not only large deviations."""
    a = np.array([1.0, 2.0, 3.0])
    ensure_deterministic("x", a, a.copy())
    with pytest.raises(DeterminismError) as exc:
        ensure_deterministic("x", a, a + 1e-15)
    assert exc.value.code == "E500"
    with pytest.raises(DeterminismError):
        ensure_deterministic("x", a, np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def test_budget_remaining_never_negative() -> None:
    """A budget must never report negative headroom."""
    budget = Budget(max_utility_evals=10)
    assert budget.remaining(0) == 10
    assert budget.remaining(10) == 0
    assert budget.remaining(99) == 0


def test_valuation_result_rejects_non_finite() -> None:
    """NaN/Inf in a valuation vector must raise E302 at construction."""
    with pytest.raises(ValuaError) as exc:
        ValuationResult(np.array([1.0, np.nan]), "tmc", 10, 0.5)
    assert exc.value.code == "E302"


def test_utility_spec_rejects_unknown_model() -> None:
    """Utility models are a closed set; typos must fail loudly."""
    spec = UtilitySpec("knn", {"k": 5})
    assert spec.describe() == "knn(k=5)"
    with pytest.raises(ValuaError):
        UtilitySpec("xgboost")


def test_dataset_normalises_dtypes() -> None:
    """float32 input must not leak through: the float64 contract is enforced here."""
    X = np.zeros((3, 2), dtype=np.float32)
    y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    ds = Dataset(X, y, "d", "classification", np.array([2]), np.array([0, 1]))
    assert ds.X.dtype == np.float64
    assert ds.y.dtype == np.int64


def test_console_encoding_is_utf8_capable() -> None:
    """Status glyphs must be printable on a GBK console (pitfall library D)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    print("✅ ⚠️ core invariants: utf-8 console ok")


def test_error_catalog_matches_architecture_spec() -> None:
    """The spec table and the code catalogue must not drift apart.

    ``docs/architecture_spec.md`` section 7 is what a reader consults to
    understand a failure, so a code that exists in one place but not the other
    sends them to the wrong place during an incident. Parsed with a regex rather
    than a TOML loader because the table is prose-adjacent Markdown.
    """
    spec = pathlib.Path(__file__).resolve().parents[1] / "docs" / "architecture_spec.md"
    text = spec.read_text(encoding="utf-8")
    # Only the section-7 table: codes appear as ``| | E204 | ...`` rows.
    section = text.split("## 7. 错误码", 1)[-1].split("## 8.", 1)[0]
    spec_codes = set(re.findall(r"\|\s*\**(E\d{3})\**\s*\|", section))

    assert spec_codes, "could not parse any codes out of the spec table"
    assert spec_codes == set(ERROR_CATALOG), (
        "spec/code drift:/n"
        f"  only in spec: {sorted(spec_codes - set(ERROR_CATALOG))}\n"
        f"  only in code: {sorted(set(ERROR_CATALOG) - spec_codes)}"
    )


def test_e300_and_e103_are_distinct_codes() -> None:
    """Ruling: E300 means only 'unavailable now'; a corrupt registry is E103.

    Merging them would make ``except E300`` swallow a wiring bug as if an
    optional tier were merely missing, and the benchmark would quietly ship with
    a method missing for no visible reason.
    """
    assert error_class("E300") is not error_class("E103")
    assert MethodUnavailableError.code == "E300"
    assert DuplicateRegistrationError.code == "E103"
    # Both are catalogued, and neither borrows the other's meaning.
    assert ERROR_CATALOG["E300"] != ERROR_CATALOG["E103"]


def test_duplicate_registration_raises_e103_at_decoration_time() -> None:
    """The collision must surface at ``@register``, not at lookup time."""

    @register("dup_e103")
    class First:
        @classmethod
        def available(cls) -> bool:
            return True

    try:
        with pytest.raises(DuplicateRegistrationError) as exc:

            @register("dup_e103")
            class Second:
                @classmethod
                def available(cls) -> bool:
                    return True

        assert exc.value.code == "E103"
    finally:
        unregister("dup_e103")


# ---------------------------------------------------------------------------
# Regression guards for bugs that produced plausible-looking numbers
# ---------------------------------------------------------------------------
def test_beta_weights_index_coverage_matches_coalition_sizes() -> None:
    """The weight table must be indexable by every reachable ``|S|``.

    A permutation estimator forms coalitions of size ``0..n-1`` (point ``i`` is
    absent from ``S``), so the table needs exactly ``n`` entries. Reading
    ``weights[size]`` with a shorter table returns garbage **without raising**,
    and the resulting "beta-Shapley" was 28x too large while still looking like
    a plausible vector.
    """
    for n in (6, 8, 10, 12):
        w = beta_weights(n, 1.0, 1.0)
        assert w.size == n, f"beta pmf must span 0..{n - 1}, got {w.size}"
        # Classical weights for (1,1) are the reference the beta pmf reproduces.
        c = classical_weights(n)
        assert c.size == n
        # p_k = C(n-1,k) * u_k, so p_k / C(n-1,k) recovers the size weight.
        for k in range(n):
            assert abs(w[k] / comb(n - 1, k) - c[k]) < 1e-12, f"n={n} k={k}"


def test_classical_weights_reproduce_shapley_efficiency() -> None:
    """``(a,b) = (1,1)`` must equal classical Shapley, so the weights must sum right."""
    for n in (4, 6, 8):
        w = classical_weights(n)
        # sum_k C(n-1,k) * w_k == 1 over the reachable coalition sizes.
        total = sum(comb(n - 1, k) * w[k] for k in range(n))
        assert abs(total - 1.0) < 1e-12, f"n={n}: sum = {total}"


def test_beta_shapley_exact_rejects_a_mis_sized_weight_table() -> None:
    """A wrong-length table must raise, not index past the end."""

    class _U:
        n_evals = 0

        def evaluate(self, subset):
            self.n_evals += 1
            return 0.0

    with pytest.raises(ConfigSchemaError):
        beta_shapley_exact(_U(), 5, np.empty(0), np.ones(3))


def test_dod_picks_the_lowest_l2rel_as_the_strongest_baseline() -> None:
    """L2Rel is an error, so the strongest baseline is the SMALLEST.

    Using max() here picked LOO (1.76) over TMC (0.14) and reported a
    comfortable PASS against a bar no real baseline would have set.
    """
    agg = {
        "valuafuse": {"l2_rel_mean": 0.10, "n_seeds": 3.0},
        "tmc": {"l2_rel_mean": 0.20, "n_seeds": 3.0},
        "beta": {"l2_rel_mean": 0.90, "n_seeds": 3.0},
        "loo": {"l2_rel_mean": 1.80, "n_seeds": 3.0},
    }
    verdict = dod_verdict(agg)
    assert verdict["strongest_baseline"] == "tmc", "picked the wrong baseline"
    assert abs(verdict["ratio"] - 0.5) < 1e-12
    assert verdict["verdict"] == "PASS"
    # A flagship that only beats the *worst* baseline must NOT pass.
    agg["valuafuse"]["l2_rel_mean"] = 1.0
    verdict2 = dod_verdict(agg)
    assert verdict2["verdict"] == "FAIL", "passed against the wrong bar"
    assert verdict2["strongest_baseline"] == "tmc"


def test_dgp_rejects_n_train_larger_than_the_dataset() -> None:
    """``n`` is the TOTAL row count; clamping would silently shrink the gold track.

    Exact Shapley costs ``2^n``, so running the gold track at n=5 while
    reporting n=10 is not a rounding detail -- it changes the difficulty of the
    whole benchmark and hides it.
    """
    with pytest.raises(ValueError, match="must be <"):
        generate("gaussian_mixture", n=10, d=3, seed=7, n_train=10)


# ---------------------------------------------------------------------------
# logreg utility: the coverage gap that let 112 green tests miss a real failure
# ---------------------------------------------------------------------------
def _logreg_dataset() -> Dataset:
    # n_train=10 out of 30 rows, so the training pool does NOT start at id 0.
    # That is what exposed the index-space bug below.
    return generate("gaussian_mixture", n=30, d=6, seed=7, n_train=10)


def test_logreg_raises_e303_on_a_single_class_subset() -> None:
    """A one-class coalition has no decision boundary, so E303 is the right answer.

    This is a mathematical fact, not a limitation to paper over. A silent
    fallback (returning the class prior, say) would let a gold-standard utility
    invent a value, and every downstream number would inherit the fiction.
    """
    data = _logreg_dataset()
    oracle = make_utility("logreg", data, Budget(4096), seed=7, k=5)
    singleton = [0]  # local position of the first training point
    with pytest.raises(UtilityUndefinedError) as exc:
        oracle.evaluate(singleton)
    assert exc.value.code == "E303"
    assert exc.value.context["model"] == "logreg"


def test_logreg_returns_a_value_for_a_multi_class_subset() -> None:
    """The normal path must work, not just the error path."""
    data = _logreg_dataset()
    oracle = make_utility("logreg", data, Budget(4096), seed=7, k=5)
    subset = [0, 1, 2, 3, 4, 5]
    value = oracle.evaluate(subset)
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0
    assert oracle.n_evals == 1, "the E303 path must not silently consume budget"


def test_logreg_is_deterministic_across_repeated_evaluation() -> None:
    """Same subset twice must give the same value: the gold standard has to be stable."""
    data = _logreg_dataset()
    subset = [0, 1, 2, 3, 4, 5, 6, 7]
    a = make_utility("logreg", data, Budget(4096), seed=7, k=5)
    b = make_utility("logreg", data, Budget(4096), seed=7, k=5)
    assert a.evaluate(subset) == b.evaluate(subset)


def test_utility_rejects_a_row_outside_the_training_pool() -> None:
    """Subset ids are global; a validation row is not a legal coalition member.

    Without the global->local map this silently scored another point's value;
    with it, the call fails loudly.
    """
    data = _logreg_dataset()
    oracle = make_utility("logreg", data, Budget(4096), seed=7, k=5)
    # Subset indices are positions in the training pool, so anything at or
    # beyond n_train is illegal -- and would silently score garbage otherwise.
    for bad in ([data.n_train], [-1]):
        with pytest.raises(ShapeMismatchError) as exc:
            oracle.evaluate(bad)
        assert exc.value.code == "E201"


def test_utility_indices_are_local_training_pool_positions() -> None:
    """Subset indices are positions 0..n_train-1 and must pass through intact.

    The oracle is built from ``X[train_idx]``, so position ``i`` already means
    "the i-th training point". Any remapping here would silently score a
    different subset while still returning a plausible number -- which is
    exactly the class of bug this suite exists to catch.
    """
    data = _logreg_dataset()
    oracle = make_utility("logreg", data, Budget(4096), seed=7, k=5)
    for local in range(data.n_train):
        assert oracle._to_local([local]).tolist() == [local]


def test_permutation_estimator_under_logreg_raises_e303_by_contract() -> None:
    """KNOWN LIMITATION, pinned as a test: permutation x logreg fails fast.

    Every permutation's first arrival is a singleton coalition, hence single
    class, hence E303 under logreg. This is why cross-validation cannot use
    logreg as the target utility, and it is correct that it fails rather than
    quietly substituting a value. If someone later makes the permutation
    estimator skip singletons, this test should be revisited deliberately.
    """
    data = _logreg_dataset()
    oracle = make_utility("logreg", data, Budget(4096), seed=7, k=5)
    rng = np.random.default_rng(7)
    with pytest.raises(UtilityUndefinedError) as exc:
        tmc_once(oracle, data.n_train, rng, TMCConfig(n_perm=2, tol=0.0, burn_in=1))
    assert exc.value.code == "E303", "permutation x logreg must raise, not return a value"


def test_permutation_estimator_works_under_knn_and_sgd() -> None:
    """The same estimator must succeed where the utility is defined on singletons."""
    data = _logreg_dataset()
    for model in ("knn", "sgd"):
        oracle = make_utility(model, data, Budget(8192), seed=7, k=5)
        rng = np.random.default_rng(7)
        phi, used, n_perm = tmc_once(
            oracle, data.n_train, rng, TMCConfig(n_perm=2, tol=0.0, burn_in=1)
        )
        assert used > 0, f"{model} performed no evaluations"
        assert n_perm > 0
        assert np.all(np.isfinite(phi)), f"{model} produced non-finite values"


def test_logreg_end_to_end_valuation_path_runs() -> None:
    """Brute-force gold under logreg: n small enough to avoid singletons only.

    A full 2^n enumeration necessarily includes singletons, so this drives the
    **multi-class path end to end** by asking for a coalition-based comparison
    that stays definable, and separately asserts the full enumeration's refusal
    is E303 rather than a crash.
    """
    data = _logreg_dataset()
    oracle = make_utility("logreg", data, Budget(1 << 16), seed=7, k=5)
    with pytest.raises(UtilityUndefinedError):
        exact_shapley(oracle, data.n_train)

    # The LOO path only ever removes one point, so it stays multi-class when the
    # pool has enough of both classes -- this is the realistic logreg use.
    loo = LeaveOneOut()
    present = set(int(data.y[i]) for i in data.train_idx)
    if len(present) >= 2:
        result = loo.value(
            data, make_utility("logreg", data, Budget(4096), seed=7), Budget(4096), None
        )
        assert result.n_values == data.n_train
