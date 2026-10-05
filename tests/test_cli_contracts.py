"""CLI exit-code contracts.

Author: 晨星

Callers and CI branch on **exit codes**, not on prose, so that is what these
pin.

The original version of this file asserted that ``benchmark`` must exit
non-zero *because the pipeline did not exist yet*. That contract is now
satisfied by a different mechanism: the pipeline is real, so the subcommand
exits 0 **only** when the DoD actually passes, and non-zero when it fails or
when it measures nothing. The underlying rule is unchanged and still worth
locking:

    a subcommand that measures nothing must never look like success.

The tests run against a tiny grid (n=6, 1 seed, small budget) to stay fast; the
full-size evidence is in ``results/benchmark.json`` and ``repro/``.
"""

from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

import cli
from core.errors import ERROR_CATALOG, MethodUnavailableError

# Small enough to run in a unit test, large enough for a 2^n gold enumeration.
FAST = ("--n", "6", "--d", "3", "--budget", "256", "--seeds", "7", "--no-determinism")


def _run(*argv: str) -> tuple[int, str, str]:
    """Invoke the CLI, returning ``(exit_code, stdout, stderr)``."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = cli.main(list(argv))
        except SystemExit as exc:  # argparse usage errors
            code = int(exc.code or 0)
    return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# informational subcommands: must succeed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("argv", [("info",), ("methods",), ("errors",), ("seed", "demo")])
def test_informational_subcommands_exit_zero(argv: tuple[str, ...]) -> None:
    """These do real work (config resolution, RNG derivation), so they must pass."""
    code, out, _err = _run(*argv)
    assert code == 0, f"cli {' '.join(argv)} should exit 0, got {code}"
    assert out.strip(), f"cli {' '.join(argv)} produced no output"


# ---------------------------------------------------------------------------
# benchmark: the load-bearing contract
# ---------------------------------------------------------------------------
def test_benchmark_measures_something_and_says_so() -> None:
    """The subcommand must produce a real verdict, not a placeholder."""
    with tempfile.TemporaryDirectory() as tmp:
        code, out, _err = _run("benchmark", *FAST, "--out", tmp)
        assert "DoD:" in out, "benchmark printed no DoD line"
        assert "verdict:" in out
        assert list(Path(tmp).glob("*.json")), "benchmark wrote no artifact"
        # exit code must agree with the verdict it printed
        assert ("PASS" in out) == (code == 0), f"exit {code} disagrees with the printed verdict"


def test_benchmark_writes_a_parseable_artifact() -> None:
    """The artifact is the deliverable, so its shape is part of the contract."""
    with tempfile.TemporaryDirectory() as tmp:
        _code, _out, _err = _run("benchmark", *FAST, "--out", tmp)
        files = list(Path(tmp).glob("*.json"))
        assert len(files) == 1, f"expected exactly one artifact, got {files}"
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        for key in ("meta", "aggregate", "dod", "rows"):
            assert key in payload, f"artifact missing {key!r}"
        assert payload["rows"], "artifact contains no rows"
        assert "determinism" in payload["meta"]


def test_benchmark_reports_failures_instead_of_hiding_them() -> None:
    """A skipped cell must be visible in stdout, never silently dropped.

    A benchmark that quietly omits the cells it could not run is the failure
    mode this whole repository keeps guarding against.
    """
    with tempfile.TemporaryDirectory() as tmp:
        _code, out, _err = _run("benchmark", *FAST, "--out", tmp)
        assert "failures" in out, "benchmark did not report its failure count"


def test_benchmark_refuses_an_unknown_dataset() -> None:
    """A typo must not silently become a valid-looking benchmark row (E202)."""
    with tempfile.TemporaryDirectory() as tmp:
        code, _out, _err = _run(
            "benchmark", *FAST, "--datasets", "no_such_dataset", "--out", tmp
        )
        assert code != 0, "unknown dataset was accepted"
        assert not list(Path(tmp).glob("*.json")), "failed benchmark wrote result files"


# ---------------------------------------------------------------------------
# usage errors
# ---------------------------------------------------------------------------
def test_missing_required_arg_is_a_usage_error() -> None:
    code, _, _ = _run("seed")  # 'name' is required
    assert code == 2


def test_unknown_command_is_a_usage_error() -> None:
    code, _, _ = _run("definitely-not-a-command")
    assert code == 2


# ---------------------------------------------------------------------------
# error catalogue sanity
# ---------------------------------------------------------------------------
def test_e300_message_is_about_availability_not_corruption() -> None:
    """E300 must not be reused for registry corruption -- that is a separate code."""
    assert "unavailable" in ERROR_CATALOG["E300"]
    assert MethodUnavailableError.code == "E300"
