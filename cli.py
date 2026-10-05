"""Command-line entry point.

Author: 晨星

Deliberately thin: parse arguments, call into the layers, print. No business
logic and no computation lives here (architecture spec section 4). The CLI is
also the only place allowed to print, so it owns the console-encoding fix.
"""

from __future__ import annotations

import argparse
import contextlib
import sys

from core.config import ENV_PREFIX, config_from_env
from core.errors import ERROR_CATALOG, ValuaError
from core.registry import available_methods, registry_snapshot, tier_report
from core.seed import SeedBank
from eval.report import format_dod_table
from pipeline.benchmark import benchmark

__all__ = ["main"]


def _use_utf8_console() -> None:
    """Force UTF-8 output so status glyphs do not raise UnicodeEncodeError.

    Windows consoles default to GBK, where ``✅``/``⚠️`` raise
    UnicodeEncodeError the moment they are printed (pitfall library D).
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(ValueError, OSError):
                stream.reconfigure(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="valuaforge",
        description="ValuaForge - data valuation (Data Shapley) for training sets",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show resolved configuration and tier availability")

    seed_cmd = sub.add_parser("seed", help="derive a named random stream")
    seed_cmd.add_argument("name", help="stream name, e.g. val/tmc/digits/7")

    sub.add_parser("methods", help="list registered valuation methods")

    sub.add_parser("errors", help="list every error code and its meaning")

    bench = sub.add_parser("benchmark", help="run the benchmark grid")
    bench.add_argument(
        "--n", type=int, default=10, help="Shapley problem size (gold costs 2^n)"
    )
    bench.add_argument("--d", type=int, default=6, help="feature count")
    bench.add_argument(
        "--budget", type=int, default=2048, help="utility evaluations per method"
    )
    bench.add_argument("--out", default=None, help="output directory")
    bench.add_argument("--seeds", default=None, help="comma-separated seeds")
    bench.add_argument("--datasets", default=None, help="comma-separated dataset names")
    bench.add_argument(
        "--no-determinism",
        action="store_true",
        help="skip the double-run bitwise check (faster, weaker evidence)",
    )

    return parser


def _cmd_info(_args: argparse.Namespace) -> int:
    cfg = config_from_env()
    print("ValuaForge configuration")
    print(f"  {'key':<20} {'value'}")
    print(f"  {'-' * 20} {'-' * 40}")
    for key, value in cfg.as_dict().items():
        print(f"  {key:<20} {value!r}")
    print()
    print(f"  environment prefix: {ENV_PREFIX}*  (overrides defaults)")
    print(f"  env override active: {'yes' if len(sys.argv) > 2 else 'no explicit flags'}")
    print()
    print("Tier availability")
    for name, ok in sorted(tier_report().items()):
        print(f"  {name:<20} {'available' if ok else 'MISSING'}")
    return 0


def _cmd_seed(args: argparse.Namespace) -> int:
    cfg = config_from_env()
    bank = SeedBank(cfg.seed)
    gen = bank.stream(args.name)
    draws = gen.standard_normal(8)
    print(f"stream : {args.name}")
    print(f"seed   : {cfg.seed}")
    print(f"entropy: {bank.entropy_for(args.name)}")
    print("draws  :")
    for i, value in enumerate(draws):
        print(f"  [{i}] {float(value):.17g}")
    return 0


def _cmd_methods(_args: argparse.Namespace) -> int:
    names = available_methods()
    print(f"Registered methods: {len(names)}")
    for name, qualname in sorted(registry_snapshot().items()):
        mark = "available" if name in names else "unavailable"
        print(f"  {name:<20} {mark:<12} {qualname}")
    if not names:
        print("  (none registered yet -- valuation layer not implemented)")
    return 0


def _cmd_errors(_args: argparse.Namespace) -> int:
    print(f"Error codes: {len(ERROR_CATALOG)}")
    for code, meaning in sorted(ERROR_CATALOG.items()):
        print(f"  {code}  {meaning}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the real benchmark and print the DoD table.

    Exits non-zero when the DoD fails, so a Makefile target or CI step cannot
    read a missed gate as success. The previous version raised E300 here on
    purpose ("not implemented yet"), which was correct while the layers were
    absent but is now obsolete: the pipeline measures real valuations.
    """
    cfg = config_from_env()
    out = args.out or cfg.output_dir
    seeds = (
        tuple(int(s) for s in args.seeds.split(",") if s.strip())
        if args.seeds
        else tuple(cfg.seeds)
    )
    datasets = (
        tuple(d.strip() for d in args.datasets.split(",") if d.strip())
        if args.datasets
        else tuple(d for d in cfg.datasets if d in ("gaussian_mixture", "label_noise"))
    )
    print(
        f"benchmark: n={args.n} d={args.d} budget={args.budget} "
        f"seeds={list(seeds)} datasets={list(datasets)}"
    )
    sys.stdout.flush()

    payload = benchmark(
        datasets=datasets,
        seeds=seeds,
        n=args.n,
        d=args.d,
        budget=args.budget,
        out=out,
        check_determinism=not args.no_determinism,
    )
    print(format_dod_table(payload["aggregate"], payload["dod"]))
    print()
    print(f"determinism: {payload['meta']['determinism']}")
    print(f"failures   : {len(payload['meta'].get('failures', []))}")
    print(f"artifact   : {payload['_path']}")

    verdict = str(payload["dod"].get("verdict"))
    if verdict != "PASS":
        print()
        print(f"DoD verdict: {verdict}")
        return 1
    return 0


_COMMANDS = {
    "info": _cmd_info,
    "seed": _cmd_seed,
    "methods": _cmd_methods,
    "errors": _cmd_errors,
    "benchmark": _cmd_benchmark,
}


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    _use_utf8_console()
    args = _build_parser().parse_args(argv)
    handler = _COMMANDS[args.command]
    try:
        return handler(args)
    except ValuaError as exc:
        # Structured, code-first output: callers grep the code, not the prose.
        print(f"error {exc.code}: {exc.message}", file=sys.stderr)
        for key in sorted(exc.context):
            print(f"  {key} = {exc.context[key]!r}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
