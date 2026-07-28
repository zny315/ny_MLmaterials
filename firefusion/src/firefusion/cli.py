from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import run_benchmark
from .reporting import summarize_runs
from .transfer import run_transfer


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tabpfn-device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="firefusion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _common(subparsers.add_parser("benchmark", help="run lab/literature benchmarks"))
    _common(subparsers.add_parser("transfer", help="run nested Bayesian transfer"))
    _common(subparsers.add_parser("all", help="run benchmark, transfer, and summary"))
    summary = subparsers.add_parser("summarize", help="combine result directories")
    summary.add_argument("--inputs", required=True, nargs="+", type=Path)
    summary.add_argument("--output", required=True, type=Path)
    summary.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "benchmark":
        path = run_benchmark(args.config, args.data, args.output,
                             tabpfn_device=args.tabpfn_device,
                             shard_index=args.shard_index, shard_count=args.shard_count,
                             overwrite=args.overwrite)
    elif args.command == "transfer":
        path = run_transfer(args.config, args.data, args.output,
                            tabpfn_device=args.tabpfn_device,
                            shard_index=args.shard_index, shard_count=args.shard_count,
                            overwrite=args.overwrite)
    elif args.command == "summarize":
        path = summarize_runs(args.inputs, args.output, overwrite=args.overwrite)
    else:
        root = args.output
        if root.exists() and any(root.iterdir()) and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite nonempty output directory: {root}")
        root.mkdir(parents=True, exist_ok=True)
        benchmark = run_benchmark(args.config, args.data, root / "benchmark",
                                  tabpfn_device=args.tabpfn_device,
                                  shard_index=args.shard_index, shard_count=args.shard_count,
                                  overwrite=args.overwrite)
        transfer = run_transfer(args.config, args.data, root / "transfer",
                                tabpfn_device=args.tabpfn_device,
                                shard_index=args.shard_index, shard_count=args.shard_count,
                                overwrite=args.overwrite)
        summarize_runs([benchmark, transfer], root / "summary", overwrite=args.overwrite)
        path = root
    print(path.resolve())


if __name__ == "__main__":
    main()
