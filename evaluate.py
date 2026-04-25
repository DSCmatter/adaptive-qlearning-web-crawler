"""Reproducible evaluation wrapper for crawler benchmarks."""

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run crawler evaluation with a fixed seed.")
    parser.add_argument("--max-pages", type=int, default=20, help="Per-run crawl budget.")
    parser.add_argument("--runs-per-seed", type=int, default=1, help="Repeated runs per seed URL.")
    parser.add_argument("--max-seeds-per-topic", type=int, default=2, help="Seed URLs per topic.")
    parser.add_argument("--output-prefix", default="PHASE_6_EVAL", help="Output filename prefix.")
    parser.add_argument("--crawler-filter", default="", help="Comma-separated crawler names.")
    parser.add_argument("--seed-url", action="append", default=[], help="Explicit seed URL to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--include-run-details", action="store_true", help="Save traces in JSON.")
    parser.add_argument("--enable-diagnostics", action="store_true", help="Enable adaptive diagnostics.")
    parser.add_argument("--disable-online-updates", action="store_true", help="Freeze online updates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "experiments/evaluate_baseline.py",
        "--max-pages",
        str(args.max_pages),
        "--runs-per-seed",
        str(args.runs_per_seed),
        "--max-seeds-per-topic",
        str(args.max_seeds_per_topic),
        "--output-prefix",
        args.output_prefix,
        "--random-seed",
        str(args.seed),
    ]

    if args.crawler_filter:
        command.extend(["--crawler-filter", args.crawler_filter])
    for seed_url in args.seed_url:
        command.extend(["--seed-url", seed_url])
    if args.include_run_details:
        command.append("--include-run-details")
    if args.enable_diagnostics:
        command.append("--enable-diagnostics")
    if args.disable_online_updates:
        command.append("--disable-online-updates")

    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
