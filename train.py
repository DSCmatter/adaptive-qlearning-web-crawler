"""Reproducible training wrapper for the adaptive crawler project."""

import argparse
import os
import random
import runpy


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train project models with a fixed seed.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--stage",
        choices=("all", "gnn", "agent"),
        default="all",
        help="Training stage to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.stage in ("all", "gnn"):
        runpy.run_path("experiments/train_gnn.py", run_name="__main__")

    if args.stage in ("all", "agent"):
        runpy.run_path("experiments/train_agent.py", run_name="__main__")


if __name__ == "__main__":
    main()
