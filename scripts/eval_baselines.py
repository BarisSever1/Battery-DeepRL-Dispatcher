"""Evaluate rule-based baselines against the BESSEnv environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Tuple

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.policies import (
    evaluate_policy,
    pv_da_plan,
    pv_ds3_plan,
)
from src.envs import BESSEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline policies")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed")
    parser.add_argument(
        "--degradation-model",
        type=str,
        default="nonlinear",
        choices=["nonlinear", "linear"],
        help="Degradation cost model used by the environment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = BESSEnv(degradation_model=args.degradation_model)
    seeds = [args.seed + i for i in range(args.episodes)]

    baselines: List[Tuple[str, Callable[[BESSEnv, int], List[float]]]] = [
        ("PV+DS3 (Reserve Only)", pv_ds3_plan),
        ("PV+DA (Heuristic)", pv_da_plan),
    ]

    for name, plan_fn in baselines:
        evaluate_policy(env, name, plan_fn, seeds)


if __name__ == "__main__":
    main()


