"""
Training script to predict position from poses.

Usage:
    python scripts/poses_to_position.py                   # Basic training
    python scripts/poses_to_position.py --wandb           # With wandb logging
    python scripts/poses_to_position.py --sweep           # Start new sweep
    python scripts/poses_to_position.py --sweep SWEEP_ID  # Resume sweep
"""

import sys
from functools import reduce
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb
import motorlab as ml

from motorlab.wandb import (
    init_wandb,
    run_sweep,
    load_sweep_config,
    merge_sweep_config,
    handle_sweep,
)
from scripts.main import setup_base_parser, configure_logging_behavior


REPRESENTATION = "pc"
OLD_GBYK = True

SCHEDULER_CONFIGS = [
    {"type": "step_lr", "step_size": 50, "gamma": 0.7},
    {"type": "step_lr", "step_size": 100, "gamma": 0.8},
    {"type": "step_lr", "step_size": 200, "gamma": 0.9},
    {"type": "cosine_annealing", "eta_min": 1e-5},
    {"type": "cosine_annealing", "eta_min": 1e-4},
    {"type": "cosine_annealing", "eta_min": 1e-3},
    {"type": "reduce_on_plateau", "factor": 0.8, "patience": 5, "min_lr": 1e-6},
    {
        "type": "reduce_on_plateau",
        "factor": 0.9,
        "patience": 10,
        "min_lr": 1e-5,
    },
    {
        "type": "reduce_on_plateau",
        "factor": 0.95,
        "patience": 15,
        "min_lr": 1e-4,
    },
]

SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "val_mse", "goal": "minimize"},
    "parameters": {
        "training.lr": {"values": [1e-3, 5e-3, 1e-2, 2e-2]},
        "training.weight_decay": {"values": [0, 1e-5, 1e-4, 1e-3]},
        "training.scheduler": {"values": SCHEDULER_CONFIGS},
        # "poses.representation": {"values": list(reversed(ml.utils.REPRESENTATIONS))},
    },
}


def load_base_config() -> dict:
    """Get the base configuration for poses_to_position training."""
    sessions = ml.sessions.OLD_GBYK if OLD_GBYK else ml.sessions.GBYK
    experiment = "old_gbyk" if OLD_GBYK else "gbyk"
    config = ml.config.load_default(experiment, sessions)

    config["log_dir"] = "artifacts/logs/poses_to_position/"
    config["intervals"]["include_homing"] = not OLD_GBYK

    ml.utils.setup_representation(config, REPRESENTATION, OLD_GBYK)

    return config


def parse_sweep_config(config: dict, sweep_config: dict) -> dict:
    """
    Parse sweep configuration using dot-notation.

    Converts "a.b.c" keys to nested dictionary access: config["a"]["b"]["c"]
    For keys without dots, sets config[key] = value directly.
    """
    for key, value in sweep_config.items():
        keys = key.split(".")
        parent = reduce(lambda d, k: d.setdefault(k, {}), keys[:-1], config)
        parent[keys[-1]] = value

    return config


def train_single_run():
    """Train a single model run (called by wandb agent in sweep mode)."""
    wandb.init()

    config = load_base_config()
    sweep_config = load_sweep_config()
    config = merge_sweep_config(config, sweep_config)
    config = parse_sweep_config(config, sweep_config)

    # ! WARNING: changes in-place
    configure_logging_behavior(
        config,
        use_wandb=True,
        use_sweep=True,
    )

    ml.model.train(config)


def main():
    """Main entry point."""
    parser = setup_base_parser()
    args = parser.parse_args()

    if args.sweep is not None:
        sweep_arg = None if args.sweep == "new" else args.sweep
        sweep_id = handle_sweep(sweep_arg, SWEEP_CONFIG)
        run_sweep(sweep_id, train_single_run)
    else:
        config = load_base_config()
        configure_logging_behavior(
            config,
            use_wandb=args.wandb,
            use_sweep=False,
        )
        if args.wandb:
            init_wandb()
        ml.model.train(config)


if __name__ == "__main__":
    main()
