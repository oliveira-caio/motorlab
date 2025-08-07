"""
Training script to predict position from poses.

Usage:
    python scripts/poses_to_position.py                   # Basic training
    python scripts/poses_to_position.py --wandb           # With wandb logging
    python scripts/poses_to_position.py --sweep           # Start new sweep
    python scripts/poses_to_position.py --sweep SWEEP_ID  # Resume sweep
"""

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb
import motorlab as ml

from motorlab.wandb import (
    init_wandb,
    start_sweep,
    run_sweep,
    load_sweep_config,
    merge_sweep_config,
)
from scripts.main import setup_base_parser, configure_logging_behavior


REPRESENTATION = "pc"
IS_OLD = False
SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "val_mse", "goal": "minimize"},
    "parameters": {
        "representation": {"values": list(reversed(ml.utils.REPRESENTATIONS))},
    },
}


def load_base_config() -> dict:
    """Get the base configuration for poses_to_position training."""
    sessions = ml.sessions.OLD_GBYK if IS_OLD else ml.sessions.GBYK
    experiment = "old_gbyk" if IS_OLD else "gbyk"
    config = ml.config.load_default(experiment, sessions)

    config["log_dir"] = "artifacts/logs/poses_to_position/"
    config["model"]["n_layers"] = 5
    config["intervals"]["include_homing"] = not IS_OLD

    ml.utils.setup_representation(config, REPRESENTATION, IS_OLD)

    return config


def train_single_run():
    """Train a single model run (called by wandb agent in sweep mode)."""
    wandb.init()

    config = load_base_config()
    sweep_config = load_sweep_config()

    config = merge_sweep_config(config, sweep_config)
    if "representation" in sweep_config:
        ml.utils.setup_representation(
            config,
            sweep_config["representation"],
            IS_OLD,
        )

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

    if args.sweep:
        if args.sweep == "new":
            sweep_id = start_sweep(SWEEP_CONFIG)
            print(f"Starting sweep: {sweep_id}")
        else:
            sweep_id = args.sweep
            print(f"Resuming sweep: {sweep_id}")
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
