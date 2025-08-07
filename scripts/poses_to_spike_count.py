"""
Training script to predict spike count from poses.

Usage:
    python scripts/poses_to_spike_count.py                   # Basic training
    python scripts/poses_to_spike_count.py --wandb           # With wandb logging
    python scripts/poses_to_spike_count.py --sweep           # Start new sweep
    python scripts/poses_to_spike_count.py --sweep SWEEP_ID  # Resume sweep
"""

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import motorlab as ml
import wandb

from scripts.main import setup_base_parser, configure_logging_behavior
from scripts.utils.wandb import (
    init_wandb,
    start_sweep,
    run_sweep,
    load_sweep_config,
    merge_sweep_config,
)


IS_OLD = True
SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "val_local_corr", "goal": "maximize"},
    "parameters": {
        "representation": {"values": ml.utils.REPRESENTATIONS},
    },
}


def load_base_config():
    """Get the base configuration for poses_to_spike_count training."""
    experiment = "gbyk"
    sessions = ml.sessions.OLD_GBYK
    config = ml.config.load_default(experiment, sessions)

    config["checkpoint_dir"] = "artifacts/checkpoint/poses_to_spike_count"
    config["config_dir"] = "artifacts/config/poses_to_spike_count"
    config["log_dir"] = "artifacts/logs/poses_to_spike_count"
    config["track"]["stdout"] = False
    config["in_modalities"] = ["poses"]
    config["out_modalities"] = ["spike_count"]
    config["loss_fn"] = {"spike_count": "poisson"}
    config["metric"] = {"spike_count": "correlation"}
    config["intervals"]["include_homing"] = False

    config["training"]["lr"] = 5e-3
    config["training"]["eta_min"] = 3e-4
    config["training"]["n_epochs"] = 1000

    config["model"]["architecture"] = "fc"
    config["model"]["embedding_dim"] = 256
    config["model"]["hidden_dim"] = 512
    config["model"]["readout"] = "softplus"

    config["poses"]["representation"] = "allocentric"

    return config


def train_single_run():
    """Train a single model run (called by wandb agent in sweep mode)."""
    if wandb.run is None:
        wandb.init(project="motorlab")

    config = load_base_config()
    sweep_config = load_sweep_config()

    if sweep_config:
        config = merge_sweep_config(config, sweep_config)
        if "representation" in sweep_config:
            ml.utils.setup_representation(
                config,
                sweep_config["representation"],
                IS_OLD,
            )

    config = configure_logging_behavior(
        config, use_wandb=config["track"]["wandb"], use_sweep=bool(sweep_config)
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
        config = configure_logging_behavior(
            config, use_wandb=args.wandb, use_sweep=False
        )
        if args.wandb:
            init_wandb(project="motorlab")
        ml.model.train(config)


if __name__ == "__main__":
    main()
