"""
Unified training script for motorlab experiments.

Usage:
    python motorlab/train.py --setup quick.yml
    python motorlab/train.py --setup detailed.yml --wandb
    python motorlab/train.py --setup sweep.yml --sweep
    python motorlab/train.py --setup sweep.yml --sweep my_custom_sweep
    python motorlab/train.py --setup sweep.yml --sweep existing_sweep_id
"""

import argparse

import motorlab as ml

from motorlab.utils import wandb as wandb_utils


def train_sweep_run(base_config: dict) -> None:
    """Training function for wandb sweep runs."""
    wandb_utils.init_run()
    sweep_params = wandb_utils.load_sweep_config()
    ml.config.deep_update(base_config, sweep_params)
    ml.model.train(base_config)


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Unified training script for motorlab experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--setup",
        required=True,
        help="Setup YAML file (from setups/ directory)",
    )

    parser.add_argument(
        "--set",
        action="append",
        help="Override config values: key=value (e.g., training.optimizer.lr=1e-3)",
    )

    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )

    parser.add_argument(
        "--sweep",
        nargs="?",
        const="",
        help="Run wandb sweep. Empty=random name, string=sweep name or ID to join/create",
    )

    args = parser.parse_args()

    setup_config = ml.config.load_setup(args.setup)
    sessions = ml.sessions.load(setup_config["experiment"])

    if args.sweep is None:
        for override in args.set or []:
            if "=" not in override:
                print(
                    f"Warning: Invalid override format '{override}', expected key=value"
                )
                continue
            key, value = override.split("=", 1)
            ml.config.apply_override(setup_config, key, value)

    default_config = ml.config.load_default(
        setup_config["experiment"], sessions, setup_config["task"]
    )
    # changes default_config in-place.
    ml.config.deep_update(default_config, setup_config)
    config = default_config

    if args.sweep is not None:
        ml.config.logging_behavior(config, use_wandb=True, use_sweep=True)

        if "sweep" not in config:
            raise ValueError(
                f"Setup file {args.setup} must specify 'sweep' configuration for sweep mode"
            )

        sweep_config = config["sweep"]
        sweep_id = wandb_utils.handle_sweep(
            args.sweep if args.sweep else None, sweep_config
        )

        def train_function():
            train_sweep_run(config)

        wandb_utils.run_sweep(sweep_id, train_function)
        return

    ml.config.logging_behavior(config, use_wandb=args.wandb, use_sweep=False)

    if args.wandb:
        wandb_utils.init_run()

    ml.model.train(config)


if __name__ == "__main__":
    main()
