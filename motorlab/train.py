"""
Unified training script for motorlab experiments.

Usage:
    python motorlab/train.py --task poses_to_location --experiment gbyk
    python motorlab/train.py --task poses_to_spike_count --experiment old_gbyk --setup quick.yml --wandb
    python motorlab/train.py --task poses_to_location --experiment pg --set training.optimizer.lr=1e-3 --sweep new
"""

import argparse

from copy import deepcopy
from pathlib import Path

import wandb
import yaml

import motorlab as ml


TASK_REGISTRY = {
    "poses_to_location": {
        "data": {"input_modalities": "poses", "output_modalities": "location"},
        "training": {"loss_function": "mse", "metric": "mse"},
        "model": {"readout": "linear"},
        "modalities": {"poses": {"representation": "egocentric"}},
    },
    "poses_to_spike_count": {
        "data": {
            "input_modalities": "poses",
            "output_modalities": "spike_count",
        },
        "training": {
            "loss_function": "poisson",
            "metric": "correlation",
            "optimizer": {"lr": 5e-3},
            "max_epochs": 1000,
        },
        "model": {
            "architecture": "fc",
            "embedding_dim": 128,
            "hidden_dim": 128,
            "readout": "softplus",
        },
        "modalities": {
            "poses": {"representation": "allocentric"},
            "intervals": {"include_homing": False},
        },
    },
}


DATASET_CONFIGS = {
    "gbyk": {
        "sessions": ml.sessions.GBYK,
        "modalities": {"intervals": {"include_homing": True}},
    },
    "old_gbyk": {
        "sessions": ml.sessions.OLD_GBYK,
        "modalities": {"intervals": {"include_homing": False}},
    },
    "pg": {"sessions": ml.sessions.PG},
}


def deep_update(base_dict: dict, update_dict: dict) -> None:
    """
    Deep merge update_dict into base_dict, preserving nested structure.

    Parameters
    ----------
    base_dict : dict
        Dictionary to update (modified in-place)
    update_dict : dict
        Dictionary with updates to apply
    """
    for key, value in update_dict.items():
        if (
            key in base_dict
            and isinstance(base_dict[key], dict)
            and isinstance(value, dict)
        ):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def parse_value(value: str):
    """
    Parse string value to appropriate Python type.

    Parameters
    ----------
    value : str
        String value to parse

    Returns
    -------
    Parsed value (bool, int, float, or str)
    """
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    # Return as string
    return value


def apply_override(config: dict, key: str, value: str) -> None:
    """
    Apply a.b.c=value to config["a"]["b"]["c"]=parsed_value.

    Parameters
    ----------
    config : dict
        Configuration dictionary to update
    key : str
        Dot-separated key path (e.g., "training.optimizer.lr")
    value : str
        String value to parse and set
    """
    keys = key.split(".")
    target = config
    for k in keys[:-1]:
        target = target.setdefault(k, {})
    target[keys[-1]] = parse_value(value)


def configure_logging_behavior(
    config: dict,
    use_wandb: bool,
    use_sweep: bool,
) -> None:
    """
    Configure logging behavior based on flags.

    Parameters
    ----------
    config : dict
        Configuration dictionary to update
    use_wandb : bool
        Whether wandb logging is enabled
    use_sweep : bool
        Whether running in sweep mode
    """
    if use_wandb or use_sweep:
        config["tracking"]["wandb"] = True
        config["tracking"]["stdout"] = False
    else:
        config["tracking"]["wandb"] = False
        config["tracking"]["stdout"] = config["tracking"].get("stdout", True)


def train_run():
    """Train a single model run (called by wandb agent in sweep mode)."""
    # TODO: Implement sweep-specific logic when needed
    wandb.init()


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Unified training script for motorlab experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASK_REGISTRY.keys()),
        help="Task to train",
    )

    parser.add_argument(
        "--experiment",
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset/experiment to use",
    )

    parser.add_argument(
        "--setup", help="Setup YAML file (from setups/ directory)"
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
        type=str,
        nargs="?",
        const="new",
        help="Run wandb sweep. Pass sweep ID to resume existing sweep or 'new' to create one.",
    )

    args = parser.parse_args()

    # 1. Load default config (lowest precedence)
    dataset_config = DATASET_CONFIGS[args.experiment]
    config = ml.config.load_default(
        args.experiment, dataset_config["sessions"], args.task
    )

    # 2. Apply task-specific settings
    task_config = deepcopy(TASK_REGISTRY[args.task])
    deep_update(config, task_config)

    # 3. Apply dataset-specific settings
    deep_update(config, dataset_config)

    # 4. Apply setup YAML (medium precedence)
    if args.setup:
        setup_path = Path("setups") / args.setup
        if setup_path.exists():
            with open(setup_path) as f:
                setup_config = yaml.safe_load(f)
            if setup_config:
                deep_update(config, setup_config)
        else:
            print(f"Warning: Setup file {setup_path} not found")

    # 5. Apply command-line overrides (highest precedence)
    for override in args.set or []:
        if "=" not in override:
            print(
                f"Warning: Invalid override format '{override}', expected key=value"
            )
            continue
        key, value = override.split("=", 1)
        apply_override(config, key, value)

    if args.sweep is not None:
        # For now, just show what would happen
        print(f"Sweep mode requested: {args.sweep}")
        print("Sweep functionality not yet implemented")
        return

    configure_logging_behavior(config, use_wandb=args.wandb, use_sweep=False)

    if args.wandb:
        ml.wandb.init()

    ml.model.train(config)


if __name__ == "__main__":
    main()
