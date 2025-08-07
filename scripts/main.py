"""
Shared utilities and CLI argument parsing for training scripts.
"""

import argparse
import sys

from pathlib import Path


def setup_base_parser() -> argparse.ArgumentParser:
    """
    Create base argument parser with common flags for all training scripts.

    Returns
    -------
    argparse.ArgumentParser
        Base parser with common arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train machine learning models with optional wandb integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (disables stdout printing)",
    )

    parser.add_argument(
        "--sweep",
        type=str,
        nargs="?",
        const="new",
        help="Run wandb sweep. Pass sweep ID to resume existing sweep, or no value to start new sweep (disables stdout printing)",
    )

    return parser


def add_project_root_to_path():
    """Add project root to Python path for imports."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def configure_logging_behavior(
    config: dict, use_wandb: bool, use_sweep: bool
) -> None:
    """
    Configure logging behavior based on flags.

    Parameters
    ----------
    config : dict
        Base configuration dictionary.
    use_wandb : bool
        Whether wandb logging is enabled.
    use_sweep : bool
        Whether running in sweep mode.

    Returns
    -------
    dict
        Updated configuration with appropriate logging settings.
    """
    if use_wandb or use_sweep:
        config["track"]["wandb"] = True
        config["track"]["stdout"] = False
    else:
        config["track"]["wandb"] = False
        config["track"]["stdout"] = config["track"].get("stdout", True)
