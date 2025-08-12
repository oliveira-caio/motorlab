"""Utility functions for wandb integration."""

import copy

import wandb


def init(name: str | None = None):
    wandb.init(
        project="motorlab",
        name=name,
    )


def start_sweep(sweep_config: dict) -> str:
    sweep_id = wandb.sweep(sweep_config, project="motorlab")
    print(f"Created sweep: {sweep_id}")
    return sweep_id


def run_sweep(sweep_id: str, train_function):
    wandb.agent(sweep_id, train_function, project="motorlab")


def check_sweep_exists(sweep_id: str) -> bool:
    """
    Check if a wandb sweep exists.

    Parameters
    ----------
    sweep_id : str
        The sweep ID to check

    Returns
    -------
    bool
        True if sweep exists, False otherwise
    """
    try:
        api = wandb.Api()
        api.sweep(f"motorlab/{sweep_id}")
        return True
    except Exception:
        # If any error occurs (sweep not found, network issues, etc.), assume it doesn't exist
        return False


def handle_sweep(sweep_arg: str | None, sweep_config: dict) -> str:
    """
    Handle sweep creation or resumption based on argument.

    Parameters
    ----------
    sweep_arg : str | None
        Sweep argument: None for new random sweep, string for specific sweep ID
    sweep_config : dict
        Sweep configuration to use for new sweeps

    Returns
    -------
    str
        Sweep ID to use
    """
    if sweep_arg is None:
        # No argument provided, create new sweep with random ID
        sweep_id = start_sweep(sweep_config)
        print(f"Created new sweep: {sweep_id}")
        return sweep_id

    # Check if provided ID exists
    if check_sweep_exists(sweep_arg):
        print(f"Resuming existing sweep: {sweep_arg}")
        return sweep_arg
    else:
        # ID doesn't exist, create new sweep with this ID as name
        sweep_config_with_name = sweep_config.copy()
        sweep_config_with_name["name"] = sweep_arg
        sweep_id = start_sweep(sweep_config_with_name)
        print(f"Created new sweep with name '{sweep_arg}': {sweep_id}")
        return sweep_id


def load_sweep_config() -> dict:
    """Get configuration from wandb sweep, if running in a sweep."""
    return dict(wandb.config)


def merge_sweep_config(base_config: dict, sweep_config: dict) -> dict:
    """
    Merge sweep parameters into base config.

    Only parameters present in sweep_config will override base_config values.
    This preserves all non-swept parameters from the base configuration.

    Parameters
    ----------
    base_config : dict
        Base configuration dictionary.
    sweep_config : dict
        Sweep parameters from wandb.

    Returns
    -------
    dict
        Merged configuration.
    """
    merged = copy.deepcopy(base_config)

    def deep_merge(target, source):
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                deep_merge(target[key], value)
            else:
                target[key] = value

    deep_merge(merged, sweep_config)
    return merged


def download_sweep_data(
    sweep_id: str, entity: str = "sinzlab", project: str = "motorlab"
) -> list:
    """Download all runs from a wandb sweep."""
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs_data = []

    for run in sweep.runs:
        run_data = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "created_at": run.created_at,
            "runtime": run._attrs.get("runtime", 0),
        }

        for key, value in run.config.items():
            run_data[f"config.{key}"] = value

        for key, value in run.summary.items():
            if not key.startswith("_"):
                run_data[f"summary.{key}"] = value

        runs_data.append(run_data)

    return runs_data
