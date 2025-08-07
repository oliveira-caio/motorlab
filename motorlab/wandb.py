"""Utility functions for wandb integration."""

import copy

import wandb


def init_wandb(name: str | None = None):
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
