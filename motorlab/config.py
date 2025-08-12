"""
Motorlab configuration system.
"""

from pathlib import Path

import yaml


SAMPLING_RATE = 50


def load_default(
    experiment: str,
    sessions: list[str],
    task: str = "poses_to_location",
) -> dict:
    """
    Get a default configuration dictionary for a given experiment and sessions.

    Parameters
    ----------
    experiment : str
        Name of the experiment ('gbyk', 'pg', etc.)
    sessions : list of str
        List of session names
    task : str, optional
        Name of the ML task, by default "poses_to_location"

    Returns
    -------
    dict
        Default configuration dictionary.
    """
    config = {
        "task": task,
        "experiment": experiment,
        "sessions": sessions,
        "seed": 0,
        "paths": {
            "artifacts_dir": "artifacts",
            "data_dir": "data",
        },
        "data": {
            "input_modalities": "poses",
            "output_modalities": "location",
            "dataset": {
                "stride": 1000,  # milliseconds
                "concat_input": True,
                "concat_output": True,
            },
            "dataloader": {
                "variable_length": True,
                "min_length": 100,  # milliseconds
                "max_length": 4000,  # milliseconds
                "batch_size": 64,
            },
        },
        "model": {
            "architecture": "fc",
            "embedding_dim": 256,
            "hidden_dim": 256,
            "n_layers": 3,
            "readout": "linear",
        },
        "training": {
            "max_epochs": 500,
            "loss_function": "mse",
            "metric": "mse",
            "optimizer": {"type": "adam", "lr": 1e-2, "weight_decay": 1e-4},
            "scheduler": {"type": "step_lr", "step_size": 100, "gamma": 0.8},
            "validation": {"frequency": 25, "gradient_threshold": 0.5},
            "early_stopping": {
                "enabled": False,
                "patience": 6,
                "min_delta": 0.0,
                "gradient_threshold": 0.5,
            },
        },
        "tracking": {
            "stdout": True,
            "wandb": False,
            "checkpoint": True,
            "logging": True,
        },
        "modalities": {
            "location": {
                "representation": "com",
            },
            "poses": {
                "representation": "egocentric",
            },
            "intervals": {
                "include_trial": True,
                "include_homing": True,
                "include_sitting": True,
                "balance_intervals": False,
            },
        },
    }
    return config


def save(config: dict) -> None:
    """
    Save configuration dictionary to a YAML file.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    """
    config_dir = Path(
        config["paths"]["config_save_dir"]
        if "config_save_dir" in config
        else config["paths"]["config_dir"]
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{config['uid']}.yml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)


def preprocess(config: dict) -> dict:
    """
    Preprocess the user-friendly config into a format suitable for internal use.

    This function:
    - Converts string modalities to lists
    - Creates readout_map from readout if not provided
    - Converts string metrics to per-modality dictionaries
    - Auto-generates paths from task name
    - Adds backward compatibility keys for existing codebase
    - Adds any other implicit keys or transformations

    Parameters
    ----------
    config : dict
        User-friendly configuration dictionary

    Returns
    -------
    dict
        Preprocessed configuration dictionary ready for internal use
    """
    processed_config = config.copy()

    if isinstance(processed_config["data"]["input_modalities"], str):
        processed_config["data"]["input_modalities"] = [
            processed_config["data"]["input_modalities"]
        ]

    if isinstance(processed_config["data"]["output_modalities"], str):
        processed_config["data"]["output_modalities"] = [
            processed_config["data"]["output_modalities"]
        ]

    readout = processed_config["model"]["readout"]
    if isinstance(readout, dict):
        processed_config["model"]["readout_map"] = readout
    else:
        processed_config["model"]["readout_map"] = {
            modality: readout
            for modality in processed_config["data"]["output_modalities"]
        }

    if "metric" not in processed_config["training"]:
        processed_config["training"]["metric"] = None

    if isinstance(processed_config["training"].get("metric"), str):
        metric_value = processed_config["training"]["metric"]
        processed_config["training"]["metric"] = {
            modality: metric_value
            for modality in processed_config["data"]["output_modalities"]
        }

    if isinstance(processed_config["training"].get("loss_function"), str):
        loss_fn_value = processed_config["training"]["loss_function"]
        processed_config["training"]["loss_function"] = {
            modality: loss_fn_value
            for modality in processed_config["data"]["output_modalities"]
        }

    for path_type in ["checkpoint", "config", "logs"]:
        key = f"{path_type}_dir"
        if key not in processed_config:
            base_dir = processed_config["paths"]["artifacts_dir"]
            task = processed_config["task"]
            processed_config["paths"][key] = f"{base_dir}/{path_type}/{task}"

    if "stride" in config["data"]["dataset"]:
        processed_config["data"]["dataset"]["stride"] = (
            config["data"]["dataset"]["stride"] // SAMPLING_RATE
        )

    dataloader_config = processed_config["data"]["dataloader"]
    variable_length = dataloader_config.get("variable_length", False)

    if variable_length:
        if (
            "min_length" not in dataloader_config
            or "max_length" not in dataloader_config
        ):
            raise ValueError(
                "Variable length requires 'min_length' and 'max_length'"
            )

        processed_config["data"]["dataloader"]["min_length"] = (
            dataloader_config["min_length"] // SAMPLING_RATE
        )
        processed_config["data"]["dataloader"]["max_length"] = (
            dataloader_config["max_length"] // SAMPLING_RATE
        )
    else:
        if "length" in dataloader_config:
            length = dataloader_config["length"]
        elif "max_length" in dataloader_config:
            length = dataloader_config["max_length"]
        else:
            raise ValueError("Fixed length requires 'length' or 'max_length'")

        processed_config["data"]["dataloader"]["max_length"] = (
            length // SAMPLING_RATE
        )

    return processed_config
