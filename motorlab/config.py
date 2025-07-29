"""
Comprehensive configuration system for motorlab experiments.

This file contains ALL possible configuration options used across the codebase.
Each option is documented with its purpose, possible values, and usage notes.

IMPORTANT NOTES:
- Some options are mutually exclusive or context-dependent
- If training from scratch, don't provide 'uid'
- For regression tasks, don't use 'n_classes'
- For transfer learning, use load/save directory separation
- String keys that can be dictionaries: 'loss_fn', 'metric', 'in_modalities', 'out_modalities'
"""


def preprocess(config: dict) -> dict:
    """
    Preprocess the user-friendly config into a format suitable for internal use.

    This function:
    - Converts string modalities to lists
    - Creates readout_map from readout if not provided
    - Converts string metrics to per-modality dictionaries
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

    if isinstance(processed_config["in_modalities"], str):
        processed_config["in_modalities"] = [processed_config["in_modalities"]]

    if isinstance(processed_config["out_modalities"], str):
        processed_config["out_modalities"] = [
            processed_config["out_modalities"]
        ]

    if "readout_map" not in processed_config["model"]:
        readout = processed_config["model"]["readout"]
        if isinstance(readout, dict):
            processed_config["model"]["readout_map"] = readout
        else:
            processed_config["model"]["readout_map"] = {
                modality: readout
                for modality in processed_config["out_modalities"]
            }

    if "metric" not in processed_config:
        processed_config["metric"] = None

    if isinstance(processed_config.get("metric"), str):
        metric_value = processed_config["metric"]
        processed_config["metric"] = {
            modality: metric_value
            for modality in processed_config["out_modalities"]
        }

    if isinstance(processed_config.get("loss_fn"), str):
        loss_fn_value = processed_config["loss_fn"]
        processed_config["loss_fn"] = {
            modality: loss_fn_value
            for modality in processed_config["out_modalities"]
        }

    return processed_config


def load_default(experiment: str, sessions: list[str]) -> dict:
    """
    Get a default configuration dictionary for a given experiment and sessions.

    Parameters
    ----------
    experiment : str
        Name of the experiment ('gbyk', 'pg', etc.)
    sessions : list of str
        List of session names

    Returns
    -------
    dict
        Default configuration dictionary.
    """
    config = {
        # =================================================================
        # CORE DATA & EXPERIMENT CONFIGURATION
        # =================================================================
        # Path of the data. This should be relative to the project root.
        "data_dir": f"data/{experiment}",
        # Experiment options: 'gbyk' and 'pg'
        "experiment": experiment,
        "sessions": sessions,
        "seed": 0,
        # The sampling rate I'll convert the data into.
        "sampling_rate": 20,
        # =================================================================
        # INPUT/OUTPUT MODALITIES
        # =================================================================
        # Can be string (single modality) or list (multiple modalities)
        # Options: "poses", "speed", "acceleration", "spike_count", "position"
        # Examples:
        #   Single: "poses" or "position"
        #   Multiple: ["poses", "spike_count"] for multi-modal input/output
        "in_modalities": "poses",
        "out_modalities": "position",
        # =================================================================
        # LOSS FUNCTION & METRICS
        # =================================================================
        # Can be string (single task) or dict (multi-task)
        # Options for single task: "mse", "crossentropy", "poisson"
        # Multi-task example: "loss_fn": {"spike_count": "poisson", "position": "mse"},
        "loss_fn": "mse",
        # Can be string (single metric) or dict (per-modality metrics)
        # Options: "mse", "accuracy", "correlation"
        # Multi-modal example: "metric": {"position": "correlation", "spikes": "mse"},
        "metric": "mse",
        # =================================================================
        # MODEL ARCHITECTURE
        # =================================================================
        "model": {
            # Options: "fc", "gru", "lstm", "linreg"
            "architecture": "fc",
            "embedding_dim": 256,
            "hidden_dim": 256,
            "n_layers": 1,
            # READOUT CONFIGURATION (can be string or dict)
            # Options: "linear", "softplus"
            # Multi-modal example:
            #   "readout_map": {"spike_count": "softplus", "position": "linear"},
            "readout": "linear",
            # OPTIONAL MODEL PARAMETERS
            # For classification (exclude for regression)
            # "n_classes": 15,
            # "dropout": 0.1,
            # For RNN architectures such as (gru, lstm)
            # "bidirectional": True,
        },
        # =================================================================
        # TRAINING CONFIGURATION
        # =================================================================
        "training": {
            # Maximum number of training epochs
            "n_epochs": 200,
            # Learning rate
            "lr": 5e-3,
        },
        # =================================================================
        # DATASET CONFIGURATION
        # =================================================================
        "dataset": {
            # Sequence length in timesteps (5 seconds @ 20Hz)
            "seq_length": 5 * 20,
            # Stride between sequences (2 seconds @ 20Hz)
            "stride": 2 * 20,
            "batch_size": 64,
            "entire_trials": False,
            # If multi-modality training, concatenate or not the modalities into a single tensor
            "concat_input": True,
            "concat_output": True,
        },
        # =================================================================
        # DATA INTERVALS & SPLITTING
        # =================================================================
        "intervals": {
            # Excluding trials can be useful for testing only during homing.
            "include_trial": True,
            "include_homing": True,
            # Excluding the sitting can be useful to focus only when the monkey is walking.
            "include_sitting": True,
            # Balance intervals removes the imbalance between sitting and walking, since there are usually way more frames of sitting.
            "balance_intervals": False,
        },
        # =================================================================
        # MODALITY-SPECIFIC CONFIGURATIONS
        # =================================================================
        "position": {
            # Options: "com" (for regression) and "tile" (for classification)
            "representation": "com",
        },
        "poses": {
            # Options: "egocentric", "allocentric", "centered", "trunk"
            "representation": "egocentric",
            # "keypoints_to_exclude": ["head"],
            # "project_to_pca": False,
            # "divide_variance": False,
            # PCA components to exclude per session
            # "pcs_to_exclude": {"bex_20230621_spikes_sorted_SES": [0, 1, 2], "bex_20230624_spikes_sorted_SES": [1, 2]},"},
        },
        "kinematics": {
            # Options: "com_vec", "kps_vec", "kps_mag", "com_mag"
            "representation": "com_vec",
        },
        "spikes": {
            # Options: "m1", "pmd", "dlpfc"
            "brain_area": "m1",
        },
        # =================================================================
        # CHECKPOINT & CONFIG MANAGEMENT
        # =================================================================
        # Whether to save model and config
        "save": True,
        # Directory for saving checkpoints
        "checkpoint_dir": "checkpoint/poses_to_position",
        # Directory for saving configs
        "config_dir": "config/poses_to_position",
        # Directory for logging
        "logging_dir": "logs/poses_to_position",
        # LOADING CONFIGURATIONS (for transfer learning or evaluation)
        # Unique identifier for loading specific model
        # "uid": "timestamp",
        # Load checkpoint from specific epoch
        # "load_epoch": 25,
        # "freeze": True,
        # SEPARATE LOAD/SAVE DIRECTORIES (for transfer learning)
        # Load config from different task
        # "config_load_dir": "config/source_task",
        # Save config to different location
        # "config_save_dir": "config/target_task",
        # Load model from different task
        # "checkpoint_load_dir": "checkpoint/source_task",
        # Save model to different location
        # "checkpoint_save_dir": "checkpoint/target_task",
        # =================================================================
        # TRACKING & LOGGING
        # =================================================================
        "track": {
            # Print metrics during training
            "metrics": True,
            # Log to Weights & Biases
            "wandb": False,
            # Save checkpoints during training
            "save_checkpoint": True,
        },
    }
    return config
