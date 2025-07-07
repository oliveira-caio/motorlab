def get_default_config(experiment, sessions):
    config = {
        "DATA_DIR": f"data/{experiment}",
        "CHECKPOINT_DIR": "checkpoint/pose_to_position",
        "CONFIG_DIR": "config/pose_to_position",
        "save": True,
        "experiment": experiment,
        "include_trial": True,
        "include_homing": False,
        "in_modalities": "poses",
        "out_modalities": "position",
        "architecture": "gru",
        "sessions": sessions,
        "position_repr": "com",
        "body_repr": "egocentric",
        "loss_fn": "mse",
        "metric": "mse",
        "uid": "",
        "model": {
            "embedding_dim": 256,
            "hidden_dim": 256,
            "n_layers": 1,
            "readout": "linear",
        },
        "train": {"n_epochs": 100, "lr": 5e-3},
        "track": {"metrics": True, "wandb": False, "save_checkpoint": True},
    }
    return config


gbyk_sessions = [
    "bex_20230621_spikes_sorted_SES",  # before
    "bex_20230624_spikes_sorted_SES",  # before
    "bex_20230629_spikes_sorted_SES",  # before
    "bex_20230630_spikes_sorted_SES",  # before
    "bex_20230701_spikes_sorted_SES",  # before
    "bex_20230708_spikes_sorted_SES",  # while
    # "ken_20230614_spikes_sorted_SES",  # while and before
    "ken_20230618_spikes_sorted_SES",  # before
    "ken_20230622_spikes_sorted_SES",  # while, before and free
    "ken_20230629_spikes_sorted_SES",  # while, before and free
    "ken_20230630_spikes_sorted_SES",  # while
    "ken_20230701_spikes_sorted_SES",  # before
    "ken_20230703_spikes_sorted_SES",  # while
]
