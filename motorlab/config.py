"""
keep in mind that some of these options are redundant or conflict with each other or should be excluded depending on the task. examples: if you want to train from scratch, you should not provide an uid. if you want to train a regression, don't provide the number of classes.

if only CONFIG_DIR and CHECKPOINT_DIR are provided, i internally set load and save dir to be the same. however, we need to distinguish between loading and saving because sometimes we want to freeze the core and replace the readout to do another task. use these options in such case: CONFIG_LOAD_DIR, CONFIG_SAVE_DIR, CHECKPOINT_LOAD_DIR, CHECKPOINT_SAVE_DIR.
"""


def get_default_config(experiment: str, sessions: list[str]) -> dict:
    """
    Get a default configuration dictionary for a given experiment and sessions.

    Parameters
    ----------
    experiment : str
        Name of the experiment.
    sessions : list of str
        List of session names.

    Returns
    -------
    dict
        Default configuration dictionary.
    """
    config = {
        "DATA_DIR": f"data/{experiment}",
        "CHECKPOINT_DIR": "checkpoint/poses_to_position",
        "CONFIG_DIR": "config/poses_to_position",
        # "CONFIG_LOAD_DIR": "config/pose_to_spike_count",
        # "CONFIG_SAVE_DIR": "config/pose_to_position",
        # "CHECKPOINT_LOAD_DIR": "checkpoint/pose_to_spike_count",
        # "CHECKPOINT_SAVE_DIR": "checkpoint/pose_to_position",
        "save": True,
        "experiment": experiment,
        "in_modalities": "poses",
        "out_modalities": "position",
        "sessions": sessions,
        "loss_fn": "mse",
        "metric": "mse",
        "seed": 0,
        # "load_epoch": 25, # loads the checkpoint at this epoch
        # "uid": "{number}", # loads the respective config and checkpoint
        # "freeze_core": True,  # loads the core of the model given by uid and freezes it.
        "intervals": {
            "include_trial": True,
            "include_homing": False,
            "include_sitting": True,
            "shuffle": True,
            "percent_split": 20,  # percent of data to use for test and validation (each gets 20%)
        },
        "position": {
            "representation": "com",  # other option: tile (discrete)
        },
        "poses": {
            "representation": "egocentric",  # other options: "allocentric" and "centered"
            # "keypoints_to_exclude": ["head"],
            # "project_to_pca": False,
            # "divide_variance": False,  # if True, divide each feature by its variance before fitting the PCA.
            # todo "pcs_to_exclude": {},
        },
        "kinematics": {
            "representation": "com_vec",  # other options: kps_vec, kps_mag and com_mag
        },
        "spikes": {
            "brain_area": "m1",  # other options: "m1", "pmd", "dlpfc".
        },
        "model": {
            "architecture": "fc",  # other options: "lstm", "gru", "transformer"
            "embedding_dim": 256,
            "hidden_dim": 256,
            "n_layers": 1,
            "readout": "linear",
            # "n_classes": 15,  # exclude for regression problems
            # "dropout": 0.1,
            # "bidirectional": True, # for RNNs
        },
        "train": {"n_epochs": 200, "lr": 5e-3},
        "track": {"metrics": True, "wandb": False, "save_checkpoint": True},
        "dataset": {"seq_length": 5 * 20, "stride": 2 * 20},
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


pg_sessions = [
    "bex_20230221",
    "bex_20230222",
    "bex_20230223",
    "bex_20230224",
    "bex_20230225",
    "bex_20230226",
    "jon_20230125",
    "jon_20230126",
    "jon_20230127",
    "jon_20230130",
    "jon_20230131",
    "jon_20230202",
    "jon_20230203",
    "luk_20230126",
    "luk_20230127",
    "luk_20230130",
    "luk_20230131",
    "luk_20230202",
    "luk_20230203",
]
