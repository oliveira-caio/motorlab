"""
# this is a yaml file with all the possible config options. keep in mind that some of these options are redundant or conflict with each other or should be excluded depending on the task. examples: if you want to train from scratch, you should not provide an uid. if you want to train a regression, don't provide the number of classes.

DATA_DIR: data/gbyk

# if only CONFIG_DIR and CHECKPOINT_DIR are provided, i internally set load and save dir to be the same. however, we need to distinguish between loading and saving because sometimes we want to freeze the core and replace the readout to do another task.
CONFIG_LOAD_DIR: config/pose_to_spike_count
CONFIG_SAVE_DIR: config/pose_to_position
CHECKPOINT_LOAD_DIR: checkpoint/pose_to_spike_count
CHECKPOINT_SAVE_DIR: checkpoint/pose_to_position
CONFIG_DIR: config/pose_to_position
CHECKPOINT_DIR: checkpoint/pose_to_position

uid: "20250625113801" # if provided, loads this config and checkpoint
load_epoch: 25 # if provided, load the checkpoint at this epoch
freeze_core: True
save: True
experiment: gbyk
seed: 0
include_trial: True
include_homing: True
filter_sitting: False
in_modalities: poses
out_modalities: position
architecture: gru
sessions: ["bex_20230624_spikes_sorted_SES"]
position_repr: "com" # other option: tile (discrete)
body_repr: egocentric # other options: allocentric and centered
speed_repr: com_vec # other options: kps_vec, kps_mag and com_mag
loss_fn: crossentropy
metric: accuracy

model:
  - embedding_dim: 256
  - hidden_dim: 256
  - n_layers: 1
  - readout: "linear"
  - n_classes: 15 # exclude for regression problems

train:
  - n_epochs: 200
  - lr: 0.003
  - weight_decay: 0

track:
  - metrics: True
  - wandb: False
  - save_checkpoint: True
"""


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
