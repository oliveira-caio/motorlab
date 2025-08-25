import copy
import datetime
import pathlib
import random

import joblib
import numpy as np
import torch
import yaml

from motorlab import sessions


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def fix_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(filename: str) -> dict:
    """Load configuration from a YAML file."""
    configs_dir = pathlib.Path("configs")

    if not filename.endswith((".yml", ".yaml")):
        for ext in [".yml", ".yaml"]:
            candidate = configs_dir / (filename + ext)
            if candidate.exists():
                filename += ext
                break

    with open(configs_dir / filename, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


def save_config(d: dict) -> None:
    with open(pathlib.Path(d["logger"]["dir"]) / "config.yml", "w") as f:
        yaml.safe_dump(d, f)


def merge_dicts(d: dict, u: dict) -> dict:
    """Recursively merge two nested dictionaries.

    Parameters
    ----------
    d : dict
        The base dictionary whose values will be updated.
    u : dict
        The dictionary with new values.

    Returns
    -------
    dict
        A new dictionary containing the merged values.
    """
    result = copy.deepcopy(d)
    for k, v in u.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def parse_dotted_dict(d: dict) -> dict:
    parsed_d = {}
    for dotted_key, v in d.items():
        keys = dotted_key.split(".")
        _d = parsed_d
        for k in keys[:-1]:
            if k not in _d or not isinstance(_d[k], dict):
                _d[k] = {}
            _d = _d[k]
        _d[keys[-1]] = v
    return parsed_d


def preprocess_config(cfg: dict) -> dict:
    if "sessions" not in cfg:
        cfg["sessions"] = sessions.get(cfg["experiment"])

    default_cfg = {
        "seed": 0,
        "artifacts_dir": "artifacts/",
        "save": True,
        "data": {
            "dir": "data",
            "sampling_rate": 20,
            "dataset": {
                "stride": 1000,
                "concat_input": True,
                "concat_output": True,
            },
            "dataloader": {
                "min_length": 4000,
                "max_length": 4000,
                "batch_size": 64,
            },
            "intervals": {
                "include_trial": True,
                "include_homing": True,
                "include_sitting": True,
                "balance_intervals": False,
            },
            "modalities": {
                "location": {"representation": "com"},
                "poses": {
                    "representation": "egocentric",
                    "coordinates": "egocentric",
                    "skeleton_type": "normal",
                },
                "kinematics": {"representation": "com_vec"},
                "spikes": {"brain_areas": "all"},
                "spike_count": {"brain_areas": "all"},
            },
        },
        "model": {
            "embedding": {
                "architecture": "linear",
                "dim": 256,
            },
            # "core": {
            #     "architecture": "fc",
            #     "dims": 256,
            #     "n_layers": 3,
            # },
            "readout": {
                "map": "linear",
            },
        },
        "trainer": {
            "max_epochs": 1000,
            "validation_frequency": 25,
            "metrics": {},
            "optimizer": {
                "algorithm": "adam",
                "lr": 1e-2,
            },
            # "scheduler": {
            #     "method": "step_lr",
            #     "step_size": 1000,
            #     "gamma": 1.0,
            # },
            "early_stopper": {
                "enabled": True,
            },
        },
    }

    cfg = merge_dicts(default_cfg, cfg)

    if cfg["save"]:
        uid = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if "uid" in cfg:
            cfg["old_uid"] = cfg["uid"]
        cfg["uid"] = str(uid)

        logger_dir = (
            pathlib.Path(cfg["artifacts_dir"]) / cfg["task"] / cfg["uid"]
        )
        logger_dir.mkdir(parents=True, exist_ok=True)
        default_logger = {
            "dir": str(logger_dir),
        }
        cfg["logger"] = merge_dicts(default_logger, cfg.get("logger", {}))

    if cfg["model"]["architecture"] == "linear_regression":
        cfg["model"] = {"architecture": "linear_regression"}
    elif cfg["model"]["architecture"] == "core_readout":
        raise NotImplementedError(
            "utils.py: Core readout architecture is not implemented."
        )
    elif cfg["model"]["architecture"] == "embedding_core_readout":
        embedding_arch = cfg["model"]["embedding"]["architecture"]
        if isinstance(embedding_arch, str):
            cfg["model"]["embedding"]["architecture"] = {
                session: {
                    modality: embedding_arch
                    for modality in cfg["data"]["dataset"]["input_modalities"]
                }
                for session in cfg["sessions"]
            }
        elif isinstance(embedding_arch, dict):
            cfg["model"]["embedding"]["architecture"] = {
                session: {
                    modality: arch for modality, arch in embedding_arch.items()
                }
                for session in cfg["sessions"]
            }

        readout_map = cfg["model"]["readout"]["map"]
        if isinstance(readout_map, str):
            cfg["model"]["readout"]["map"] = {
                session: {
                    modality: readout_map
                    for modality in cfg["data"]["dataset"]["output_modalities"]
                }
                for session in cfg["sessions"]
            }
        elif isinstance(readout_map, dict):
            cfg["model"]["readout"]["map"] = {
                session: {
                    modality: arch for modality, arch in readout_map.items()
                }
                for session in cfg["sessions"]
            }

    return cfg


def load_checkpoint(dir_path: str | pathlib.Path):
    dir_path = pathlib.Path(dir_path)
    file = dir_path / "best_model.pt"
    return torch.load(file, weights_only=False)


def project_to_pc(a: np.ndarray, file: str | pathlib.Path) -> np.ndarray:
    pca = joblib.load(file)
    return pca.transform(a.copy())
