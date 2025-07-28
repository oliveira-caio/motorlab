import os

from pathlib import Path

import numpy as np
import torch
import yaml

from motorlab import modules, utils


def create(
    architecture: str,
    sessions: list[str],
    readout_map: dict,
    model_dict: dict,
    is_train: bool = True,
) -> torch.nn.Module:
    """
    Create a model based on the specified architecture and parameters.

    Parameters
    ----------
    architecture : str
        Model architecture name ('gru', 'fc', 'linreg')
    sessions : list[str]
        List of session names
    readout_map : dict
        Mapping from modality to readout type
    model_dict : dict
        Model-specific parameters including dimensions, layers, etc.
    is_train : bool, optional
        Whether creating for training (affects whether model summary is printed), by default True

    Returns
    -------
    torch.nn.Module
        Created model on the specified device
    """
    if architecture == "gru":
        model = modules.GRUModel(
            sessions=sessions,
            in_dim=model_dict["in_dim"],
            embedding_dim=model_dict["embedding_dim"],
            hidden_dim=model_dict["hidden_dim"],
            out_dim=model_dict["out_dim"],
            n_layers=model_dict["n_layers"],
            readout_map=readout_map,
            dropout=model_dict.get("dropout", 0.0),
            bidirectional=model_dict.get("bidirectional", True),
        )
    elif architecture == "fc":
        model = modules.FCModel(
            sessions=sessions,
            in_dim=model_dict["in_dim"],
            embedding_dim=model_dict["embedding_dim"],
            hidden_dim=model_dict["hidden_dim"],
            out_dim=model_dict["out_dim"],
            n_layers=model_dict["n_layers"],
            readout_map=readout_map,
        )
    elif architecture == "linreg":
        model = modules.LinRegModel(
            sessions=sessions,
            in_dim=model_dict["in_dim"],
            out_dim=model_dict["out_dim"],
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model.to(utils.DEVICE)

    if is_train:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {architecture} | # params: {n_params:,}")

    return model


def load(
    architecture: str,
    sessions: list[str],
    readout_map: dict,
    model_dict: dict,
    uid: str,
    checkpoint_dir: Path | str,
    load_epoch: int | None = None,
    freeze: bool = False,
    is_train: bool = True,
) -> torch.nn.Module:
    """
    Load a model from checkpoint.

    Parameters
    ----------
    architecture : str
        Model architecture name
    sessions : list[str]
        List of session names
    readout_map : dict
        Mapping from modality to readout type
    model_dict : dict
        Model-specific parameters
    uid : str
        Unique identifier for the checkpoint
    checkpoint_dir : Path | str
        Directory containing checkpoints
    load_epoch : int | None, optional
        Specific epoch to load (None for latest), by default None
    freeze : bool, optional
        Whether to freeze core model parameters, by default False
    is_train : bool, optional
        Whether loading for training, by default True

    Returns
    -------
    torch.nn.Module
        Loaded and configured model on the specified device
    """
    # Create the model first
    model = create(architecture, sessions, readout_map, model_dict, is_train)

    # Load checkpoint
    ckpt_dir = Path(checkpoint_dir)
    if load_epoch is not None:
        checkpoint_path = ckpt_dir / f"{uid}_{load_epoch}.pt"
    else:
        checkpoint_path = ckpt_dir / f"{uid}.pt"

    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    if freeze:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if "in_layer" in k or "core" in k
        }
        model.load_state_dict(filtered_state_dict, strict=False)

        modules_to_freeze = ["embedding", "core"]
        for module_name in modules_to_freeze:
            module = getattr(model, module_name, None)
            if module is not None and hasattr(module, "parameters"):
                for param in module.parameters():
                    param.requires_grad = False

    return model


def save_config(config):
    """
    Save configuration dictionary to a YAML file.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    """
    config_dir = Path(
        config["config_save_dir"]
        if "config_save_dir" in config
        else config["config_dir"]
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{config['uid']}.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)


def save_checkpoint(model, config, epoch=None):
    """
    Save model checkpoint to disk.

    Parameters
    ----------
    model : torch.nn.Module
        Model to save.
    config : dict
        Configuration dictionary.
    epoch : int, optional
        Epoch number for checkpoint filename. Default is None.
    """
    checkpoint_dir = Path(
        config["checkpoint_save_dir"]
        if "checkpoint_save_dir" in config
        else config["checkpoint_dir"]
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{config['uid']}_{epoch}.pt" if epoch else f"{config['uid']}.pt"
    checkpoint_path = checkpoint_dir / filename
    torch.save(model.state_dict(), checkpoint_path)


def compute_mean(model: torch.nn.Module) -> float:
    """
    Compute the mean value of all trainable parameters (excluding biases) in a model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to compute mean parameter value for.

    Returns
    -------
    float
        Mean value of trainable parameters.
    """
    mean_val = torch.mean(
        torch.cat(
            [
                p.data.flatten()
                for p in model.parameters()
                if p.requires_grad and p.dim() > 1  # filters out biases
            ]
        )
    )
    return mean_val.item()


def dump_outputs(stacked_gts: dict, stacked_preds: dict, label: str) -> None:
    """
    Save model ground truths and predictions as .npy files for each session.

    Creates a 'dump/' directory and saves ground truth and prediction arrays
    for each session as separate numpy files with the specified label prefix.

    Parameters
    ----------
    stacked_gts : dict
        Dictionary mapping session names to ground truth numpy arrays
    stacked_preds : dict
        Dictionary mapping session names to prediction numpy arrays
    label : str
        Label prefix for saved files (e.g., 'train', 'test')
    """
    dump_dir = "dump/"
    os.makedirs(dump_dir, exist_ok=True)
    for session in stacked_gts:
        np.save(
            os.path.join(dump_dir, f"{label}_gts_{session}.npy"),
            stacked_gts[session],
        )
        np.save(
            os.path.join(dump_dir, f"{label}_preds_{session}.npy"),
            stacked_preds[session],
        )
