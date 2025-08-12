import inspect
import os

from pathlib import Path

import numpy as np
import torch

from torch.nn.parallel import DistributedDataParallel as DDP

from motorlab import modules, utils
from motorlab.model import distributed


MODEL_REGISTRY = {
    "gru": modules.GRUModel,
    "fc": modules.FCModel,
    "linreg": modules.LinRegModel,
}


def register_model(name: str, model_class):
    """
    Register a new model architecture in the factory.

    Parameters
    ----------
    name : str
        Name identifier for the model architecture
    model_class : class
        Model class to register
    """
    MODEL_REGISTRY[name] = model_class


def get_available_architectures() -> list[str]:
    """
    Get list of available model architectures.

    Returns
    -------
    list[str]
        List of registered architecture names
    """
    return list(MODEL_REGISTRY.keys())


def compile_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Attempt to compile the model using torch.compile.

    Parameters
    ----------
    model : torch.nn.Module
        Model to compile

    Returns
    -------
    torch.nn.Module
        Compiled model if successful, original model if compilation fails
    """
    if utils.DEVICE.type == "cuda":
        compiled_model = torch.compile(model, mode="max-autotune")
        return compiled_model
    else:
        return model


def prepare_model_for_distributed(model: torch.nn.Module) -> torch.nn.Module:
    """
    Prepare a model for distributed training if multiple GPUs are available.

    This function:
    1. Sets up distributed training if needed
    2. Compiles the model
    3. Wraps with DistributedDataParallel if appropriate

    Parameters
    ----------
    model : torch.nn.Module
        Model to prepare for distributed training

    Returns
    -------
    torch.nn.Module
        Model ready for distributed training (wrapped in DDP if needed)
    """
    if distributed.is_available() and not distributed.is_initialized():
        if "RANK" in os.environ:
            distributed.setup_distributed()
        else:
            # Single machine, multiple GPUs available but not launched with torchrun
            if distributed.is_main_process():
                from motorlab import logger

                run_logger = logger.get()
                if run_logger.handlers:
                    run_logger.info(
                        f"Multiple GPUs detected ({torch.cuda.device_count()})"
                    )
                    run_logger.info(
                        "For multi-GPU training, launch with: torchrun --nproc_per_node={} script.py".format(
                            torch.cuda.device_count()
                        )
                    )

    # Compile the model first (before DDP wrapping)
    model = compile_model(model)

    # Wrap with DDP if distributed training is active
    if distributed.is_initialized():
        model = DDP(model, device_ids=[torch.cuda.current_device()])

        if distributed.is_main_process():
            from motorlab import logger

            run_logger = logger.get()
            if run_logger.handlers:
                run_logger.info(
                    f"Model wrapped with DistributedDataParallel on {distributed.get_world_size()} GPUs"
                )

    return model


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """
    Compute the L2 norm of gradients for all parameters in the model.

    Parameters
    ----------
    model : torch.nn.Module
        Model with computed gradients

    Returns
    -------
    float
        L2 norm of all gradients
    """
    gradients = torch.cat(
        [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    )
    total_norm = torch.linalg.norm(gradients, ord=2).item()
    return total_norm


def load_checkpoint_metadata(
    uid: str,
    checkpoint_dir: Path | str,
    load_epoch: int | None = None,
) -> dict:
    """
    Load checkpoint metadata and full checkpoint data.

    Parameters
    ----------
    uid : str
        Unique identifier for the checkpoint
    checkpoint_dir : Path | str
        Directory containing checkpoints
    load_epoch : int | None, optional
        Specific epoch to load (None for latest), by default None

    Returns
    -------
    dict
        Complete checkpoint dictionary containing model_state_dict,
        optimizer_state_dict, scheduler_state_dict, epoch, etc.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if load_epoch is not None:
        checkpoint_path = checkpoint_dir / f"{uid}_{load_epoch}.pt"
    else:
        checkpoint_path = checkpoint_dir / f"{uid}.pt"
    return torch.load(checkpoint_path, map_location="cpu")


def create(
    architecture: str,
    sessions: list[str],
    model_dict: dict,
) -> torch.nn.Module:
    """
    Create a model based on the specified architecture and parameters.

    Uses automatic parameter inspection to filter model_dict to only include
    parameters that the target model class actually accepts.

    Parameters
    ----------
    architecture : str
        Model architecture name (see get_available_architectures())
    sessions : list[str]
        List of session names
    model_dict : dict
        Model-specific parameters including dimensions, layers, etc.

    Returns
    -------
    torch.nn.Module
        Created model on the specified device

    Raises
    ------
    ValueError
        If architecture is not registered in MODEL_REGISTRY
    """
    if architecture not in MODEL_REGISTRY:
        available = ", ".join(get_available_architectures())
        raise ValueError(
            f"Unknown architecture: '{architecture}'. Available: {available}"
        )

    model_class = MODEL_REGISTRY[architecture]

    # Get valid parameters from the class constructor
    sig = inspect.signature(model_class.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}

    # Filter model_dict to only include valid parameters
    kwargs = {k: v for k, v in model_dict.items() if k in valid_params}

    model = model_class(sessions=sessions, **kwargs)
    model.to(utils.DEVICE)
    model = prepare_model_for_distributed(model)
    return model


def load_model(
    architecture: str,
    sessions: list[str],
    model_dict: dict,
    checkpoint_data: dict,
    freeze: bool = False,
    is_train: bool = True,
) -> torch.nn.Module:
    """
    Load a model from checkpoint data.

    Parameters
    ----------
    architecture : str
        Model architecture name
    sessions : list[str]
        List of session names
    model_dict : dict
        Model-specific parameters
    checkpoint_data : dict
        Checkpoint dictionary containing model_state_dict
    freeze : bool, optional
        Whether to freeze core model parameters, by default False
    is_train : bool, optional
        Whether loading for training, by default True

    Returns
    -------
    torch.nn.Module
        Loaded model on the specified device
    """
    model = create(architecture, sessions, model_dict)
    model.load_state_dict(checkpoint_data["model_state_dict"])

    if freeze:
        filtered_state_dict = {
            k: v
            for k, v in checkpoint_data["model_state_dict"].items()
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


def save_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path | str,
    epoch: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    **additional_state,
) -> None:
    """
    Save model checkpoint to disk with complete training state.

    Parameters
    ----------
    model : torch.nn.Module
        Model to save.
    checkpoint_path : Path | str
        Full path where the checkpoint will be saved.
    epoch : int, optional
        Current epoch number. Default is None.
    optimizer : torch.optim.Optimizer, optional
        Optimizer to save state from. Default is None.
    scheduler : torch.optim.lr_scheduler.LRScheduler, optional
        Learning rate scheduler to save state from. Default is None.
    **additional_state
        Additional state to save in the checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "random_state": torch.get_rng_state(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    checkpoint.update(additional_state)
    torch.save(checkpoint, checkpoint_path)


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


def compute_dimensions(
    data_dict: dict,
    in_modalities: list[str],
    out_modalities: list[str],
    sessions: list[str],
    concat_input: bool = True,
    concat_output: bool = True,
    n_classes: int | None = None,
) -> tuple[dict, dict]:
    """
    Compute input and output dimensions for model creation from data.

    Parameters
    ----------
    data_dict : dict
        Dictionary of loaded data with structure {session: {modality: array}}
    in_modalities : list[str]
        List of input modality names
    out_modalities : list[str]
        List of output modality names
    sessions : list[str]
        List of session names
    concat_input : bool, optional
        Whether to concatenate input modalities, by default True
    concat_output : bool, optional
        Whether to concatenate output modalities, by default True
    n_classes : int | None, optional
        Number of classes for classification (None for regression), by default None

    Returns
    -------
    tuple[dict, dict]
        Tuple of (in_dim, out_dim) dictionaries with structure {session: {modality: dim}}

    Examples
    --------
    >>> in_dim, out_dim = compute_model_dimensions(
    ...     data_dict, ['poses'], ['position'], ['session1'],
    ...     concat_input=True, concat_output=True
    ... )
    >>> # Returns: ({'session1': {'poses': 63}}, {'session1': {'position': 2}})
    """
    # Calculate in_dim based on concatenation settings
    if concat_input:
        # Concatenated input: single key with joined modality names
        input_key = "_".join(in_modalities)
        in_dim = {
            session: {
                input_key: sum(
                    data_dict[session][m].shape[1] for m in in_modalities
                )
            }
            for session in sessions
        }
    else:
        # Non-concatenated input: separate keys for each modality
        in_dim = {
            session: {
                modality: data_dict[session][modality].shape[1]
                for modality in in_modalities
            }
            for session in sessions
        }

    # Calculate out_dim based on concatenation settings
    if n_classes is not None:
        # For classification, use n_classes for each output modality
        if concat_output:
            output_key = "_".join(out_modalities)
            out_dim = {session: {output_key: n_classes} for session in sessions}
        else:
            out_dim = {
                session: {modality: n_classes for modality in out_modalities}
                for session in sessions
            }
    else:
        # For regression, use actual data dimensions
        if concat_output:
            output_key = "_".join(out_modalities)
            out_dim = {
                session: {
                    output_key: sum(
                        data_dict[session][m].shape[1] for m in out_modalities
                    )
                }
                for session in sessions
            }
        else:
            out_dim = {
                session: {
                    modality: data_dict[session][modality].shape[1]
                    for modality in out_modalities
                }
                for session in sessions
            }

    return in_dim, out_dim
