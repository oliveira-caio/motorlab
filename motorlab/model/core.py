import datetime
import pprint
import warnings

import torch
import wandb

from motorlab import logger, modules, utils
from motorlab.model.factory import (
    compute_dimensions,
    create,
    load_model,
    load_checkpoint_metadata,
)


class EarlyStopping:
    """
    Early stopping utility to halt training when validation loss stops improving.

    Parameters
    ----------
    patience : int
        Number of validation checks to wait for improvement before stopping
    min_delta : float
        Minimum change in validation loss to qualify as improvement
    gradient_threshold : float
        Only consider improvements when gradient norm is below this threshold
    """

    def __init__(
        self,
        patience: int = 6,
        min_delta: float = 0.0,
        gradient_threshold: float = 0.5,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.gradient_threshold = gradient_threshold
        self.best_loss = float("inf")
        self.patience_counter = 0

    def should_stop(self, val_loss: float, grad_norm: float) -> bool:
        """
        Check if training should stop based on validation loss and gradient norm.

        Parameters
        ----------
        val_loss : float
            Current validation loss
        grad_norm : float
            Current gradient norm

        Returns
        -------
        bool
            True if training should stop, False otherwise
        """
        # Only consider improvement if gradients are stable
        if grad_norm >= self.gradient_threshold:
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.patience


def create_optimizer(model, optimizer_config: dict) -> torch.optim.Optimizer:
    """
    Create an optimizer based on configuration.

    Parameters
    ----------
    optimizer_config : dict
        Optimizer configuration with 'type' and optimizer-specific parameters

    Returns
    -------
    torch.optim.Optimizer
        Configured optimizer instance
    """
    optimizer_type = optimizer_config["type"]

    if optimizer_type == "adam":
        return torch.optim.Adam(
            params=model.parameters(),
            lr=optimizer_config.get("lr", 1e-2),
            weight_decay=optimizer_config.get("weight_decay", 0),
        )
    elif optimizer_type == "sgd":
        return torch.optim.SGD(
            params=model.parameters(),
            lr=optimizer_config.get("lr", 1e-2),
            weight_decay=optimizer_config.get("weight_decay", 0.0),
            momentum=optimizer_config.get("momentum", 0.0),
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer, scheduler_config: dict
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """
    Create a learning rate scheduler based on configuration.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to attach the scheduler to
    scheduler_config : dict
        Scheduler configuration with 'type' and scheduler-specific parameters

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler
        Configured scheduler instance

    Raises
    ------
    ValueError
        If scheduler type is not recognized
    """
    scheduler_type = scheduler_config["type"]

    if scheduler_type == "step_lr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 25),
            gamma=scheduler_config.get("gamma", 0.4),
        )
    elif scheduler_type == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get("T_max", 100),
            eta_min=scheduler_config.get("eta_min", 1e-4),
        )
    elif scheduler_type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config.get("factor", 0.1),
            patience=scheduler_config.get("patience", 10),
            min_lr=scheduler_config.get("min_lr", 1e-5),
        )
    else:
        warnings.warn("It will use a constant learning rate.")
        return None


def create_early_stopping(early_stopping_config: dict) -> EarlyStopping | None:
    """
    Create an early stopping instance based on configuration.

    Parameters
    ----------
    early_stopping_config : dict
        Early stopping configuration with 'enabled' and optional parameters

    Returns
    -------
    EarlyStopping | None
        Configured early stopping instance or None if disabled
    """
    if not early_stopping_config.get("enabled", False):
        return None

    return EarlyStopping(
        patience=early_stopping_config.get("patience", 6),
        min_delta=early_stopping_config.get("min_delta", 0.0),
        gradient_threshold=early_stopping_config.get("gradient_threshold", 0.5),
    )


def setup(config: dict, train_intervals: dict, is_train: bool) -> dict:
    """
    Set up model, data, loss functions, and optionally optimizer/scheduler for training or evaluation.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model and data parameters
    train_intervals : dict
        Training intervals for each session
    is_train : bool
        Whether setting up for training (affects model configuration)

    Returns
    -------
    dict
        Dictionary containing model, data_dict, loss_fns, and optionally optimizer/scheduler
    """
    utils.fix_seed(config.get("seed", 0))

    data_dict = utils.load_all_data(
        data_dir=config["data_dir"],
        sessions=config["sessions"],
        in_modalities=config["in_modalities"],
        out_modalities=config["out_modalities"],
        experiment=config.get("experiment", "gbyk"),
        poses_config=config.get("poses", dict()),
        location_config=config.get("position", dict()),
        kinematics_config=config.get("kinematics", dict()),
        spikes_config=config.get("spikes", dict()),
        train_intervals=train_intervals,
    )

    model_dict = config["model"].copy()

    if is_train:
        in_dim, out_dim = compute_dimensions(
            data_dict=data_dict,
            in_modalities=config["in_modalities"],
            out_modalities=config["out_modalities"],
            sessions=config["sessions"],
            concat_input=config["dataset"].get("concat_input", True),
            concat_output=config["dataset"].get("concat_output", True),
            n_classes=config["model"].get("n_classes", None),
        )

        model_dict["in_dim"] = in_dim
        model_dict["out_dim"] = out_dim

    uid = config.get("uid", None)

    if uid is not None:
        if not model_dict.get("in_dim") or not model_dict.get("out_dim"):
            in_dim, out_dim = compute_dimensions(
                data_dict=data_dict,
                in_modalities=config["in_modalities"],
                out_modalities=config["out_modalities"],
                sessions=config["sessions"],
                concat_input=config["dataset"].get("concat_input", True),
                concat_output=config["dataset"].get("concat_output", True),
                n_classes=config["model"].get("n_classes", None),
            )
            model_dict["in_dim"] = in_dim
            model_dict["out_dim"] = out_dim

        checkpoint_data = load_checkpoint_metadata(
            uid=uid,
            checkpoint_dir=config.get(
                "checkpoint_load_dir", config["checkpoint_dir"]
            ),
            load_epoch=config.get("load_epoch", None),
        )

        model = load_model(
            architecture=config["model"]["architecture"],
            sessions=config["sessions"],
            readout_map=config["model"]["readout_map"],
            model_dict=model_dict,
            checkpoint_data=checkpoint_data,
            freeze=config.get("freeze_core", False),
            is_train=is_train,
        )

    else:
        model = create(
            architecture=config["model"]["architecture"],
            sessions=config["sessions"],
            model_dict=model_dict,
        )
        checkpoint_data = None

    loss_fns = {
        session: modules.DictLoss(config["loss_fn"])
        for session in config["sessions"]
    }

    result = {
        "model": model,
        "data_dict": data_dict,
        "loss_fns": loss_fns,
    }

    if is_train:
        optimizer = create_optimizer(model, config["training"]["optimizer"])
        scheduler = create_scheduler(optimizer, config["training"]["scheduler"])
        early_stopping = create_early_stopping(
            config["training"]["early_stopping"]
        )

        if checkpoint_data is not None:
            if "optimizer_state_dict" in checkpoint_data:
                optimizer.load_state_dict(
                    checkpoint_data["optimizer_state_dict"]
                )
            if scheduler and "scheduler_state_dict" in checkpoint_data:
                scheduler.load_state_dict(
                    checkpoint_data["scheduler_state_dict"]
                )
            if "random_state" in checkpoint_data:
                torch.set_rng_state(checkpoint_data["random_state"])

        result["optimizer"] = optimizer
        result["scheduler"] = scheduler
        result["early_stopping"] = early_stopping
        result["checkpoint_data"] = checkpoint_data

        if uid is None:
            uid = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            if "uid" in config:
                config["old_uid"] = config["uid"]
            config["uid"] = str(uid)

        if config["track"].get("logging", False):
            uid_msg = f"uid: {config.get('uid', 'None')}"
            config_msg = f"{pprint.pformat(config, indent=2)}"
            model_msg = f"{str(model)}"
            n_params_msg = f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"

            run_logger = logger.create(
                log_dir=config["log_dir"],
                uid=uid,
                console_output=config["track"].get("stdout", True),
            )
            run_logger.info(uid_msg)
            run_logger.info(config_msg)
            run_logger.info(model_msg)
            run_logger.info(n_params_msg)

    if config["track"].get("wandb", False):
        locked_keys = set(wandb.config.keys()) if wandb.run else set()
        config_for_wandb = {
            k: v for k, v in config.items() if k not in locked_keys
        }
        wandb.config.update(config_for_wandb)

    return result
