import torch


class EarlyStopper:
    """
    Early stopping utility to halt training when validation loss stops improving.

    Parameters
    ----------
    patience : int
        Number of validation checks to wait for improvement before stopping
    min_delta : float
        Minimum change in validation loss to qualify as improvement
    """

    def __init__(
        self,
        metric: str,
        delta: float = 0.009,
        patience: int = 6,
        minimize: bool = True,
    ):
        self.best_val = float("inf") if minimize else float("-inf")
        self.delta = delta
        self.minimize = minimize
        self.metric = metric
        self.patience = patience
        self.patience_counter = 0

    def stop(self, curr_val: float) -> bool:
        if self.minimize:
            if curr_val < (self.best_val - self.delta):
                self.best_val = curr_val
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        else:
            if curr_val > (self.best_val + self.delta):
                self.best_val = curr_val
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        return self.patience_counter >= self.patience

    def load_state_dict(self, state_dict: dict) -> None:
        self.best_val = state_dict["best_val"]
        self.delta = state_dict["delta"]
        self.minimize = state_dict["minimize"]
        self.metric = state_dict["metric"]
        self.patience = state_dict["patience"]
        self.patience_counter = state_dict["patience_counter"]

    def state_dict(self) -> dict:
        return {
            "best_val": self.best_val,
            "delta": self.delta,
            "metric": self.metric,
            "minimize": self.minimize,
            "patience": self.patience,
            "patience_counter": self.patience_counter,
        }


def create_early_stopper(cfg: dict) -> EarlyStopper | None:
    """
    Create an early stopper instance based on configuration.

    Parameters
    ----------
    early_stopper_config : dict
        Early stopper configuration with 'enabled' and optional parameters

    Returns
    -------
    EarlyStopping | None
        Configured early stopping instance or None if disabled
    """
    if not cfg.get("enabled", False):
        return None

    return EarlyStopper(
        delta=cfg.get("delta", 0.0),
        metric=cfg.get("metric", "val_loss"),
        minimize=cfg.get("minimize", True),
        patience=cfg.get("patience", 6),
    )


def create_optimizer(parameters, cfg) -> torch.optim.Optimizer:
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
    optimizer_kwargs = {k: v for k, v in cfg.items() if k != "algorithm"}
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }
    optimizer = optimizers[cfg["algorithm"]](parameters, **optimizer_kwargs)
    optimizer.zero_grad()
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
) -> torch.optim.lr_scheduler.LRScheduler:
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
    scheduler_kwargs = {k: v for k, v in cfg.items() if k != "method"}
    schedulers = {
        "step_lr": torch.optim.lr_scheduler.StepLR,
        "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR,
        "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }
    scheduler = schedulers[cfg["method"]](optimizer, **scheduler_kwargs)
    return scheduler
