from collections.abc import Callable

import numpy as np
import torch
import wandb

from motorlab import metrics, utils
from motorlab.model.factory import save_checkpoint, compute_mean


def track(metrics, config, model):
    """
    Track and log metrics during training or evaluation.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric names and values.
    config : dict
        Configuration dictionary.
    model : torch.nn.Module
        Model being trained or evaluated.
    """
    if config["track"].get("metrics", False):
        print(format_metrics(metrics))
    if config["track"].get("wandb", False):
        wandb.log({k: v for k, v in metrics.items() if k != "epoch"})
    if (
        config["track"].get("save_checkpoint", False)
        and metrics["epoch"] != config["training"]["n_epochs"]
        and (metrics["epoch"] - 1) % 25 == 0
        and metrics["epoch"] != 1
    ):
        save_checkpoint(model, config, metrics["epoch"] - 1)


def format_metrics(metrics: dict) -> str:
    """
    Format metrics for printing/logging.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric names and values

    Returns
    -------
    str
        Formatted string of metrics
    """
    formatted = []
    for key, value in metrics.items():
        if "loss" in key or "mse" in key or "corr" in key:
            formatted.append(f"{key}: {value:.4f}")
        elif "accuracy" in key:
            formatted.append(f"{key}: {value:.2f}")
        elif "norm" in key:
            formatted.append(f"{key}: {value:.8f}")
        elif "epoch" in key:
            formatted.append(f"{key}: {value:04d}")
    return " | ".join(reversed(formatted))


def iterate_entire_trials(
    model: torch.nn.Module,
    dataloaders: dict,
    seq_length: int,
    metric: dict | None = None,
) -> tuple[dict, dict, dict]:
    """
    Run model on entire trials and collect predictions and ground truths.

    This function handles variable-length trials by splitting them into fixed-length
    sequences, running the model on each sequence, and concatenating the results.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate
    dataloaders : dict
        Dictionary mapping session names to DataLoader objects containing entire trials
    seq_length : int
        Fixed sequence length for splitting variable-length trials
    metric : dict | None, optional
        Metric dictionary to compute on predictions, by default None

    Returns
    -------
    tuple[dict, dict, dict]
        Tuple containing:
        - metrics: Dict of computed metric values
        - gts: Dict mapping sessions to ground truth arrays
        - preds: Dict mapping sessions to prediction arrays
    """
    model.eval()
    gts = {session: {} for session in dataloaders}
    preds = {session: {} for session in dataloaders}

    with torch.no_grad():
        for session in dataloaders:
            for x_trial, y_trial in dataloaders[session]:
                x_trial = {k: v.to(utils.DEVICE) for k, v in x_trial.items()}

                x_seqs = [
                    {modality: seq}
                    for modality, tensor in x_trial.items()
                    for seq in torch.tensor_split(
                        tensor, tensor.shape[1] // seq_length, dim=1
                    )
                ]

                seq_preds = [model(seq_input, session) for seq_input in x_seqs]

                trial_pred = {}
                for modality in y_trial.keys():
                    modality_preds = [pred[modality] for pred in seq_preds]
                    trial_pred[modality] = torch.cat(modality_preds, dim=1)

                for modality in y_trial.keys():
                    if modality not in gts[session]:
                        gts[session][modality] = []
                        preds[session][modality] = []

                    gts[session][modality].append(
                        y_trial[modality].detach().cpu().numpy()
                    )
                    preds[session][modality].append(
                        trial_pred[modality].detach().cpu().numpy()
                    )

    stacked_gts = {}
    stacked_preds = {}
    for session in gts:
        stacked_gts[session] = {}
        stacked_preds[session] = {}
        for modality in gts[session]:
            stacked_gts[session][modality] = np.concatenate(
                gts[session][modality], axis=1
            ).squeeze(0)
            stacked_preds[session][modality] = np.concatenate(
                preds[session][modality], axis=1
            ).squeeze(0)

    track_metrics = {}
    if metric is not None:
        computed_metrics = metrics.compute(stacked_gts, stacked_preds, metric)
        track_metrics.update(computed_metrics)

    return track_metrics, stacked_gts, stacked_preds


def iterate(
    model: torch.nn.Module,
    dataloaders: dict,
    loss_fns: dict,
    optimizer: torch.optim.Optimizer | None,
    metric: dict | None = None,
    is_train: bool = False,
) -> tuple[dict, dict, dict]:
    """
    Iterate through dataloaders for training or evaluation.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train or evaluate
    dataloaders : dict
        Dictionary mapping session names to DataLoader objects
    loss_fns : dict
        Dictionary mapping session names to loss functions
    optimizer : torch.optim.Optimizer | None
        Optimizer for training (None for evaluation)
    metric : dict | None, optional
        Metric dictionary to compute on predictions, by default None
    is_train : bool, optional
        Whether in training mode, by default False

    Returns
    -------
    tuple[dict, dict, dict]
        Tuple containing:
        - stacked_gts: Dict mapping sessions to concatenated ground truth arrays
        - stacked_preds: Dict mapping sessions to concatenated prediction arrays
    """
    model.train() if is_train else model.eval()
    gts = {session: {} for session in dataloaders}
    preds = {session: {} for session in dataloaders}
    track_metrics = dict()
    losses = []

    with torch.set_grad_enabled(is_train):
        for session in dataloaders:
            for x_trial, y_trial in dataloaders[session]:
                x_trial = {k: v.to(utils.DEVICE) for k, v in x_trial.items()}
                y_trial = {k: v.to(utils.DEVICE) for k, v in y_trial.items()}
                pred = model(x_trial, session)

                for modality in y_trial.keys():
                    if modality not in gts[session]:
                        gts[session][modality] = []
                        preds[session][modality] = []
                    gts[session][modality].append(
                        y_trial[modality].detach().cpu().numpy()
                    )
                    preds[session][modality].append(
                        pred[modality].detach().cpu().numpy()
                    )

                if is_train and optimizer is not None:
                    optimizer.zero_grad()
                    loss = loss_fns[session](pred, y_trial)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

        stacked_gts = {}
        stacked_preds = {}
        for session in gts:
            stacked_gts[session] = {}
            stacked_preds[session] = {}
            for modality in gts[session]:
                stacked_gts[session][modality] = np.concatenate(
                    gts[session][modality], axis=0
                )
                stacked_preds[session][modality] = np.concatenate(
                    preds[session][modality], axis=0
                )

        if metric is not None:
            computed_metrics = metrics.compute(
                stacked_gts, stacked_preds, metric
            )
            track_metrics.update(computed_metrics)

        if losses:
            track_metrics["loss"] = np.mean(losses)

        if is_train:
            track_metrics["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")
            ).item()
            track_metrics["mean_weight"] = compute_mean(model)

    return track_metrics, stacked_gts, stacked_preds


def loop(
    model: torch.nn.Module,
    train_dataloaders: dict,
    valid_dataloaders: dict,
    loss_fns: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metric: dict | None,
    n_epochs: int,
    track_fn: Callable | None = None,
    validation_n_epochs: int = 25,
) -> None:
    """
    Main training loop with validation.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    train_dataloaders : dict
        Training dataloaders for each session
    valid_dataloaders : dict
        Validation dataloaders for each session
    loss_fns : dict
        Loss functions for each session
    optimizer : torch.optim.Optimizer
        Optimizer for training
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Learning rate scheduler
    metric : dict | None
        Metric configuration for evaluation
    n_epochs : int
        Number of training epochs
    track_fn : callable | None, optional
        Function to call for tracking metrics (receives metrics dict), by default None
    validation_n_epochs : int, optional
        Run validation every N epochs, by default 25
    """
    valid_metrics, _, _ = iterate(
        model,
        valid_dataloaders,
        loss_fns,
        optimizer,
        metric,
    )
    valid_metrics["epoch"] = 1
    if track_fn is not None:
        track_fn(valid_metrics)

    for i in range(n_epochs):
        train_metrics, _, _ = iterate(
            model,
            train_dataloaders,
            loss_fns,
            optimizer,
            metric,
            is_train=True,
        )
        train_metrics["epoch"] = i + 1
        if track_fn is not None:
            track_fn(train_metrics)

        if ((i + 1) % validation_n_epochs) == 0:
            valid_metrics, _, _ = iterate(
                model,
                valid_dataloaders,
                loss_fns,
                optimizer,
                metric,
            )
            valid_metrics["epoch"] = i + 1
            if track_fn is not None:
                track_fn(valid_metrics)

        scheduler.step()
        if train_metrics["grad_norm"] < 1e-5 or np.isnan(train_metrics["loss"]):
            break
