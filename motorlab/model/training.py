import warnings

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import wandb

from torch.cuda.amp import GradScaler

from motorlab import logger, metrics, utils
from motorlab.model import distributed
from motorlab.model.factory import (
    save_checkpoint as save_ckpt,
    compute_gradient_norm,
)


def get_mixed_precision_context():
    """
    Get the appropriate mixed precision context and scaler based on device.

    Returns
    -------
    tuple
        (autocast_context, scaler, use_scaler)
        - autocast_context: Context manager for automatic mixed precision
        - scaler: GradScaler for CUDA or None for MPS/CPU
        - use_scaler: Boolean indicating whether to use gradient scaling
    """
    if utils.DEVICE.type == "cuda":
        # CUDA supports full AMP with GradScaler
        # For distributed training, each process gets its own scaler
        return (
            torch.autocast(device_type="cuda", dtype=torch.float16),
            GradScaler(),
            True,
        )
    elif utils.DEVICE.type == "mps":
        # MPS supports autocast but not GradScaler
        return (
            torch.autocast(device_type="cpu", dtype=torch.float16),
            None,
            False,
        )
    else:
        # CPU fallback - no mixed precision
        return nullcontext(), None, False


def format_metrics(metrics: dict[str, Any]) -> str:
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
    epoch_str = None
    loss_str = None
    grad_str = None
    lr_str = None
    other_metrics = []

    for key, value in metrics.items():
        if "epoch" in key:
            epoch_str = f"{key}: {value:04d}"
        elif "loss" in key:
            loss_str = (
                f"  {key}: {value:.4f}"
                if "val" in key
                else f"{key}: {value:.4f}"
            )
        elif "grad" in key:
            grad_str = f"{key}: {value:.5f}"
        elif "mse" in key or "corr" in key:
            other_metrics.append(
                f"  {key}: {value:.4f}"
                if "val" in key
                else f"{key}: {value:.4f}"
            )
        elif "accuracy" in key:
            other_metrics.append(
                f"  {key}: {value:.2f}"
                if "val" in key
                else f"{key}: {value:.2f}"
            )
        elif "lr" in key:
            lr_str = f"lr: {value:.5f}"
        else:
            other_metrics.append(
                f"  {key}: {value:.4f}"
                if "val" in key
                else f"{key}: {value:.4f}"
            )

    formatted.append(epoch_str)
    formatted.extend(other_metrics)
    formatted.append(loss_str)
    if lr_str is not None:
        formatted.append(lr_str)
    if grad_str is not None:
        formatted.append(grad_str)
    return " | ".join(formatted)


def track(
    metrics: dict,
    model: torch.nn.Module,
    uid: str,
    best_val_loss: dict,
    checkpoint_dir: str,
    use_wandb: bool = False,
    save_checkpoint: bool = False,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    is_validation: bool = False,
    log_metrics: bool = True,
    gradient_threshold: float = 0.5,
) -> None:
    """
    Track and log metrics during training or evaluation.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric names and values.
    model : torch.nn.Module
        Model being trained or evaluated.
    print_stdout : bool, optional
        Whether to enable logging, by default True
    log_metrics : bool, optional
        Whether to log metrics to file, by default True
    use_wandb : bool, optional
        Whether to log metrics to Weights & Biases, by default False
    save_checkpoint : bool, optional
        Whether to save intermediate checkpoints during training, by default True
    checkpoint_dir : Path | str | None, optional
        Directory to save checkpoints, by default None
    uid : str | None, optional
        Unique identifier for checkpoint filenames, by default None
    n_epochs : int | None, optional
        Total number of training epochs, by default None
    optimizer : torch.optim.Optimizer | None, optional
        Optimizer for checkpoint saving, by default None
    scheduler : torch.optim.lr_scheduler.LRScheduler | None, optional
        Scheduler for checkpoint saving, by default None
    is_validation : bool, optional
        Whether these are validation metrics (adds 'val_' prefix), by default False
    best_val_loss : dict
        Dictionary to track best validation loss for checkpointing, by default None
    gradient_threshold : float, optional
        Gradient norm threshold for checkpoint saving, by default 0.5
    """
    prefix = "val_" if is_validation else "train_"
    metrics = {
        f"{prefix}{k}" if k != "epoch" and k != "grad_norm" else k: v
        for k, v in metrics.items()
    }

    if log_metrics:
        run_logger = logger.get()
        if run_logger.handlers:
            run_logger.info(format_metrics(metrics))

    if use_wandb:
        wandb.log({k: v for k, v in metrics.items() if k != "epoch"})

    if (
        save_checkpoint
        and is_validation
        and metrics["val_loss"] < best_val_loss["value"]
        and metrics["grad_norm"] < gradient_threshold
        and distributed.is_main_process()  # Only save from rank 0
    ):
        checkpoint_path = f"{checkpoint_dir}/{uid}.pt"
        # Unwrap DDP model if needed
        model_to_save = model.module if hasattr(model, "module") else model
        save_ckpt(
            model=model_to_save,
            checkpoint_path=checkpoint_path,
            epoch=metrics["epoch"],
            optimizer=optimizer,
            scheduler=scheduler,
        )
        best_val_loss["value"] = metrics["val_loss"]
        best_val_loss["saved"] = True


def iterate_entire_trials(
    model: torch.nn.Module,
    dataloaders: dict,
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
    metric : dict | None, optional
        Metric dictionary to compute on predictions, by default None

    Returns
    -------
    tuple[dict, dict, dict]
        Tuple containing:
        - metrics: Dict of computed metric values
        - gts: Dict mapping sessions to the dictionary of modalities. Each modality dictionary maps the modality to the list of ground truth arrays per trial.
        - preds: Dict mapping sessions to the dictionary of modalities. Each modality dictionary maps the modality to the list of prediction arrays per trial.
    """
    model.eval()
    seq_len = 80
    gts = {session: {} for session in dataloaders}
    preds = {session: {} for session in dataloaders}
    autocast_context, _, _ = get_mixed_precision_context()

    with torch.no_grad():
        for session in dataloaders:
            for x_trial, y_trial in dataloaders[session]:
                x_trial = {
                    modality_name: modality_tensor.to(utils.DEVICE)
                    for modality_name, modality_tensor in x_trial.items()
                }

                x_seqs = [
                    {modality_name: seq}
                    for modality_name, modality_tensor in x_trial.items()
                    for seq in torch.tensor_split(
                        modality_tensor,
                        modality_tensor.shape[1] // seq_len,
                        dim=1,
                    )
                ]

                with autocast_context:
                    seq_preds = [model(seq, session) for seq in x_seqs]

                trial_pred = {
                    modality_name: torch.cat(
                        [pred[modality_name] for pred in seq_preds], dim=1
                    )
                    for modality_name in y_trial.keys()
                }

                for modality_name in y_trial.keys():
                    if modality_name not in gts[session]:
                        gts[session][modality_name] = []
                        preds[session][modality_name] = []

                    gts[session][modality_name].append(
                        y_trial[modality_name].squeeze(0)
                    )
                    preds[session][modality_name].append(
                        trial_pred[modality_name].squeeze(0)
                    )

    stacked_gts = {
        session: {
            modality_name: torch.cat(gts[session][modality_name], dim=0)
            .detach()
            .cpu()
            .numpy()
            for modality_name in gts[session]
        }
        for session in gts
    }
    stacked_preds = {
        session: {
            modality_name: torch.cat(preds[session][modality_name], dim=0)
            .detach()
            .cpu()
            .numpy()
            for modality_name in preds[session]
        }
        for session in preds
    }

    track_metrics = {}
    if metric is not None:
        computed_metrics = metrics.compute(stacked_gts, stacked_preds, metric)
        track_metrics.update(computed_metrics)

    gts_trials = {
        session: {
            modality_name: [
                trial.detach().cpu().numpy()
                for trial in gts[session][modality_name]
            ]
            for modality_name in gts[session]
        }
        for session in gts
    }
    preds_trials = {
        session: {
            modality_name: [
                trial.detach().cpu().numpy()
                for trial in preds[session][modality_name]
            ]
            for modality_name in preds[session]
        }
        for session in preds
    }

    return track_metrics, gts_trials, preds_trials


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
    autocast_context, scaler, use_scaler = get_mixed_precision_context()

    session_iterators = {
        session: iter(dataloader) for session, dataloader in dataloaders.items()
    }
    active_sessions = sorted(dataloaders.keys())

    while active_sessions:
        total_loss = 0
        sessions_to_remove = []

        for session in active_sessions:
            try:
                x_trial, y_trial = next(session_iterators[session])
                x_trial = {k: v.to(utils.DEVICE) for k, v in x_trial.items()}
                y_trial = {k: v.to(utils.DEVICE) for k, v in y_trial.items()}
                with autocast_context:
                    pred = model(x_trial, session)

                for modality in y_trial.keys():
                    if modality not in gts[session]:
                        gts[session][modality] = []
                        preds[session][modality] = []
                    gts[session][modality].append(y_trial[modality])
                    preds[session][modality].append(pred[modality])

                with autocast_context:
                    loss = loss_fns[session](pred, y_trial)

                if is_train and optimizer is not None:
                    if use_scaler and scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    loss.backward()

                total_loss += loss.item()

            except StopIteration:
                sessions_to_remove.append(session)

        if is_train and optimizer is not None:
            if use_scaler and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(active_sessions)
        losses.append(avg_loss)

        active_sessions = [
            session
            for session in active_sessions
            if session not in sessions_to_remove
        ]

    stacked_gts = {
        session: {
            modality_name: torch.cat(gts[session][modality_name], dim=0)
            .detach()
            .cpu()
            .numpy()
            for modality_name in gts[session]
        }
        for session in gts
    }

    stacked_preds = {
        session: {
            modality_name: torch.cat(preds[session][modality_name], dim=0)
            .detach()
            .cpu()
            .numpy()
            for modality_name in preds[session]
        }
        for session in preds
    }

    if metric is not None:
        computed_metrics = metrics.compute(stacked_gts, stacked_preds, metric)
        track_metrics.update(computed_metrics)

    track_metrics["loss"] = np.mean(losses)

    if not is_train:
        track_metrics["grad_norm"] = compute_gradient_norm(model)
        model.zero_grad()

    return track_metrics, stacked_gts, stacked_preds


def loop(
    model: torch.nn.Module,
    train_dataloaders: dict,
    valid_dataloaders: dict,
    optimizer: torch.optim.Optimizer,
    loss_fns: dict,
    metric: dict | None,
    uid: str,
    n_epochs: int,
    checkpoint_dir: str,
    validation_config: dict,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    early_stopping: Any | None = None,
    use_wandb: bool = False,
    save_checkpoint: bool = False,
    log_metrics: bool = True,
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
    early_stopping : EarlyStopping | None, optional
        Early stopping instance to halt training when validation stops improving, by default None
    validation_config : dict | None, optional
        Validation configuration with 'frequency' and 'gradient_threshold', by default None
    metric : dict | None
        Metric configuration for evaluation
    n_epochs : int
        Number of training epochs
    log_metrics : bool, optional
        Whether to log metrics to file, by default True
    use_wandb : bool, optional
        Whether to log to Weights & Biases, by default False
    save_checkpoint : bool, optional
        Whether to save checkpoints, by default False
    checkpoint_dir : str
        Directory to save checkpoints, by default None
    uid : str
        Unique identifier for saving, by default None
    """
    validation_frequency = validation_config["frequency"]
    gradient_threshold = validation_config["gradient_threshold"]

    valid_metrics, _, _ = iterate(
        model,
        valid_dataloaders,
        loss_fns=loss_fns,
        optimizer=None,
        metric=metric,
    )
    valid_metrics["epoch"] = 0
    best_val_loss = {"value": valid_metrics["loss"], "saved": False}
    track(
        metrics=valid_metrics,
        model=model,
        log_metrics=log_metrics,
        use_wandb=use_wandb,
        save_checkpoint=False,
        checkpoint_dir=checkpoint_dir,
        uid=uid,
        optimizer=optimizer,
        scheduler=scheduler,
        is_validation=True,
        best_val_loss=best_val_loss,
        gradient_threshold=gradient_threshold,
    )

    for i in range(1, n_epochs + 1):
        train_metrics, _, _ = iterate(
            model,
            train_dataloaders,
            loss_fns,
            optimizer,
            metric,
            is_train=True,
        )
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_metrics["loss"])
        elif scheduler is not None:
            scheduler.step()
            train_metrics["lr"] = scheduler.get_last_lr()[0]
        train_metrics["epoch"] = i
        track(
            metrics=train_metrics,
            model=model,
            log_metrics=log_metrics,
            use_wandb=use_wandb,
            save_checkpoint=save_checkpoint,
            checkpoint_dir=checkpoint_dir,
            uid=uid,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
            gradient_threshold=gradient_threshold,
        )

        if (i % validation_frequency) == 0:
            valid_metrics, _, _ = iterate(
                model,
                valid_dataloaders,
                loss_fns=loss_fns,
                optimizer=None,
                metric=metric,
            )
            valid_metrics["epoch"] = i
            track(
                metrics=valid_metrics,
                model=model,
                log_metrics=log_metrics,
                use_wandb=use_wandb,
                save_checkpoint=save_checkpoint,
                checkpoint_dir=checkpoint_dir,
                uid=uid,
                optimizer=optimizer,
                scheduler=scheduler,
                is_validation=True,
                best_val_loss=best_val_loss,
                gradient_threshold=gradient_threshold,
            )

            if early_stopping is not None and early_stopping.should_stop(
                valid_metrics["loss"], valid_metrics["grad_norm"]
            ):
                if log_metrics:
                    run_logger = logger.get()
                    if run_logger.handlers:
                        run_logger.info(
                            f"Early stopping triggered at epoch {i}"
                        )
                break

        if np.isnan(train_metrics["loss"]):
            break

    if save_checkpoint and not best_val_loss["saved"]:
        warnings.warn(
            "No checkpoint saved since validation loss never improved with gradient norm < "
            f"{gradient_threshold}",
            UserWarning,
        )
