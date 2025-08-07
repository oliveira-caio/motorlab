import numpy as np
import torch
import wandb

from motorlab import logger, metrics, utils
from motorlab.model.factory import save_checkpoint as save_ckpt


def track(
    metrics: dict,
    model: torch.nn.Module,
    use_wandb: bool = False,
    save_checkpoint: bool = False,
    checkpoint_dir: str | None = None,
    uid: str | None = None,
    n_epochs: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    is_validation: bool = False,
    log_metrics: bool = True,
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
    """
    if is_validation:
        metrics = {
            f"val_{k}" if k != "epoch" else k: v for k, v in metrics.items()
        }

    if log_metrics:
        run_logger = logger.get()
        if run_logger.handlers:
            run_logger.info(format_metrics(metrics))

    if use_wandb:
        wandb.log({k: v for k, v in metrics.items() if k != "epoch"})

    if (
        save_checkpoint
        and checkpoint_dir is not None
        and uid is not None
        and n_epochs is not None
        and metrics["epoch"] != n_epochs
        and metrics["epoch"] % 25 == 0
        and metrics["epoch"] != 1
    ):
        checkpoint_path = f"{checkpoint_dir}/{uid}_{metrics['epoch']}.pt"
        save_ckpt(
            model=model,
            checkpoint_path=checkpoint_path,
            epoch=metrics["epoch"],
            optimizer=optimizer,
            scheduler=scheduler,
        )


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
    return "\t| ".join(reversed(formatted))


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
        - gts: Dict mapping sessions to the dictionary of modalities. Each modality dictionary maps the modality to the list of ground truth arrays per trial.
        - preds: Dict mapping sessions to the dictionary of modalities. Each modality dictionary maps the modality to the list of prediction arrays per trial.
    """
    model.eval()
    gts = {session: {} for session in dataloaders}
    preds = {session: {} for session in dataloaders}

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
                        modality_tensor.shape[1] // seq_length,
                        dim=1,
                    )
                ]

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
                        y_trial[modality_name].detach().cpu().numpy().squeeze(0)
                    )
                    preds[session][modality_name].append(
                        trial_pred[modality_name]
                        .detach()
                        .cpu()
                        .numpy()
                        .squeeze(0)
                    )

    stacked_gts = {
        session: {
            modality_name: np.concatenate(gts[session][modality_name], axis=0)
            for modality_name in gts[session]
        }
        for session in gts
    }
    stacked_preds = {
        session: {
            modality_name: np.concatenate(preds[session][modality_name], axis=0)
            for modality_name in preds[session]
        }
        for session in preds
    }

    track_metrics = {}
    if metric is not None:
        computed_metrics = metrics.compute(stacked_gts, stacked_preds, metric)
        track_metrics.update(computed_metrics)

    return track_metrics, gts, preds


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
    grad_norms = []

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

                loss = loss_fns[session](pred, y_trial)
                losses.append(loss.item())

                if is_train and optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.sqrt(
                        torch.tensor(
                            sum(
                                p.grad.norm().item() ** 2
                                for p in model.parameters()
                                if p.grad is not None
                            )
                        )
                    ).item()
                    grad_norms.append(grad_norm)
                    optimizer.step()

        stacked_gts = {
            session: {
                modality_name: np.concatenate(
                    gts[session][modality_name], axis=0
                )
                for modality_name in gts[session]
            }
            for session in gts
        }

        stacked_preds = {
            session: {
                modality_name: np.concatenate(
                    preds[session][modality_name], axis=0
                )
                for modality_name in preds[session]
            }
            for session in preds
        }

        if metric is not None:
            computed_metrics = metrics.compute(
                stacked_gts, stacked_preds, metric
            )
            track_metrics.update(computed_metrics)

        if losses:
            track_metrics["loss"] = np.mean(losses)

        if grad_norms:
            track_metrics["grad_norm"] = np.mean(grad_norms)

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
    use_wandb: bool = False,
    save_checkpoint: bool = False,
    checkpoint_dir: str | None = None,
    uid: str | None = None,
    validation_n_epochs: int = 25,
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
    metric : dict | None
        Metric configuration for evaluation
    n_epochs : int
        Number of training epochs
    print_stdout : bool, optional
        Whether to print metrics to stdout, by default True
    log_metrics : bool, optional
        Whether to log metrics to file, by default True
    use_wandb : bool, optional
        Whether to log to Weights & Biases, by default False
    save_checkpoint : bool, optional
        Whether to save checkpoints, by default False
    checkpoint_dir : str | None, optional
        Directory to save checkpoints, by default None
    uid : str | None, optional
        Unique identifier for saving, by default None
    validation_n_epochs : int, optional
        Run validation every N epochs, by default 25
    """
    valid_metrics, _, _ = iterate(
        model,
        valid_dataloaders,
        loss_fns=loss_fns,
        optimizer=None,
        metric=metric,
    )
    valid_metrics["epoch"] = 1
    track(
        metrics=valid_metrics,
        model=model,
        log_metrics=log_metrics,
        use_wandb=use_wandb,
        save_checkpoint=False,
        checkpoint_dir=checkpoint_dir,
        uid=uid,
        n_epochs=n_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        is_validation=True,
    )

    for i in range(n_epochs):
        display_epoch = i + 1

        train_metrics, _, _ = iterate(
            model,
            train_dataloaders,
            loss_fns,
            optimizer,
            metric,
            is_train=True,
        )

        scheduler.step()

        train_metrics["epoch"] = display_epoch
        track(
            metrics=train_metrics,
            model=model,
            log_metrics=log_metrics,
            use_wandb=use_wandb,
            save_checkpoint=save_checkpoint,
            checkpoint_dir=checkpoint_dir,
            uid=uid,
            n_epochs=n_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        if ((i + 1) % validation_n_epochs) == 0:
            valid_metrics, _, _ = iterate(
                model,
                valid_dataloaders,
                loss_fns=loss_fns,
                optimizer=None,
                metric=metric,
            )
            valid_metrics["epoch"] = display_epoch
            track(
                metrics=valid_metrics,
                model=model,
                log_metrics=log_metrics,
                use_wandb=use_wandb,
                save_checkpoint=False,
                checkpoint_dir=checkpoint_dir,
                uid=uid,
                n_epochs=n_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                is_validation=True,
            )

        if train_metrics["grad_norm"] < 1e-5 or np.isnan(train_metrics["loss"]):
            break
