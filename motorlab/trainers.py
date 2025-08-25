import pathlib
import pickle
import warnings

import numpy as np
import torch
import wandb

from motorlab import metrics, modules, plots


def log_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    early_stopper: modules.EarlyStopper | None,
    run_dir: pathlib.Path | str,
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
    run_dir = pathlib.Path(run_dir)
    checkpoint_path = run_dir / "best_model.pt"
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": torch.get_rng_state(),
        "early_stopper": None
        if early_stopper is None
        else early_stopper.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    artifact = wandb.Artifact("model-checkpoint", type="model")
    artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(artifact)


@torch.inference_mode()
def iterate_test(
    model: torch.nn.Module,
    dataloaders: dict,
    loss_fn: torch.nn.Module,
    logger: dict,
    metrics_dict: dict = dict(),
    plots_cfg: dict = dict(),
) -> dict[str, float]:
    model.eval()

    inputs = {session: [] for session in dataloaders}
    gts = {session: [] for session in dataloaders}
    preds = {session: [] for session in dataloaders}

    loss_sessions = []

    for session in dataloaders:
        length = dataloaders[session].max_length
        loss_intervals = 0.0
        intervals_count = 0
        for x_interval, y_interval in dataloaders[session]:
            # ! interval must have shape (1, length, n_features)
            x_splits = {
                modality: torch.tensor_split(
                    interval, interval.shape[1] // length, dim=1
                )
                for modality, interval in x_interval.items()
            }
            x_seqs = [
                {modality: seq[i] for modality, seq in x_splits.items()}
                for i in range(len(next(iter(x_splits.values()))))
            ]

            preds_seqs = [model(seq, session) for seq in x_seqs]
            pred_interval = {
                modality: torch.cat(
                    [pred[modality] for pred in preds_seqs], dim=1
                )
                for modality in y_interval.keys()
            }

            inputs[session].append(x_interval)
            gts[session].append(y_interval)
            preds[session].append(pred_interval)

            loss_intervals += loss_fn(pred_interval, y_interval).item()
            intervals_count += 1

        loss_sessions.append(loss_intervals / intervals_count)

    inputs_intervals = {
        session: {
            modality: [
                interval[modality].detach().cpu().numpy().squeeze(0)
                for interval in intervals
            ]
            for modality in intervals[0]
        }
        for session, intervals in inputs.items()
    }

    gts_intervals = {
        session: {
            modality: [
                interval[modality].detach().cpu().numpy().squeeze(0)
                for interval in intervals
            ]
            for modality in intervals[0]
        }
        for session, intervals in gts.items()
    }

    preds_intervals = {
        session: {
            modality: [
                interval[modality].detach().cpu().numpy().squeeze(0)
                for interval in intervals
            ]
            for modality in intervals[0]
        }
        for session, intervals in preds.items()
    }

    gts_stacked = {
        session: {
            modality: np.concatenate(gts_intervals[session][modality], axis=0)
            for modality in gts_intervals[session]
        }
        for session in gts_intervals.keys()
    }

    preds_stacked = {
        session: {
            modality: np.concatenate(preds_intervals[session][modality], axis=0)
            for modality in preds_intervals[session]
        }
        for session in preds_intervals
    }

    results = metrics.compute(gts_stacked, preds_stacked, metrics_dict)
    test_results = {"test_" + k: v for k, v in results.items()}
    test_results["test_loss"] = np.mean(loss_sessions).item()

    if logger.get("plots", True):
        input_figs = plots.make_inputs_fig(
            inputs=inputs_intervals,
            cfg=plots_cfg,
        )
        output_figs = plots.make_outputs_fig(
            gts=gts_intervals,
            preds=preds_intervals,
            cfg=plots_cfg,
        )
        for fig in input_figs + output_figs:
            wandb.log({"test_figs": wandb.Image(fig)})

    if logger.get("predictions", True):
        output = {"gts": gts_intervals, "preds": preds_intervals}
        output_path = pathlib.Path(logger["dir"]) / "predictions.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(output, f)
        artifact = wandb.Artifact("predictions", type="dataset")
        artifact.add_file(str(output_path))
        wandb.log_artifact(artifact)

    return test_results


def iterate_val(
    model: torch.nn.Module,
    dataloaders: dict,
    loss_fn: torch.nn.Module,
    logger: dict,
    epoch: int,
    metrics_dict: dict = dict(),
    plots_cfg: dict = dict(),
) -> dict[str, float]:
    model.eval()

    inputs = {session: [] for session in dataloaders}
    gts = {session: [] for session in dataloaders}
    preds = {session: [] for session in dataloaders}

    loss_sessions = []

    with torch.no_grad():
        for session in dataloaders:
            length = dataloaders[session].max_length
            losses = []
            for x_interval, y_interval in dataloaders[session]:
                # ! interval must have shape (1, interval_length, n_features)
                x_splits = {
                    modality: torch.tensor_split(
                        interval, 1 + interval.shape[1] // length, dim=1
                    )
                    for modality, interval in x_interval.items()
                }
                x_seqs = [
                    {modality: seq[i] for modality, seq in x_splits.items()}
                    for i in range(len(next(iter(x_splits.values()))))
                ]

                preds_seqs = [model(seq, session) for seq in x_seqs]
                pred_interval = {
                    modality: torch.cat(
                        [pred[modality] for pred in preds_seqs], dim=1
                    )
                    for modality in y_interval.keys()
                }

                inputs[session].append(x_interval)
                gts[session].append(y_interval)
                preds[session].append(pred_interval)
                losses.append(loss_fn(pred_interval, y_interval).item())

            loss_sessions.append(np.mean(losses).item())

    inputs_intervals = {
        session: {
            modality: [
                interval[modality].detach().cpu().numpy().squeeze(0)
                for interval in intervals
            ]
            for modality in intervals[0]
        }
        for session, intervals in inputs.items()
    }

    gts_intervals = {
        session: {
            modality: [
                interval[modality].detach().cpu().numpy().squeeze(0)
                for interval in intervals
            ]
            for modality in intervals[0]
        }
        for session, intervals in gts.items()
    }

    preds_intervals = {
        session: {
            modality: [
                interval[modality].detach().cpu().numpy().squeeze(0)
                for interval in intervals
            ]
            for modality in intervals[0]
        }
        for session, intervals in preds.items()
    }

    gts_stacked = {
        session: {
            modality: np.concatenate(gts_intervals[session][modality], axis=0)
            for modality in gts_intervals[session]
        }
        for session in gts_intervals.keys()
    }

    preds_stacked = {
        session: {
            modality: np.concatenate(preds_intervals[session][modality], axis=0)
            for modality in preds_intervals[session]
        }
        for session in preds_intervals
    }

    results = metrics.compute(gts_stacked, preds_stacked, metrics_dict)
    val_results = {"val_" + k: v for k, v in results.items()}
    val_results.update(
        {
            "val_loss": np.mean(loss_sessions).item(),
            "weights_norm": torch.linalg.norm(
                torch.cat([p.data.flatten() for p in model.parameters()]), ord=2
            ).item(),
        }
    )

    if logger.get("plots", True):
        input_figs = (
            []
            if epoch != 0
            else plots.make_inputs_fig(
                inputs=inputs_intervals,
                cfg=plots_cfg,
            )
        )
        output_figs = plots.make_outputs_fig(
            gts=gts_intervals,
            preds=preds_intervals,
            cfg=plots_cfg,
        )
        for fig in input_figs + output_figs:
            wandb.log({"val_figs": wandb.Image(fig)})

    return val_results


def iterate_train(
    model: torch.nn.Module,
    dataloaders: dict,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics_dict: dict | None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad()

    # tracks ground truth and predictions to compute metrics
    gts = {session: [] for session in dataloaders.keys()}
    preds = {session: [] for session in dataloaders.keys()}

    session_iters = {session: iter(dl) for session, dl in dataloaders.items()}
    active_sessions = sorted(dataloaders.keys())
    loss_per_batch = []
    # grad_per_batch = []

    while active_sessions:
        total_loss = 0.0
        sessions_to_remove = []
        for session in active_sessions:
            try:
                x_seq, y_seq = next(session_iters[session])
                pred = model(x_seq, session)

                gts[session].append(y_seq)
                preds[session].append(pred)

                loss = loss_fn(pred, y_seq)
                loss.backward()
                total_loss += loss.item()
                # grads = torch.cat(
                #     [
                #         p.grad.flatten()
                #         for p in model.parameters()
                #         if p.grad is not None
                #     ]
                # )
                # grad_norm = torch.linalg.norm(grads, ord=2).item()
                # grad_per_batch.append(grad_norm)
            except StopIteration:
                sessions_to_remove.append(session)

            optimizer.step()
            optimizer.zero_grad()

        loss_per_batch.append(total_loss / len(active_sessions))
        active_sessions = [
            session
            for session in active_sessions
            if session not in sessions_to_remove
        ]

    gts_stacked = {
        session: {
            modality: torch.cat([batch[modality] for batch in batches], dim=0)
            .detach()
            .cpu()
            .numpy()
            for modality in batches[0].keys()
        }
        for session, batches in gts.items()
    }

    preds_stacked = {
        session: {
            modality: torch.cat([pred[modality] for pred in batches], dim=0)
            .detach()
            .cpu()
            .numpy()
            for modality in batches[0].keys()
        }
        for session, batches in preds.items()
    }

    results = metrics.compute(gts_stacked, preds_stacked, metrics_dict)
    train_results = {"train_" + k: v for k, v in results.items()}
    train_results.update(
        {
            "train_loss": np.mean(loss_per_batch).item(),
            # "grad_norm": np.mean(grad_per_batch).item(),
        }
    )

    return train_results


def run(
    model: torch.nn.Module,
    dataloaders: dict[str, dict[str, torch.utils.data.DataLoader]],
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    early_stopper: modules.EarlyStopper | None,
    cfg: dict,
    logger: dict,
    plots_cfg: dict,
) -> None:
    """
    Run the training, validation and test process.
    """
    wandb.define_metric("epoch")
    wandb.define_metric("train_global_corr", step_metric="epoch")
    wandb.define_metric("train_local_corr", step_metric="epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("train_rmse", step_metric="epoch")
    wandb.define_metric("val_global_corr", step_metric="epoch")
    wandb.define_metric("val_local_corr", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")
    wandb.define_metric("val_rmse", step_metric="epoch")
    wandb.define_metric("test_global_corr", step_metric="epoch")
    wandb.define_metric("test_local_corr", step_metric="epoch")
    wandb.define_metric("test_loss", step_metric="epoch")
    wandb.define_metric("test_rmse", step_metric="epoch")
    wandb.define_metric("grad_norm", step_metric="epoch")
    wandb.define_metric("lr", step_metric="epoch")
    wandb.define_metric("weights_norm", step_metric="epoch")

    val_results = iterate_val(
        model=model,
        dataloaders=dataloaders["val"],
        loss_fn=loss_fn,
        metrics_dict=cfg["metrics"],
        plots_cfg=plots_cfg,
        logger=logger,
        epoch=0,
    )
    wandb.log(val_results)

    if early_stopper is not None:
        stopper_metric = early_stopper.metric
        minimize = early_stopper.minimize
    else:
        stopper_metric = "val_loss"
        minimize = True

    best_stopper_metric = val_results[stopper_metric]
    ignore_test = False

    for epoch in range(1, cfg["max_epochs"] + 1):
        wandb.log({"epoch": epoch})

        train_results = iterate_train(
            model=model,
            dataloaders=dataloaders["train"],
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics_dict=cfg["metrics"],
        )
        wandb.log(train_results)

        if np.isnan(train_results["train_loss"]):
            ignore_test = True
            warnings.warn("Stopping training due to NaN loss.")
            break

        if (epoch % cfg["validation_frequency"]) == 0:
            val_results = iterate_val(
                model=model,
                dataloaders=dataloaders["val"],
                loss_fn=loss_fn,
                metrics_dict=cfg["metrics"],
                logger=logger,
                plots_cfg=plots_cfg,
                epoch=epoch,
            )
            wandb.log(val_results)

            if early_stopper is not None and early_stopper.stop(
                val_results[stopper_metric]
            ):
                break

            if logger.get("checkpoints", True):
                if (
                    minimize
                    and val_results[stopper_metric] < best_stopper_metric
                ) or (
                    not minimize
                    and val_results[stopper_metric] > best_stopper_metric
                ):
                    log_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        early_stopper=early_stopper,
                        run_dir=logger["dir"],
                    )
                    best_stopper_metric = val_results[stopper_metric]

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_results["train_loss"])
        else:
            scheduler.step()
            last_lr = scheduler.get_last_lr()[0]
            wandb.log({"lr": last_lr})

    if not ignore_test:
        test_results = iterate_test(
            model=model,
            dataloaders=dataloaders["test"],
            loss_fn=loss_fn,
            metrics_dict=cfg["metrics"],
            logger=logger,
            plots_cfg=plots_cfg,
        )
        wandb.log(test_results)
