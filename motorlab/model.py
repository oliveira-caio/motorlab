import datetime

from pathlib import Path

import numpy as np
import torch
import wandb
import yaml


from . import data
from . import datasets
from . import metrics
from . import modules
from . import utils


def track(metrics, config, model):
    if config["track"].get("metrics", False):
        print(format_metrics(metrics))
    if config["track"].get("wandb", False):
        wandb.log({k: v for k, v in metrics.items() if k != "epoch"})
    if (
        config["track"].get("save_checkpoint", False)
        and metrics["epoch"] != config["train"]["n_epochs"]
        and (metrics["epoch"] - 1) % 25 == 0
        and metrics["epoch"] != 1
    ):
        save_checkpoint(model, config, metrics["epoch"] - 1)


def format_metrics(metrics):
    formatted = []
    for key, value in metrics.items():
        if "loss" in key or "mse" in key:
            formatted.append(f"{key}: {value:.4f}")
        elif "accuracy" in key:
            formatted.append(f"{key}: {value:.2f}")
        elif "norm" in key:
            formatted.append(f"{key}: {value:.8f}")
        elif "epoch" in key:
            formatted.append(f"{key}: {value:04d}")
        elif "correlation" in key:
            formatted.append(f"global correlation: {value[0]:.4f}")
            formatted.append(f"local correlation: {value[1]:.4f}")
    return " | ".join(reversed(formatted))


def dump_model_outputs(stacked_gts, stacked_preds, label):
    import os

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


def model_mean(model):
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


def iterate(
    model,
    dataloaders,
    in_modalities,
    out_modalities,
    loss_fns,
    optimizer,
    config,
    is_train=False,
):
    model.train() if is_train else model.eval()
    gts = {session: [] for session in dataloaders}
    preds = {session: [] for session in dataloaders}
    track_metrics = dict()
    losses = []

    with torch.set_grad_enabled(is_train):
        for session in dataloaders:
            for d in dataloaders[session]:
                x = torch.cat([d[m] for m in in_modalities], dim=-1).to(
                    utils.device
                )
                y = torch.cat([d[m] for m in out_modalities], dim=-1).to(
                    utils.device
                )

                pred = model(x, session)
                gts[session].append(y)
                preds[session].append(pred)

                if is_train:
                    optimizer.zero_grad()
                    loss = loss_fns[session](pred, y)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

        stacked_gts = {
            session: torch.cat(gts[session], dim=0).detach().cpu().numpy()
            for session in gts
        }

        stacked_preds = {
            session: torch.cat(preds[session], dim=0).detach().cpu().numpy()
            for session in preds
        }

        if "metric" in config:
            track_metrics[config["metric"]] = metrics.compute(
                stacked_gts, stacked_preds, config["metric"]
            )

        if is_train:
            grad_norm = sum(
                p.grad.norm().item()
                for p in model.parameters()
                if p.grad is not None
            )
            track_metrics["grad_norm"] = grad_norm
            track_metrics["loss"] = np.mean(losses)

    return track_metrics, stacked_gts, stacked_preds


def loop(
    model,
    train_dataloaders,
    valid_dataloaders,
    in_modalities,
    out_modalities,
    loss_fns,
    optimizer,
    scheduler,
    config,
):
    valid_metrics, _, _ = iterate(
        model,
        valid_dataloaders,
        in_modalities,
        out_modalities,
        loss_fns,
        optimizer,
        config,
    )
    valid_metrics["epoch"] = 1
    if "track" in config:
        track(valid_metrics, config, model)

    for i in range(config["train"]["n_epochs"]):
        train_metrics, _, _ = iterate(
            model,
            train_dataloaders,
            in_modalities,
            out_modalities,
            loss_fns,
            optimizer,
            config,
            is_train=True,
        )
        train_metrics["epoch"] = i + 1
        if "track" in config:
            track(train_metrics, config, model)

        if ((i + 1) % 25) == 0:
            valid_metrics, _, _ = iterate(
                model,
                valid_dataloaders,
                in_modalities,
                out_modalities,
                loss_fns,
                optimizer,
                config,
            )
            valid_metrics["epoch"] = i + 1
            if "track" in config:
                track(valid_metrics, config, model)

        scheduler.step()
        if train_metrics["grad_norm"] < 1e-5 or np.isnan(train_metrics["loss"]):
            break


def create_model(config, is_train):
    if config["architecture"] == "gru":
        model = modules.GRUModel(config)
    elif config["architecture"] == "fc":
        model = modules.FCModel(config)
    else:
        raise ValueError(
            f"architecture not implemented: {config['architecture']}."
        )

    if "uid" in config or not is_train:
        CHECKPOINT_DIR = Path(
            config["CHECKPOINT_LOAD_DIR"]
            if "CHECKPOINT_LOAD_DIR" in config
            else config["CHECKPOINT_DIR"]
        )

        if "load_epoch" in config:
            CHECKPOINT_PATH = (
                CHECKPOINT_DIR / f"{config['uid']}_{config['load_epoch']}.pt"
            )
        else:
            CHECKPOINT_PATH = CHECKPOINT_DIR / f"{config['uid']}.pt"

        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))

    if config.get("freeze_core", False):
        CHECKPOINT_DIR = Path(
            config["CHECKPOINT_LOAD_DIR"]
            if "CHECKPOINT_LOAD_DIR" in config
            else config["CHECKPOINT_DIR"]
        )

        if "load_epoch" in config:
            CHECKPOINT_PATH = (
                CHECKPOINT_DIR / f"{config['uid']}_{config['load_epoch']}.pt"
            )
        else:
            CHECKPOINT_PATH = CHECKPOINT_DIR / f"{config['uid']}.pt"

        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if "in_layer" in k or "core" in k
        }
        model.load_state_dict(filtered_state_dict, strict=False)

        # todo: to make this more flexible, create a list of modules to freeze, loop over this list and freeze only the params of the listed modules.
        for param in model.embedding.parameters():
            param.requires_grad = False

        for param in model.core.parameters():
            param.requires_grad = False

    if is_train:
        print(model)

    model.to(utils.device)
    return model


def create_loss_fns(config):
    return {
        session: modules.losses_map(config["loss_fn"])
        for session in config["sessions"]
    }


def create_datasets(data_dict, intervals, config):
    return {
        session: datasets.Deterministic(
            data_dict[session],
            intervals[session],
            config,
        )
        for session in data_dict
    }


def create_dataloaders(datasets, is_train=False):
    return {
        session: torch.utils.data.DataLoader(
            datasets[session], batch_size=64, shuffle=is_train
        )
        for session in datasets
    }


def save_config(config):
    CONFIG_DIR = Path(
        config["CONFIG_SAVE_DIR"]
        if "CONFIG_SAVE_DIR" in config
        else config["CONFIG_DIR"]
    )
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH = CONFIG_DIR / f"{config['uid']}.yaml"
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f)


def save_checkpoint(model, config, epoch=None):
    CHECKPOINT_DIR = Path(
        config["CHECKPOINT_SAVE_DIR"]
        if "CHECKPOINT_SAVE_DIR" in config
        else config["CHECKPOINT_DIR"]
    )
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{config['uid']}_{epoch}.pt" if epoch else f"{config['uid']}.pt"
    CHECKPOINT_PATH = CHECKPOINT_DIR / filename
    torch.save(model.state_dict(), CHECKPOINT_PATH)


def setup(config, is_train):
    utils.fix_seed(config.get("seed", 0))
    in_modalities = utils.list_modalities(config["in_modalities"])
    out_modalities = utils.list_modalities(config["out_modalities"])
    data_dict = data.load_all(config)

    if is_train:
        config["model"]["in_dim"] = {
            session: sum(
                [data_dict[session][m].shape[1] for m in in_modalities]
            )
            for session in config["sessions"]
        }

        if "n_classes" in config["model"]:
            config["model"]["out_dim"] = {
                session: config["model"]["n_classes"]
                for session in config["sessions"]
            }
        else:
            config["model"]["out_dim"] = {
                session: sum(
                    [data_dict[session][m].shape[1] for m in out_modalities]
                )
                for session in config["sessions"]
            }

    model = create_model(config, is_train)
    loss_fns = create_loss_fns(config)

    if is_train:
        uid = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if "uid" in config:
            config["old_uid"] = config["uid"]
        config["uid"] = str(uid)
        print(f"uid: {uid}")

    return model, data_dict, loss_fns


def train(config):
    model, data_dict, loss_fns = setup(config, is_train=True)

    _, train_intervals, valid_intervals = utils.extract_intervals(config)
    train_datasets = create_datasets(
        data_dict, train_intervals, config["dataset"]
    )
    train_dataloaders = create_dataloaders(train_datasets, is_train=False)
    valid_datasets = create_datasets(
        data_dict, valid_intervals, config["dataset"]
    )
    valid_dataloaders = create_dataloaders(valid_datasets)

    params = (
        model.parameters()
        if not config.get("freeze_core", False)
        else filter(lambda p: p.requires_grad, model.parameters())
    )
    optimizer = torch.optim.Adam(
        params,
        lr=config["train"].get("lr", 5e-3),
        weight_decay=config["train"].get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=25, gamma=0.4
    )

    loop(
        model,
        train_dataloaders,
        valid_dataloaders,
        utils.list_modalities(config["in_modalities"]),
        utils.list_modalities(config["out_modalities"]),
        loss_fns,
        optimizer,
        scheduler,
        config,
    )

    if config.get("save", False):
        save_config(config)
        save_checkpoint(model, config)


def evaluate(config):
    model, data_dict, loss_fns = setup(config, is_train=False)
    test_intervals = utils.extract_intervals(config)[0]
    test_datasets = create_datasets(
        data_dict, test_intervals, config["dataset"]
    )
    test_dataloaders = create_dataloaders(test_datasets)

    metrics, gts, preds = iterate(
        model,
        test_dataloaders,
        utils.list_modalities(config["in_modalities"]),
        utils.list_modalities(config["out_modalities"]),
        loss_fns,
        optimizer=None,
        config=config,
    )

    return metrics, gts, preds
