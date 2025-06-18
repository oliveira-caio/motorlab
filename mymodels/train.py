import datetime

from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

from mymodels import data, datasets, epoch, modules, utils


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


losses_dict = {
    "poisson": torch.nn.PoissonNLLLoss(log_input=False, full=True),
}


def track(metrics, config):
    if config["metrics"]:
        print(format_metrics(metrics))
    if config["wandb"]:
        wandb.log({k: v for k, v in metrics.items() if k != "epoch"})


def format_metrics(metrics):
    formatted = []
    for key, value in metrics.items():
        if key == "epoch":
            formatted.append(f"{key}: {value:04d}")
        elif "loss" in key or "correlation" in key:
            formatted.append(f"{key}: {value:.4f}")
        elif "accuracy" in key:
            formatted.append(f"{key}: {value:.2f}")
        elif "norm" in key:
            formatted.append(f"{key}: {value:.8f}")
        else:
            formatted.append(f"{key}: {value}")
    return " | ".join(formatted)


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
    for i in range(config["n_epochs"]):
        train_metrics, _, _ = epoch.iterate(
            model,
            train_dataloaders,
            in_modalities,
            out_modalities,
            loss_fns,
            optimizer,
            scheduler,
            "train",
            config,
        )

        if config["track"]:
            track(train_metrics, config["track"])

        if ((i + 1) % 20) == 0:
            valid_metrics, _, _ = epoch.iterate(
                model,
                valid_dataloaders,
                in_modalities,
                out_modalities,
                loss_fns,
                optimizer,
                scheduler,
                "validation",
                config,
            )
            track(valid_metrics, config["track"])

        if train_metrics["grad_norm"] < 1e-5 or np.isnan(train_metrics["loss"]):
            break


def run(config):
    utils.fix_seed(config["seed"])
    in_modalities = utils.list_modalities(config["in_modalities"])
    out_modalities = utils.list_modalities(config["out_modalities"])
    data_dict = data.load_all(config)

    config["in_dim"] = {
        session: sum([data_dict[session][m].shape[1] for m in in_modalities])
        for session in config["sessions"]
    }
    config["out_dim"] = {
        session: sum([data_dict[session][m].shape[1] for m in out_modalities])
        for session in config["sessions"]
    }

    tiles_data = {session: v["tiles"] for session, v in data_dict.items()}
    _, train_intervals, valid_intervals = utils.extract_intervals(config, tiles_data)

    train_datasets = {
        session: datasets.Deterministic(data_dict[session], train_intervals[session])
        for session in config["sessions"]
    }
    train_dataloaders = {
        session: torch.utils.data.DataLoader(
            train_datasets[session], batch_size=64, shuffle=True
        )
        for session in train_datasets
    }
    valid_datasets = {
        session: datasets.Deterministic(data_dict[session], valid_intervals[session])
        for session in config["sessions"]
    }
    valid_dataloaders = {
        session: torch.utils.data.DataLoader(
            valid_datasets[session], batch_size=64, shuffle=False
        )
        for session in valid_datasets
    }

    if config["architecture"] == "gru":
        model = modules.GRUModel(config)
    elif config["architecture"] == "fcnn":
        model = modules.FCModel(config)
    else:
        raise ValueError(f"architecture not implemented: {config['architecture']}.")

    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        config["train"].get("lr", 3e-4),
        weight_decay=config["train"].get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config["train"].get("T_max", 20)
    )
    loss_fns = {
        session: losses_dict[config["loss_fn"]] for session in config["sessions"]
    }

    loop(
        model,
        train_dataloaders,
        valid_dataloaders,
        in_modalities,
        out_modalities,
        loss_fns,
        optimizer,
        scheduler,
        config,
    )

    if config["save"]:
        uid = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        config["uid"] = uid
        print(f"uid: {uid}")

        CONFIG_DIR = Path(config["CONFIG_DIR"])
        CONFIG_PATH = CONFIG_DIR / f"{uid}.yaml"
        with open(CONFIG_PATH, "w") as f:
            yaml.safe_dump(config, f)

        CHECKPOINT_DIR = Path(config["CHECKPOINT_DIR"])
        CHECKPOINT_PATH = CHECKPOINT_DIR / f"{uid}.pt"
        torch.save(model.state_dict(), CHECKPOINT_PATH)
