import datetime

from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

from motorlab import data, datasets, epoch, metrics, modules, utils


def iterate(
    model,
    dataloaders,
    in_modalities,
    out_modalities,
    loss_fns,
    optimizer,
    scheduler,
    mode,
    config,
):
    is_train = mode == "train"
    model.train() if is_train else model.eval()
    gts = dict()
    preds = dict()
    track_metrics = dict()
    losses = []

    with torch.set_grad_enabled(is_train):
        for session in dataloaders:
            gts[session] = []
            preds[session] = []
            for d in dataloaders[session]:
                x = torch.cat([d[m] for m in in_modalities], dim=-1).to(utils.device)
                y = torch.cat([d[m] for m in out_modalities], dim=-1).to(utils.device)

                pred = model(x, session)
                gts[session].append(y)
                preds[session].append(pred)

                if is_train:
                    optimizer.zero_grad()
                    loss = loss_fns[session](pred, y)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

        stacked_preds = {
            session: torch.cat(preds[session], dim=0).detach().cpu().numpy()
            for session in preds
        }
        stacked_gts = {
            session: torch.cat(gts[session], dim=0).detach().cpu().numpy()
            for session in gts
        }
        track_metrics[config["metric"]] = metrics.compute(
            stacked_preds, stacked_gts, config["metric"]
        )

        if is_train:
            scheduler.step()
            grad_norm = sum(
                p.grad.norm().item() for p in model.parameters() if p.grad is not None
            )
            track_metrics["grad_norm"] = grad_norm
            track_metrics["loss"] = np.mean(losses)

    return track_metrics, stacked_preds, stacked_gts


def track(metrics, config):
    if config.get("metrics", False):
        print(format_metrics(metrics))
    if config.get("wandb", False):
        wandb.log({k: v for k, v in metrics.items() if k != "epoch"})


def format_metrics(metrics):
    formatted = []
    for key, value in metrics.items():
        if "loss" in key:
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
    for i in range(config["train"]["n_epochs"]):
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
        train_metrics["epoch"] = i
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


def create_model(config, is_train):
    """Create the model based on the configuration."""
    if config["architecture"] == "gru":
        model = modules.GRUModel(config)
    elif config["architecture"] == "fcnn":
        model = modules.FCModel(config)
    else:
        raise ValueError(f"architecture not implemented: {config['architecture']}.")

    if is_train:
        model.to(utils.device)
        print(model)
    else:
        CHECKPOINT_DIR = Path(config["CHECKPOINT_DIR"])
        CHECKPOINT_PATH = CHECKPOINT_DIR / f"{config['uid']}.pt"
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=utils.device))
        model.to(utils.device)

    return model


def create_loss_fns(config):
    """Create the loss functions based on the configuration."""
    return {
        session: modules.losses_map(config["loss_fn"]) for session in config["sessions"]
    }


def create_datasets(data_dict, intervals):
    return {
        session: datasets.Deterministic(data_dict[session], intervals[session])
        for session in data_dict
    }


def create_dataloaders(datasets):
    return {
        session: torch.utils.data.DataLoader(
            datasets[session], batch_size=64, shuffle=True
        )
        for session in datasets
    }


def save_model(model, config):
    uid = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    config["uid"] = uid
    print(f"uid: {uid}")

    CONFIG_DIR = Path(config["CONFIG_DIR"])
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH = CONFIG_DIR / f"{uid}.yaml"
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f)

    CHECKPOINT_DIR = Path(config["CHECKPOINT_DIR"])
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH = CHECKPOINT_DIR / f"{uid}.pt"
    torch.save(model.state_dict(), CHECKPOINT_PATH)


def setup(config, is_train):
    utils.fix_seed(config["seed"])
    in_modalities = utils.list_modalities(config["in_modalities"])
    out_modalities = utils.list_modalities(config["out_modalities"])
    data_dict = data.load_all(config)

    if is_train:
        config["model"]["in_dim"] = {
            session: sum([data_dict[session][m].shape[1] for m in in_modalities])
            for session in config["sessions"]
        }

        if "n_classes" in config["model"]:
            config["model"]["out_dim"] = {
                session: config["model"]["n_classes"] for session in config["sessions"]
            }
        else:
            config["model"]["out_dim"] = {
                session: sum([data_dict[session][m].shape[1] for m in out_modalities])
                for session in config["sessions"]
            }

    model = create_model(config, is_train)
    loss_fns = create_loss_fns(config)

    return model, data_dict, loss_fns


def train(config):
    model, data_dict, loss_fns = setup(config, is_train=True)
    _, train_intervals, valid_intervals = utils.extract_intervals(config)
    train_datasets = create_datasets(data_dict, train_intervals)
    train_dataloaders = create_dataloaders(train_datasets)
    valid_datasets = create_datasets(data_dict, valid_intervals)
    valid_dataloaders = create_dataloaders(valid_datasets)

    optimizer = torch.optim.Adam(
        model.parameters(),
        config["train"].get("lr", 3e-4),
        weight_decay=config["train"].get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config["train"].get("T_max", 20)
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
        save_model(model, config)


def evaluate(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    test_intervals = utils.extract_intervals(config)[0]
    model, data_dict, loss_fns = setup(config, is_train=False)
    test_datasets = create_datasets(data_dict, test_intervals)
    test_dataloaders = create_dataloaders(test_datasets)

    metrics, preds, gts = epoch.iterate(
        model,
        test_dataloaders,
        utils.list_modalities(config["in_modalities"]),
        utils.list_modalities(config["out_modalities"]),
        loss_fns,
        optimizer=None,
        scheduler=None,
        mode="eval",
        config=config,
    )

    return metrics, preds, gts
