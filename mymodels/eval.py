from pathlib import Path

import torch
import yaml

from mymodels import epoch, modules, utils


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def run(
    run_uid,
    test_datasets,
    loss_fns,
    CONFIG_DIR,
    CHECKPOINT_DIR,
):
    CONFIG_DIR = Path(CONFIG_DIR)
    CONFIG_PATH = CONFIG_DIR / f"{run_uid}.yaml"
    CHECKPOINT_DIR = Path(CHECKPOINT_DIR)
    CHECKPOINT_PATH = CHECKPOINT_DIR / f"{run_uid}.pt"

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["seed"])

    if config["architecture"] == "gru":
        model = modules.GRUModel(config, config["sessions"], config["readout"])
    elif config["architecture"] == "fcnn":
        model = modules.FCModel(config, config["sessions"], config["readout"])
    else:
        raise ValueError(f"architecture {config['architecture']} not implemented.")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)

    test_dataloaders = {
        session: torch.utils.data.DataLoader(
            test_datasets[session], batch_size=64, shuffle=False
        )
        for session in test_datasets
    }

    in_modalities = utils.list_modalities(config["in_modalities"])
    out_modalities = utils.list_modalities(config["out_modalities"])

    metrics, preds, gts = epoch.iterate(
        model,
        test_dataloaders,
        in_modalities,
        out_modalities,
        loss_fns,
        optimizer=None,
        scheduler=None,
        mode="eval",
        config=config,
    )

    return metrics, preds, gts
