from pathlib import Path

import torch

from motorlab import models, modules


def state(
    run_dir: Path | str,
    sessions: list[str],
    model_config: dict,
    optimizer_config: dict,
    scheduler_config: dict,
):
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / "best_model.pt"
    checkpoint_dict = torch.load(checkpoint_path)

    model = models.create(sessions, model_config)
    model.load_state_dict(checkpoint_dict["model_state_dict"])

    optimizer = modules.create_optimizer(model, optimizer_config)
    optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

    scheduler = modules.create_scheduler(optimizer, scheduler_config)
    scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])

    torch.set_rng_state(checkpoint_dict["random_state"])

    return model, optimizer, scheduler
