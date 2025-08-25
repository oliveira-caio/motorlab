"""
Unified training script for motorlab experiments.

Usage:
    motorlab --config quick.yml
    motorlab --config detailed.yml
    motorlab --config sweep.yml --sweep
    motorlab --config sweep.yml --sweep my_custom_sweep
    motorlab --config sweep.yml --sweep existing_sweep_id
"""

import argparse
import pprint

import torch
import wandb

from motorlab import data, metrics, models, modules, trainers, utils


def start_sweep(cfg: dict, sweep_cfg: dict):
    def sweep_train():
        utils.preprocess_config(cfg)
        wandb.init(name=cfg.get("uid"))
        cfg["logger"]["plots"] = False
        run_cfg = utils.parse_dotted_dict(dict(wandb.config))
        merged_cfg = utils.merge_dicts(cfg, run_cfg)
        start_training(merged_cfg)

    entity, project = "sinzlab", "motorlab"
    api = wandb.Api()

    sweep_name = sweep_cfg.get("name")
    sweep_id = None

    if not sweep_name:
        sweep_id = wandb.sweep(sweep_cfg, entity=entity, project=project)

    else:
        proj = api.project(name=project, entity=entity)
        sweeps = proj.sweeps()
        matches = [s for s in sweeps if s.name == sweep_name]
        if matches:
            matches.sort(
                key=lambda s: getattr(s, "created_at", 0), reverse=True
            )
            sweep_id = matches[0].id
        else:
            sweep_id = wandb.sweep(sweep_cfg, entity=entity, project=project)

    wandb.agent(
        sweep_id,
        function=sweep_train,
        entity=entity,
        project=project,
        count=100,
    )


def start_training(cfg: dict):
    utils.fix_seed(cfg["seed"])

    dataloaders = data.create_dataloaders_tiers(
        cfg["sessions"],
        cfg["experiment"],
        cfg["data"],
    )

    model = models.create(cfg["model"], cfg["data"]["dataset"]["output_dims"])
    print(model)
    loss_fn = metrics.MultiModalLoss(**cfg["trainer"]["loss_fns"])
    optimizer = modules.create_optimizer(
        model.parameters(), cfg["trainer"]["optimizer"]
    )
    scheduler = modules.create_scheduler(optimizer, cfg["trainer"]["scheduler"])
    early_stopper = modules.create_early_stopper(
        cfg["trainer"]["early_stopper"]
    )
    if cfg.get("old_uid"):
        checkpoint_dir = (
            f"{cfg['artifacts_dir']}/{cfg['task']}/{cfg['old_uid']}"
        )
        state_dict = utils.load_checkpoint(dir_path=checkpoint_dir)
        if not cfg.get("freeze_core", False):
            model.load_state_dict(state_dict["model"], strict=True)
            optimizer.load_state_dict(state_dict["optimizer"])
        else:
            model.load_state_dict(state_dict["model"], strict=False)
            exclude_params = {
                getattr(model, layer).parameters()
                for layer in ["core", "embedding"]
            }
            params_to_optimize = [
                p
                for p in model.parameters()
                if p.requires_grad and p not in exclude_params
            ]
            optimizer = modules.create_optimizer(
                params_to_optimize, cfg["trainer"]["optimizer"]
            )

        scheduler.load_state_dict(state_dict["scheduler"])
        torch.set_rng_state(state_dict["rng_state"])
        if early_stopper is not None and not cfg.get("freeze_core", False):
            early_stopper.load_state_dict(state_dict["early_stopper"])

    wandb.config.update(cfg)
    pprint.pprint(cfg, depth=6)

    plots_cfg = {
        "experiment": cfg["experiment"],
        "sampling_rate": cfg["data"]["sampling_rate"],
        "skeleton_type": cfg["data"]["modalities"]
        .get("poses", {})
        .get("skeleton_type"),
    }
    trainers.run(
        model=model,
        dataloaders=dataloaders,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopper=early_stopper,
        cfg=cfg["trainer"],
        logger=cfg["logger"],
        plots_cfg=plots_cfg,
    )

    if cfg["save"]:
        utils.save_config(cfg)


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Unified training script for motorlab experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Config YAML file (from setups/ directory)",
    )

    parser.add_argument(
        "--sweep",
        nargs="?",
        const="",
        help="Run wandb sweep. Empty=random name, string=sweep name or ID to join/create",
    )

    args = parser.parse_args()
    cfg = utils.load_config(args.config)
    cfg = utils.preprocess_config(cfg)

    if args.sweep:
        sweep_cfg = utils.load_config(args.sweep)
        start_sweep(cfg, sweep_cfg)
    else:
        wandb.init(project="motorlab", entity="sinzlab", name=cfg.get("uid"))
        start_training(cfg)


if __name__ == "__main__":
    main()
