"""Utility functions for wandb integration."""

import wandb


def init_run(name: str | None = None):
    wandb.init(
        project="motorlab",
        name=name,
    )


def start_sweep(sweep_config: dict) -> str:
    sweep_id = wandb.sweep(sweep_config, project="motorlab")
    print(f"Created sweep: {sweep_id}")
    return sweep_id


def run_sweep(sweep_id: str, train_function):
    wandb.agent(sweep_id, train_function, project="motorlab")


def check_sweep_exists(sweep_id: str) -> bool:
    try:
        api = wandb.Api()
        api.sweep(f"motorlab/{sweep_id}")
        return True
    except Exception:
        return False


def handle_sweep(sweep_arg: str | None, sweep_config: dict) -> str:
    if sweep_arg is None:
        sweep_id = start_sweep(sweep_config)
        print(f"Created new sweep: {sweep_id}")
        return sweep_id
    if check_sweep_exists(sweep_arg):
        print(f"Resuming existing sweep: {sweep_arg}")
        return sweep_arg
    else:
        sweep_config_with_name = sweep_config.copy()
        sweep_config_with_name["name"] = sweep_arg
        sweep_id = start_sweep(sweep_config_with_name)
        print(f"Created new sweep with name '{sweep_arg}': {sweep_id}")
        return sweep_id


def load_sweep_config() -> dict:
    return dict(wandb.config)


def download_sweep_data(sweep_id: str) -> list:
    api = wandb.Api()
    sweep = api.sweep(f"sinzlab/motorlab/{sweep_id}")
    runs_data = []
    for run in sweep.runs:
        run_data = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "created_at": run.created_at,
            "runtime": run._attrs.get("runtime", 0),
        }
        for key, value in run.config.items():
            run_data[f"config.{key}"] = value
        for key, value in run.summary.items():
            if not key.startswith("_"):
                run_data[f"summary.{key}"] = value
        runs_data.append(run_data)
    return runs_data
