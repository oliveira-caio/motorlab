import datetime

import torch

from motorlab import (
    data,
    datasets,
    intervals,
    utils,
    modules,
    config as config_module,
)
from motorlab.model.training import iterate, iterate_entire_trials, loop
from motorlab.model.factory import (
    compute_dimensions,
    create,
    load_model,
    save_checkpoint,
    load_checkpoint_metadata,
)


def setup(config: dict, train_intervals: dict, is_train: bool) -> dict:
    """
    Set up model, data, loss functions, and optionally optimizer/scheduler for training or evaluation.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing model and data parameters
    train_intervals : dict
        Training intervals for each session
    is_train : bool
        Whether setting up for training (affects model configuration)

    Returns
    -------
    dict
        Dictionary containing model, data_dict, loss_fns, and optionally optimizer/scheduler
    """
    utils.fix_seed(config.get("seed", 0))

    data_dict = data.load_all(
        data_dir=config["data_dir"],
        sessions=config["sessions"],
        in_modalities=config["in_modalities"],
        out_modalities=config["out_modalities"],
        experiment=config.get("experiment", "gbyk"),
        poses_config=config.get("poses", dict()),
        position_config=config.get("position", dict()),
        kinematics_config=config.get("kinematics", dict()),
        spikes_config=config.get("spikes", dict()),
        train_intervals=train_intervals,
    )

    model_dict = config["model"].copy()

    if is_train:
        in_dim, out_dim = compute_dimensions(
            data_dict=data_dict,
            in_modalities=config["in_modalities"],
            out_modalities=config["out_modalities"],
            sessions=config["sessions"],
            concat_input=config["dataset"].get("concat_input", True),
            concat_output=config["dataset"].get("concat_output", True),
            n_classes=config["model"].get("n_classes", None),
        )

        model_dict["in_dim"] = in_dim
        model_dict["out_dim"] = out_dim

    uid = config.get("uid", None)

    if uid is not None:
        if not model_dict.get("in_dim") or not model_dict.get("out_dim"):
            in_dim, out_dim = compute_dimensions(
                data_dict=data_dict,
                in_modalities=config["in_modalities"],
                out_modalities=config["out_modalities"],
                sessions=config["sessions"],
                concat_input=config["dataset"].get("concat_input", True),
                concat_output=config["dataset"].get("concat_output", True),
                n_classes=config["model"].get("n_classes", None),
            )
            model_dict["in_dim"] = in_dim
            model_dict["out_dim"] = out_dim

        checkpoint_data = load_checkpoint_metadata(
            uid=uid,
            checkpoint_dir=config.get(
                "checkpoint_load_dir", config["checkpoint_dir"]
            ),
            load_epoch=config.get("load_epoch", None),
        )

        model = load_model(
            architecture=config["model"]["architecture"],
            sessions=config["sessions"],
            readout_map=config["model"]["readout_map"],
            model_dict=model_dict,
            checkpoint_data=checkpoint_data,
            freeze=config.get("freeze_core", False),
            is_train=is_train,
        )

    else:
        model = create(
            architecture=config["model"]["architecture"],
            sessions=config["sessions"],
            readout_map=config["model"]["readout_map"],
            model_dict=model_dict,
            is_train=is_train,
        )
        checkpoint_data = None

    loss_fns = {
        session: modules.DictLoss(config["loss_fn"])
        for session in config["sessions"]
    }

    result = {
        "model": model,
        "data_dict": data_dict,
        "loss_fns": loss_fns,
    }

    if is_train:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"].get("lr", 5e-3)
        )

        schedulers = {
            "step_lr": torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config["training"].get("step_size", 25),
                gamma=config["training"].get("gamma", 0.4),
            ),
            "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["training"].get("T_max", 100),
                eta_min=config["training"].get("eta_min", 1e-4),
            ),
        }
        scheduler = schedulers[config["training"].get("scheduler", "step_lr")]

        if checkpoint_data is not None:
            if "optimizer_state_dict" in checkpoint_data:
                optimizer.load_state_dict(
                    checkpoint_data["optimizer_state_dict"]
                )
            if "scheduler_state_dict" in checkpoint_data:
                scheduler.load_state_dict(
                    checkpoint_data["scheduler_state_dict"]
                )
            if "random_state" in checkpoint_data:
                torch.set_rng_state(checkpoint_data["random_state"])

        result["optimizer"] = optimizer
        result["scheduler"] = scheduler
        result["checkpoint_data"] = checkpoint_data

        uid = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if "uid" in config:
            config["old_uid"] = config["uid"]
        config["uid"] = str(uid)
        print(f"uid: {uid}")

    return result


def train(config: dict) -> None:
    """
    Train a model using the provided configuration.

    This function sets up the data pipeline, model, and training process based
    on the provided configuration. It handles dataset creation, dataloader
    setup, optimizer initialization, and the main training loop.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - data_dir: Path to data directory
        - sessions: List of session names to include
        - experiment: Experiment identifier
        - intervals: Interval configuration for data splitting
        - model: Model architecture and parameters
        - dataset: dataset configuration (batch_size, seq_length, etc.)
        - train: Training parameters (lr, n_epochs)
        - loss_fn: Loss function specification
    """
    config = config_module.preprocess(config)

    test_intervals, train_intervals, valid_intervals = (
        intervals.load_all_by_tiers(
            data_dir=config["data_dir"],
            sessions=config["sessions"],
            experiment=config["experiment"],
            include_trial=config["intervals"].get("include_trial", True),
            include_homing=config["intervals"].get("include_homing", False),
            include_sitting=config["intervals"].get("include_sitting", True),
            balance_intervals=config["intervals"].get(
                "balance_intervals", False
            ),
            sampling_rate=config["intervals"].get("sampling_rate", 20),
        )
    )

    setup_result = setup(config, train_intervals, is_train=True)
    model = setup_result["model"]
    data_dict = setup_result["data_dict"]
    loss_fns = setup_result["loss_fns"]
    optimizer = setup_result["optimizer"]
    scheduler = setup_result["scheduler"]

    train_datasets = datasets.load_datasets(
        data_dict,
        train_intervals,
        config["in_modalities"],
        config["out_modalities"],
        entire_trials=config["dataset"].get("entire_trials", False),
        seq_length=config["dataset"].get("seq_length", 20),
        stride=config["dataset"].get("stride", 20),
        concat_input=config["dataset"].get("concat_input", True),
        concat_output=config["dataset"].get("concat_output", True),
    )

    train_dataloaders = datasets.load_dataloaders(
        train_datasets,
        batch_size=config["dataset"].get("batch_size", 64),
        shuffle=True,
    )

    valid_datasets = datasets.load_datasets(
        data_dict,
        valid_intervals,
        config["in_modalities"],
        config["out_modalities"],
        entire_trials=config["dataset"].get("entire_trials", False),
        seq_length=config["dataset"].get("seq_length", 20),
        stride=config["dataset"].get("stride", 20),
        concat_input=config["dataset"].get("concat_input", True),
        concat_output=config["dataset"].get("concat_output", True),
    )

    valid_dataloaders = datasets.load_dataloaders(
        valid_datasets,
        batch_size=config["dataset"].get("batch_size", 64),
        shuffle=False,
    )

    loop(
        model,
        train_dataloaders,
        valid_dataloaders,
        loss_fns,
        optimizer,
        scheduler,
        config["metric"],
        config["training"].get("n_epochs", 250),
        print_metrics=config["track"].get("metrics", False),
        use_wandb=config["track"].get("wandb", False),
        save_checkpoint=config["track"].get("checkpoint", False),
        checkpoint_dir=config.get(
            "checkpoint_save_dir", config["checkpoint_dir"]
        ),
        uid=config["uid"],
    )

    if config.get("save", False):
        config_module.save(config)

        checkpoint_dir = config.get(
            "checkpoint_save_dir", config["checkpoint_dir"]
        )
        checkpoint_path = f"{checkpoint_dir}/{config['uid']}.pt"
        save_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            optimizer=optimizer,
            scheduler=scheduler,
        )


def evaluate(config: dict) -> tuple[dict, dict, dict]:
    """
    Evaluate a model using the provided configuration.

    This function loads a trained model and evaluates it on test data,
    computing specified metrics and returning ground truths and predictions.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - data_dir: Path to data directory
        - sessions: List of session names to include
        - experiment: Experiment identifier
        - intervals: Interval configuration for data splitting
        - model: Model architecture and parameters
        - dataset: Dataset configuration
        - uid: Model checkpoint identifier
        - metric: Evaluation metric to compute

    Returns
    -------
    tuple[dict, dict, dict]
        Tuple containing (metrics, ground_truths, predictions) where:
        - metrics: Dict of computed metric values
        - ground_truths: Dict mapping sessions to ground truth arrays
        - predictions: Dict mapping sessions to prediction arrays
    """
    config = config_module.preprocess(config)

    test_intervals, train_intervals, valid_intervals = (
        intervals.load_all_by_tiers(
            data_dir=config["data_dir"],
            sessions=config["sessions"],
            experiment=config["experiment"],
            include_trial=config["intervals"].get("include_trial", True),
            include_homing=config["intervals"].get("include_homing", False),
            include_sitting=config["intervals"].get("include_sitting", True),
            balance_intervals=config["intervals"].get(
                "balance_intervals", False
            ),
            sampling_rate=config["intervals"].get("sampling_rate", 20),
        )
    )

    setup_result = setup(config, train_intervals, is_train=False)
    model = setup_result["model"]
    data_dict = setup_result["data_dict"]
    loss_fns = setup_result["loss_fns"]

    test_datasets = datasets.load_datasets(
        data_dict,
        test_intervals,
        config["in_modalities"],
        config["out_modalities"],
        entire_trials=config["dataset"].get("entire_trials", False),
        seq_length=config["dataset"].get("seq_length", 20),
        stride=config["dataset"].get("stride", 20),
        concat_input=config["dataset"].get("concat_input", True),
        concat_output=config["dataset"].get("concat_output", True),
    )

    test_dataloaders = datasets.load_dataloaders(
        test_datasets,
        batch_size=config["dataset"].get("batch_size", 64),
        shuffle=False,
    )

    if not config["dataset"].get("entire_trials", False):
        metrics, gts, preds = iterate(
            model,
            test_dataloaders,
            loss_fns,
            optimizer=None,
            metric=config["metric"],
        )
    else:
        metrics, gts, preds = iterate_entire_trials(
            model,
            test_dataloaders,
            config["dataset"].get("seq_length", 20),
            metric=config["metric"],
        )

    return metrics, gts, preds
