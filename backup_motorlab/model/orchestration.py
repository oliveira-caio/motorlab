"""
High-level orchestration functions for training and evaluation workflows.

This module contains the main entry points that coordinate the entire machine learning
pipeline, from data loading to model training and evaluation.
"""

from motorlab import config as config_module, datasets
from motorlab.modalities import intervals
from motorlab.model.core import setup
from motorlab.model.training import loop, iterate, iterate_entire_trials
from motorlab.model.factory import save_checkpoint


def train(config: dict) -> None:
    """
    Train a model using the provided configuration.

    This function orchestrates the complete training pipeline: data loading,
    model setup, training loop execution, and checkpoint saving.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - paths: Path configuration (data_dir, artifacts_dir)
        - sessions: List of session names to include
        - experiment: Experiment identifier
        - modalities: Modality configurations including intervals
        - model: Model architecture and parameters
        - data: Data configuration (modalities, dataset, dataloader)
        - training: Training parameters (max_epochs, optimizer, etc.)
        - tracking: Logging and checkpoint configuration
    """
    config = config_module.preprocess(config)

    intervals_config = config["modalities"]["intervals"]
    _, train_intervals, valid_intervals = intervals.load_all_by_tiers(
        data_dir=config["paths"]["data_dir"],
        sessions=config["sessions"],
        experiment=config["experiment"],
        include_trial=intervals_config.get("include_trial", True),
        include_homing=intervals_config.get("include_homing", False),
        include_sitting=intervals_config.get("include_sitting", True),
        balance_intervals=intervals_config.get("balance_intervals", False),
    )

    setup_result = setup(config, train_intervals, is_train=True)
    model = setup_result["model"]
    data_dict = setup_result["data_dict"]
    loss_fns = setup_result["loss_fns"]
    optimizer = setup_result["optimizer"]
    scheduler = setup_result["scheduler"]
    early_stopping = setup_result["early_stopping"]

    train_datasets = datasets.load_datasets(
        data_dict=data_dict,
        intervals=train_intervals,
        input_modalities=config["data"]["input_modalities"],
        output_modalities=config["data"]["output_modalities"],
        stride=config["data"]["dataset"]["stride"],
        concat_input=config["data"]["dataset"]["concat_input"],
        concat_output=config["data"]["dataset"]["concat_output"],
    )

    train_dataloaders = datasets.load_dataloaders(
        datasets=train_datasets,
        min_length=config["data"]["dataloader"]["min_length"],
        max_length=config["data"]["dataloader"]["max_length"],
        variable_length=config["data"]["dataloader"]["variable_length"],
        batch_size=config["data"]["dataloader"]["batch_size"],
        shuffle=True,
        test_mode=False,
    )

    valid_datasets = datasets.load_datasets(
        data_dict=data_dict,
        intervals=valid_intervals,
        input_modalities=config["data"]["input_modalities"],
        output_modalities=config["data"]["output_modalities"],
        stride=config["data"]["dataset"]["stride"],
        concat_input=config["data"]["dataset"]["concat_input"],
        concat_output=config["data"]["dataset"]["concat_output"],
    )

    valid_dataloaders = datasets.load_dataloaders(
        datasets=valid_datasets,
        min_length=config["data"]["dataloader"]["max_length"],
        max_length=config["data"]["dataloader"]["max_length"],  # fixed length
        variable_length=config["data"]["dataloader"]["variable_length"],
        batch_size=config["data"]["dataloader"]["batch_size"],
        shuffle=False,
        test_mode=False,
    )

    loop(
        model=model,
        train_dataloaders=train_dataloaders,
        valid_dataloaders=valid_dataloaders,
        optimizer=optimizer,
        loss_fns=loss_fns,
        metric=config["training"]["metric"],
        uid=config["uid"],
        n_epochs=config["training"]["max_epochs"],
        checkpoint_dir=config["paths"].get(
            "checkpoint_save_dir", config["paths"]["checkpoint_dir"]
        ),
        validation_config=config["training"]["validation"],
        scheduler=scheduler,
        early_stopping=early_stopping,
        use_wandb=config["tracking"].get("wandb", False),
        save_checkpoint=config["tracking"].get("checkpoint", False),
        log_metrics=config["tracking"].get("logging", True),
    )

    if config.get("save", False):
        config_module.save(config)

        checkpoint_dir = config["paths"].get(
            "checkpoint_save_dir", config["paths"]["checkpoint_dir"]
        )
        checkpoint_path = f"{checkpoint_dir}/{config['uid']}.pt"
        save_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            optimizer=optimizer,
            scheduler=scheduler,
        )


def evaluate(config: dict, test_mode: bool) -> tuple[dict, dict, dict]:
    """
    Evaluate a model using the provided configuration.

    This function orchestrates the complete evaluation pipeline: data loading,
    model setup, and metric computation on test data.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - paths: Path configuration (data_dir, artifacts_dir)
        - sessions: List of session names to include
        - experiment: Experiment identifier
        - modalities: Modality configurations including intervals
        - model: Model architecture and parameters
        - data: Data configuration (modalities, dataset, dataloader)
        - training: Training parameters and metric specification
        - uid: Model checkpoint identifier
    test_mode : bool
        Whether to use test mode (entire trials, batch_size=1)

    Returns
    -------
    tuple[dict, dict, dict]
        Tuple containing (metrics, ground_truths, predictions) where:
        - metrics: Dict of computed metric values
        - ground_truths: Dict mapping sessions to ground truth arrays
        - predictions: Dict mapping sessions to prediction arrays
    """
    config = config_module.preprocess(config)
    config["tracking"]["wandb"] = False  # Disable wandb for evaluation

    intervals_config = config["modalities"]["intervals"]
    test_intervals, train_intervals, _ = intervals.load_all_by_tiers(
        data_dir=config["paths"]["data_dir"],
        sessions=config["sessions"],
        experiment=config["experiment"],
        include_trial=intervals_config.get("include_trial", True),
        include_homing=intervals_config.get("include_homing", False),
        include_sitting=intervals_config.get("include_sitting", True),
        balance_intervals=intervals_config.get("balance_intervals", False),
    )

    setup_result = setup(config, train_intervals, is_train=False)
    model = setup_result["model"]
    data_dict = setup_result["data_dict"]
    loss_fns = setup_result["loss_fns"]

    test_datasets = datasets.load_datasets(
        data_dict=data_dict,
        intervals=test_intervals,
        input_modalities=config["data"]["input_modalities"],
        output_modalities=config["data"]["output_modalities"],
        stride=config["data"]["dataset"]["stride"],
        concat_input=config["data"]["dataset"]["concat_input"],
        concat_output=config["data"]["dataset"]["concat_output"],
    )

    test_dataloaders = datasets.load_dataloaders(
        datasets=test_datasets,
        min_length=config["data"]["dataloader"]["max_length"],
        max_length=config["data"]["dataloader"]["max_length"],  # fixed length
        variable_length=config["data"]["dataloader"]["variable_length"],
        batch_size=config["data"]["dataloader"]["batch_size"],
        shuffle=False,
        test_mode=test_mode,
    )

    if test_mode:
        metrics, gts, preds = iterate_entire_trials(
            model,
            test_dataloaders,
            metric=config["training"]["metric"],
        )
    else:
        metrics, gts, preds = iterate(
            model,
            test_dataloaders,
            loss_fns,
            optimizer=None,
            metric=config["training"]["metric"],
            is_train=False,
        )

    return metrics, gts, preds
