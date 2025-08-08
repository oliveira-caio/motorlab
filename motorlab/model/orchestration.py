"""
High-level orchestration functions for training and evaluation workflows.

This module contains the main entry points that coordinate the entire machine learning
pipeline, from data loading to model training and evaluation.
"""

from motorlab import config as config_module, datasets, intervals
from motorlab.model.core import setup
from motorlab.model.training import loop, iterate, iterate_entire_trials
from motorlab.model.factory import save_checkpoint


def prepare_data_pipeline(
    config: dict, data_dict: dict, intervals_dict: dict, shuffle: bool = True
) -> tuple[dict, dict]:
    """
    Prepare datasets and dataloaders from configuration and intervals.

    This function handles the common data preparation logic shared between
    training and evaluation workflows.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing dataset parameters
    data_dict : dict
        Dictionary containing loaded data for all sessions
    intervals_dict : dict
        Dictionary mapping session names to interval lists
    shuffle : bool, optional
        Whether to shuffle the data, by default True

    Returns
    -------
    tuple[dict, dict]
        Tuple of (datasets, dataloaders) dictionaries
    """
    datasets_dict = datasets.load_datasets(
        data_dict=data_dict,
        intervals=intervals_dict,
        in_modalities=config["in_modalities"],
        out_modalities=config["out_modalities"],
        entire_trials=config["dataset"].get("entire_trials", False),
        seq_length=config["dataset"].get("seq_length", 20),
        stride=config["dataset"].get("stride", 20),
        concat_input=config["dataset"].get("concat_input", True),
        concat_output=config["dataset"].get("concat_output", True),
    )

    dataloaders_dict = datasets.load_dataloaders(
        datasets_dict,
        batch_size=config["dataset"].get("batch_size", 64),
        shuffle=shuffle,
    )

    return datasets_dict, dataloaders_dict


def train(config: dict) -> None:
    """
    Train a model using the provided configuration.

    This function orchestrates the complete training pipeline: data loading,
    model setup, training loop execution, and checkpoint saving.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - data_dir: Path to data directory
        - sessions: List of session names to include
        - experiment: Experiment identifier
        - intervals: Interval configuration for data splitting
        - model: Model architecture and parameters
        - dataset: Dataset configuration (batch_size, seq_length, etc.)
        - training: Training parameters (lr, n_epochs, etc.)
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
    early_stopping = setup_result["early_stopping"]

    train_datasets, train_dataloaders = prepare_data_pipeline(
        config, data_dict, train_intervals, shuffle=True
    )

    valid_datasets, valid_dataloaders = prepare_data_pipeline(
        config, data_dict, valid_intervals, shuffle=False
    )

    loop(
        model=model,
        train_dataloaders=train_dataloaders,
        valid_dataloaders=valid_dataloaders,
        optimizer=optimizer,
        loss_fns=loss_fns,
        metric=config["metric"],
        uid=config["uid"],
        n_epochs=config["training"].get("n_epochs", 250),
        checkpoint_dir=config.get(
            "checkpoint_save_dir", config["checkpoint_dir"]
        ),
        validation_config=config["training"]["validation"],
        scheduler=scheduler,
        early_stopping=early_stopping,
        use_wandb=config["track"].get("wandb", False),
        save_checkpoint=config["track"].get("checkpoint", False),
        log_metrics=config["track"].get("logging", True),
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

    This function orchestrates the complete evaluation pipeline: data loading,
    model setup, and metric computation on test data.

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
    config["track"]["wandb"] = False  # Disable wandb for evaluation

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

    test_datasets, test_dataloaders = prepare_data_pipeline(
        config, data_dict, test_intervals, shuffle=False
    )

    if not config["dataset"].get("entire_trials", False):
        metrics, gts, preds = iterate(
            model,
            test_dataloaders,
            loss_fns,
            optimizer=None,
            metric=config["metric"],
            is_train=False,
        )
    else:
        metrics, gts, preds = iterate_entire_trials(
            model,
            test_dataloaders,
            config["dataset"].get("seq_length", 20),
            metric=config["metric"],
        )

    return metrics, gts, preds
