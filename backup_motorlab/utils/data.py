from pathlib import Path

import numpy as np
import yaml

from sklearn.decomposition import PCA

from motorlab.modalities import location, poses, spikes


DEFAULT_DTYPE = np.float32


def project_to_pca(
    data: np.ndarray,
    intervals: list | None = None,
    divide_variance: bool = False,
) -> tuple[np.ndarray, PCA]:
    """
    Project data to principal components using PCA.

    Parameters
    ----------
    data : np.ndarray
        Data array of shape (n_frames, n_features).
    intervals : list, optional
        List of (start, end) intervals to fit PCA. If None, use all data. Default is None.
    divide_variance : bool, optional
        Whether to standardize data before PCA. Default is False.

    Returns
    -------
    tuple
        (Transformed data, fitted PCA object)
    """
    if intervals:
        data_ = np.concatenate([data[s:e] for s, e in intervals], axis=0)
    else:
        data_ = data

    if divide_variance:
        data_ -= data_.mean(axis=0)
        data_ /= data_.std(axis=0)

    pca = PCA(n_components=data_.shape[-1])
    pca.fit(data_)

    data_ = transform_with_nans(data, pca)

    return data_, pca


def transform_with_nans(data: np.ndarray, pca: PCA) -> np.ndarray:
    """
    Transform data using PCA, handling NaN values.

    If any feature in a frame is NaN, the entire transformed frame will be NaN.
    Otherwise, applies normal PCA transformation.

    Parameters
    ----------
    data : np.ndarray
        Data array of shape (n_frames, n_features), may contain NaNs.
    pca : PCA
        Fitted PCA object.

    Returns
    -------
    np.ndarray
        Transformed data of shape (n_frames, n_components).
    """
    transformed = np.full(
        (data.shape[0], pca.n_components_), np.nan, dtype=np.float32
    )
    valid_mask = ~np.isnan(data).any(axis=1)

    if valid_mask.any():
        transformed[valid_mask] = pca.transform(data[valid_mask])

    return transformed


def load_yaml(path: Path | str) -> dict:
    """
    Load a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    path : Path or str
        Path to the YAML file.

    Returns
    -------
    dict
        Contents of the YAML file as a dictionary.
    """
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_from_memmap(modality_dir: Path | str) -> np.ndarray:
    """
    Load data from memory-mapped file with metadata.

    Parameters
    ----------
    modality_dir : Path or str
        Directory containing data.mem and meta.yml files

    Returns
    -------
    np.ndarray
        Loaded data array
    """
    modality_dir = Path(modality_dir)
    meta = load_yaml(modality_dir / "meta.yml")

    mm_data = np.memmap(
        modality_dir / "data.mem",
        dtype=meta["dtype"],
        mode="r",
        shape=(meta["n_timestamps"], meta["n_signals"]),
    )

    return np.array(mm_data).astype(DEFAULT_DTYPE)


def load(
    data_dir: Path | str,
    session: str,
    modalities: list[str],
    experiment: str,
    poses_config: dict = dict(),
    location_config: dict = dict(),
    kinematics_config: dict = dict(),
    spikes_config: dict = dict(),
    train_intervals: dict = dict(),
) -> dict:
    """
    Load all modalities for a single session based on configuration.

    Parameters
    ----------
    data_dir : Path or str
        Path to the data directory
    session : str
        Session name to load
    modalities : list[str]
        List of modalities to load
    experiment : str
        Experiment name/identifier
    poses_config : dict, optional
        Configuration for poses processing
    location_config : dict, optional
        Configuration for location processing
    kinematics_config : dict, optional
        Configuration for kinematics processing
    spikes_config : dict, optional
        Configuration for spikes processing
    train_intervals : dict, optional
        Training intervals for each session

    Returns
    -------
    dict
        Dictionary with structure: {modality: data}
    """
    data_dir = Path(data_dir)
    experiment_dir = data_dir / experiment
    session_data = {}

    if "poses" in modalities:
        session_data["poses"] = poses.load(
            data_dir,
            session,
            experiment,
            poses_config,
            train_intervals,
        )

    if "location" in modalities:
        representation = location_config.get("representation", "com")
        session_data["location"] = location.load(
            experiment_dir,
            session,
            representation,
        )

    if "speed" in modalities:
        representation = (
            kinematics_config.get("representation", "com_vec")
            if kinematics_config
            else "com_vec"
        )

        if representation.startswith("com_"):
            kinematic_data = poses.load_com(data_dir / experiment, session)
        else:
            kinematic_data = session_data["poses"]

        session_data["speed"] = poses.compute_speed(
            kinematic_data,
            representation,
        )

    if "acceleration" in modalities:
        session_data["acceleration"] = poses.compute_acceleration(
            session_data["speed"],
        )

    if "spike_count" in modalities:
        session_data["spike_count"] = spikes.load(
            data_dir,
            session,
            experiment,
            spikes_config,
            "spike_count",
        )

    if "spikes" in modalities:
        session_data["spikes"] = spikes.load(
            data_dir,
            session,
            experiment,
            spikes_config,
            "spikes",
        )

    return session_data


def load_all(
    data_dir: Path | str,
    sessions: list[str],
    in_modalities: list[str],
    out_modalities: list[str],
    experiment: str,
    poses_config: dict = dict(),
    location_config: dict = dict(),
    kinematics_config: dict = dict(),
    spikes_config: dict = dict(),
    train_intervals: dict = dict(),
) -> dict:
    """
    Load all modalities for all sessions based on configuration.

    Parameters
    ----------
    data_dir : Path or str
        Path to the data directory
    sessions : list[str]
        List of session names to load
    in_modalities : list[str]
        List of input modalities
    out_modalities : list[str]
        List of output modalities
    experiment : str
        Experiment name/identifier
    poses_config : dict, optional
        Configuration for poses processing
    location_config : dict, optional
        Configuration for location processing
    kinematics_config : dict, optional
        Configuration for kinematics processing
    spikes_config : dict, optional
        Configuration for spikes processing
    train_intervals : dict, optional
        Training intervals for each session

    Returns
    -------
    dict
        Nested dictionary with structure: {session: {modality: data}}
    """
    modalities = list(set(in_modalities + out_modalities))
    data = {}

    for session in sessions:
        data[session] = load(
            data_dir,
            session,
            modalities,
            experiment,
            poses_config,
            location_config,
            kinematics_config,
            spikes_config,
            train_intervals,
        )

    return data
