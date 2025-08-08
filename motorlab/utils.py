import random

from pathlib import Path

import numpy as np
import torch
import yaml

from sklearn.decomposition import PCA

from motorlab import poses, room, spikes, plot


DEFAULT_DTYPE = np.float32

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

REPRESENTATIONS = [
    "allocentric",
    "centered",
    "egocentric",
    "trunk",
    "head",
    "pc",
    "loose",
    "medium",
    "strict",
    "draconian",
]

PCS_PATH = Path("artifacts/tables/analysis_pca")

KEYPOINTS = {
    "gbyk": {
        "e_tail": 0,
        "l_ankle": 1,
        "l_ear": 2,
        "l_elbow": 3,
        "l_eye": 4,
        "l_hip": 5,
        "l_knee": 6,
        "l_shoulder": 7,
        "l_wrist": 8,
        "r_ankle": 9,
        "r_ear": 10,
        "r_elbow": 11,
        "r_eye": 12,
        "r_hip": 13,
        "r_knee": 14,
        "r_shoulder": 15,
        "r_wrist": 16,
        "s_tail": 17,
        "head": 18,
        "neck": 19,
        "nose": 20,
    },
    "old_gbyk": {
        "l_wrist": 0,
        "l_elbow": 1,
        "l_shoulder": 2,
        "r_wrist": 3,
        "r_elbow": 4,
        "r_shoulder": 5,
        "l_ankle": 6,
        "l_knee": 7,
        "l_hip": 8,
        "r_ankle": 9,
        "r_knee": 10,
        "r_hip": 11,
        "e_tail": 12,
        "s_tail": 13,
        "neck": 14,
        "head": 15,
        "l_ear": 16,
        "r_ear": 17,
        "l_eye": 18,
        "r_eye": 19,
        "nose": 20,
    },
    "pg": {
        "neck": 0,
        "spine": 1,
        "head": 2,
        "l_ear": 3,
        "r_ear": 4,
        "l_eye": 5,
        "r_eye": 6,
        "nose": 7,
        "l_shoulder": 8,
        "l_elbow": 9,
        "l_wrist": 10,
        "l_upperarm": 11,
        "l_lowerarm": 12,
        "r_shoulder": 13,
        "r_elbow": 14,
        "r_wrist": 15,
        "r_upperarm": 16,
        "r_lowerarm": 17,
        "l_hip": 18,
        "l_knee": 19,
        "l_ankle": 20,
        "l_upperleg": 21,
        "l_lowerleg": 22,
        "r_hip": 23,
        "r_knee": 24,
        "r_ankle": 25,
        "r_upperleg": 26,
        "r_lowerleg": 27,
        "s_tail": 28,
        "m_tail": 29,
        "e_tail": 30,
    },
}


EXTRA_KEYPOINTS_IDXS = {
    "gbyk": [
        KEYPOINTS["gbyk"]["l_eye"],
        KEYPOINTS["gbyk"]["r_eye"],
        KEYPOINTS["gbyk"]["head"],
    ],
    "pg": [
        KEYPOINTS["pg"]["spine"],
        KEYPOINTS["pg"]["head"],
        KEYPOINTS["pg"]["l_eye"],
        KEYPOINTS["pg"]["r_eye"],
        KEYPOINTS["pg"]["l_upperarm"],
        KEYPOINTS["pg"]["l_lowerarm"],
        KEYPOINTS["pg"]["r_upperarm"],
        KEYPOINTS["pg"]["r_lowerarm"],
        KEYPOINTS["pg"]["l_upperleg"],
        KEYPOINTS["pg"]["l_lowerleg"],
        KEYPOINTS["pg"]["r_upperleg"],
        KEYPOINTS["pg"]["r_lowerleg"],
        KEYPOINTS["pg"]["m_tail"],
    ],
}

SKELETON = {
    "normal": [
        ("s_tail", "e_tail"),
        ("s_tail", "l_hip"),
        ("s_tail", "r_hip"),
        ("r_knee", "r_hip"),
        ("r_knee", "r_ankle"),
        ("l_knee", "l_hip"),
        ("l_knee", "l_ankle"),
        ("r_elbow", "r_shoulder"),
        ("r_elbow", "r_wrist"),
        ("l_elbow", "l_shoulder"),
        ("l_elbow", "l_wrist"),
        ("neck", "s_tail"),
        ("neck", "head"),
        ("neck", "nose"),
        ("neck", "l_shoulder"),
        ("neck", "r_shoulder"),
        ("neck", "l_ear"),
        ("neck", "r_ear"),
        ("nose", "l_eye"),
        ("nose", "r_eye"),
        ("nose", "l_ear"),
        ("nose", "r_ear"),
        ("head", "l_ear"),
        ("head", "r_ear"),
        ("head", "l_eye"),
        ("head", "r_eye"),
    ],
    "reduced": [
        ("s_tail", "e_tail"),
        ("s_tail", "l_hip"),
        ("s_tail", "r_hip"),
        ("r_knee", "r_hip"),
        ("r_knee", "r_ankle"),
        ("l_knee", "l_hip"),
        ("l_knee", "l_ankle"),
        ("r_elbow", "r_shoulder"),
        ("r_elbow", "r_wrist"),
        ("l_elbow", "l_shoulder"),
        ("l_elbow", "l_wrist"),
        ("neck", "s_tail"),
        ("neck", "nose"),
        ("neck", "l_shoulder"),
        ("neck", "r_shoulder"),
        ("neck", "l_ear"),
        ("neck", "r_ear"),
        ("nose", "l_ear"),
        ("nose", "r_ear"),
    ],
    "extended": [
        ("m_tail", "e_tail"),
        ("m_tail", "s_tail"),
        ("s_tail", "spine"),
        ("s_tail", "l_hip"),
        ("s_tail", "r_hip"),
        ("r_upperleg", "r_hip"),
        ("r_upperleg", "r_knee"),
        ("r_lowerleg", "r_knee"),
        ("r_lowerleg", "r_ankle"),
        ("l_upperleg", "l_hip"),
        ("l_upperleg", "l_knee"),
        ("l_lowerleg", "l_knee"),
        ("l_lowerleg", "l_ankle"),
        ("r_upperarm", "r_shoulder"),
        ("r_upperarm", "r_elbow"),
        ("r_lowerarm", "r_elbow"),
        ("r_lowerarm", "r_wrist"),
        ("l_upperarm", "l_shoulder"),
        ("l_upperarm", "l_elbow"),
        ("l_lowerarm", "l_elbow"),
        ("l_lowerarm", "l_wrist"),
        ("neck", "spine"),
        ("neck", "head"),
        ("neck", "nose"),
        ("neck", "l_shoulder"),
        ("neck", "r_shoulder"),
        ("neck", "l_ear"),
        ("neck", "r_ear"),
        ("nose", "l_eye"),
        ("nose", "r_eye"),
        ("nose", "l_ear"),
        ("nose", "r_ear"),
        ("head", "l_ear"),
        ("head", "r_ear"),
        ("head", "l_eye"),
        ("head", "r_eye"),
    ],
}

# todo: fix this
JOINT_ANGLES_IDXS = {
    "gbyk": [
        ("l_shoulder", "neck", "l_elbow"),
        ("r_shoulder", "neck", "r_elbow"),
        ("l_hip", "s_tail", "l_knee"),
        ("r_hip", "s_tail", "r_knee"),
        ("l_elbow", "l_wrist", "l_hand"),
        ("r_elbow", "r_wrist", "r_hand"),
    ],
    "pg": [
        ("l_shoulder", "neck", "l_elbow"),
        ("r_shoulder", "neck", "r_elbow"),
        ("l_hip", "s_tail", "l_knee"),
        ("r_hip", "s_tail", "r_knee"),
        ("l_elbow", "l_wrist", "l_hand"),
        ("r_elbow", "r_wrist", "r_hand"),
    ],
}


def indices_com_keypoints(experiment: str) -> list[int]:
    """
    Get the indices of keypoints for the specified experiment.

    Parameters
    ----------
    experiment : str
        Name of the experiment (e.g., "gbyk", "old_gbyk", "pg").

    Returns
    -------
    list[int]
        List of keypoint indices for the specified experiment.
    """
    com_keypoints = [
        "l_hip",
        "r_hip",
        "l_shoulder",
        "r_shoulder",
        "neck",
        "s_tail",
    ]
    return [KEYPOINTS[experiment][key] for key in com_keypoints]


def get_neckless_skeleton():
    """
    Return the skeleton with neck connections to head, nose, and ears removed.

    Returns
    -------
    list
        Skeleton edges with specified neck connections removed.
    """
    remove_edges = [
        ("neck", "head"),
        ("neck", "nose"),
        ("neck", "l_ear"),
        ("neck", "r_ear"),
    ]
    return [edge for edge in SKELETON if edge not in remove_edges]


def compute_tile_distribution(tiles, intervals=None):
    """
    Compute the normalized distribution of tile values.

    Parameters
    ----------
    tiles : np.ndarray
        Array of tile indices.
    intervals : list, optional
        List of (start, end) intervals to select tiles. Default is None.

    Returns
    -------
    np.ndarray
        Normalized distribution of tile values.
    """
    n_tiles = 15
    if intervals:
        tiles = np.concatenate([tiles[s:e] for s, e in intervals])
    counts = np.array([(tiles == t).sum() for t in range(n_tiles)])
    distr = counts / counts.sum()
    return distr


def fix_seed(seed: int = 0) -> None:
    """
    Fix random seed for reproducibility across random, numpy, and torch.

    Parameters
    ----------
    seed : int, optional
        Random seed value. Default is 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_params(model):
    """
    Count the number of parameters in the model core, embedding, and readout.

    Parameters
    ----------
    model : object
        Model object with core, embedding, and readout attributes.

    Returns
    -------
    dict
        Number of parameters for 'core', 'embedding', and 'readout'.
    """
    core_params = sum(p.numel() for p in model.core.parameters())
    embedding_params = sum(
        sum(p.numel() for p in v.parameters())
        for v in model.embedding.linear.values()
    )
    readout_params = sum(
        sum(p.numel() for p in v.parameters())
        for v in model.readout.linear.values()
    )
    return {
        "core": core_params,
        "embedding": embedding_params,
        "readout": readout_params,
    }


def params_per_session(model):
    """
    Compute the percentage and total number of parameters per session.

    Parameters
    ----------
    model : object
        Model object with core, embedding, and readout attributes.

    Returns
    -------
    dict
        For each session, percentage and total number of parameters for embedding, core, and readout.
    """
    core_param_count = sum(p.numel() for p in model.core.parameters())
    session_names = model.embedding.linear.keys()
    result = {}
    for session in session_names:
        emb_count = sum(
            p.numel() for p in model.embedding.linear[session].parameters()
        )
        read_count = sum(
            p.numel() for p in model.readout.linear[session].parameters()
        )
        total = emb_count + core_param_count + read_count
        result[session] = {
            "embedding": 100 * emb_count / total,
            "core": 100 * core_param_count / total,
            "readout": 100 * read_count / total,
            "total_params": total,
        }
    return result


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


def align_intervals(
    data: np.ndarray,
    intervals: list[list[int]],
    overlap: int = 10,
    method: str = "end",
) -> np.ndarray:
    """
    Align intervals of data to a common length, either by start or end.

    This function takes a data array of shape (n_frames, n_features) and a list
    of intervals, where each interval is a (start, end) pair. It returns a new
    array of shape (n_intervals, max_len, n_features), where each interval is
    aligned either by its start or end.

    If aligning by 'start', all intervals are assumed to start at the same
    time point, and the array is padded with NaNs at the end as needed.
    If aligning by 'end', all intervals are assumed to end at the same time
    point, and the array is padded with NaNs at the beginning.

    To determine max_len, the function finds the maximum interval length that
    is shared by at least `overlap + 1` intervals. This strategy reduces
    variance when averaging over intervals.

    Parameters
    ----------
    data : np.ndarray
        Data array of shape (n_frames, n_features).
    intervals : list of list of int
        List of (start, end) intervals.
    overlap : int, optional
        Minimum number of intervals with the same length to consider for max_len. Default is 10.
    method : {'start', 'end'}, optional
        'start' to align by start, 'end' to align by end. Default is 'end'.

    Returns
    -------
    np.ndarray
        Aligned data of shape (n_intervals, max_len, n_features) with NaNs for padding.
    """
    lengths = [e - s + 1 for s, e in intervals]
    lengths = np.array(lengths)
    max_len = max(
        [
            length
            for length in lengths
            if np.sum(lengths >= length) >= (overlap + 1)
        ]
    )

    aligned_data = np.full((len(intervals), max_len, data.shape[-1]), np.nan)
    for i, (start, end) in enumerate(intervals):
        if method == "start":
            aligned_data[i, : end - start + 1] = (
                data[start : end + 1]
                if end - start + 1 <= max_len
                else data[start : start + max_len]
            )
        elif method == "end":
            aligned_data[i, -(end - start + 1) :] = (
                data[start : end + 1]
                if end - start + 1 <= max_len
                else data[end - max_len + 1 : end + 1]
            )

    return aligned_data


def load_pcs_to_exclude(name: str) -> dict[str, list[int]]:
    with open(PCS_PATH / f"{name}.yml") as f:
        return yaml.safe_load(f)


def setup_representation(
    config: dict,
    representation: str,
    old_gbyk: bool,
) -> None:
    """Set up the representation for poses."""
    prefix = "old_" if old_gbyk else ""

    config["poses"]["representation"] = representation
    if representation == "trunk":
        config["poses"]["keypoints_to_exclude"] = [
            "e_tail",
            "head",
            "l_ear",
            "l_eye",
            "r_ear",
            "r_eye",
            "nose",
        ]
    elif representation == "head":
        config["poses"]["keypoints_to_exclude"] = [
            "e_tail",
            "s_tail",
            "l_hip",
            "l_knee",
            "l_ankle",
            "r_hip",
            "r_knee",
            "r_ankle",
            "l_shoulder",
            "l_elbow",
            "l_wrist",
            "r_shoulder",
            "r_elbow",
            "r_wrist",
        ]
    elif representation == "pc":
        config["poses"]["project_to_pca"] = True
    elif representation == "loose":
        config["poses"]["project_to_pca"] = True
        config["poses"]["pcs_to_exclude"] = load_pcs_to_exclude(
            prefix + "loose"
        )
    elif representation == "medium":
        config["poses"]["project_to_pca"] = True
        config["poses"]["pcs_to_exclude"] = load_pcs_to_exclude(
            prefix + "medium"
        )
    elif representation == "strict":
        config["poses"]["project_to_pca"] = True
        config["poses"]["pcs_to_exclude"] = load_pcs_to_exclude(
            prefix + "strict"
        )
    elif representation == "draconian":
        config["poses"]["project_to_pca"] = True
        config["poses"]["pcs_to_exclude"] = load_pcs_to_exclude(
            prefix + "draconian"
        )
    else:
        raise ValueError(f"Unknown representation: {representation}")


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
    meta_file = modality_dir / "meta.yml"

    with meta_file.open("r") as f:
        meta = yaml.safe_load(f)

    mm_data = np.memmap(
        modality_dir / "data.mem",
        dtype=meta["dtype"],
        mode="r",
        shape=(meta["n_timestamps"], meta["n_signals"]),
    )

    return np.array(mm_data).astype(DEFAULT_DTYPE)


def load_data(
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
        session_data["location"] = room.load_location(
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


def load_all_data(
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
        data[session] = load_data(
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
