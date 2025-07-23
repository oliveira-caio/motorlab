import random

import numpy as np
import torch

from sklearn.decomposition import PCA


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


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


COM_KEYPOINTS_IDXS = {
    experiment: [
        KEYPOINTS[experiment]["l_hip"],
        KEYPOINTS[experiment]["r_hip"],
        KEYPOINTS[experiment]["l_shoulder"],
        KEYPOINTS[experiment]["r_shoulder"],
        KEYPOINTS[experiment]["neck"],
        KEYPOINTS[experiment]["s_tail"],
    ]
    for experiment in ["gbyk", "pg"]
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
        ["S_tail", "E_tail"],
        ["S_tail", "L_hip"],
        ["S_tail", "R_hip"],
        ["R_knee", "R_hip"],
        ["R_knee", "R_ankle"],
        ["L_knee", "L_hip"],
        ["L_knee", "L_ankle"],
        ["R_elbow", "R_shoulder"],
        ["R_elbow", "R_wrist"],
        ["L_elbow", "L_shoulder"],
        ["L_elbow", "L_wrist"],
        ["neck", "S_tail"],
        ["neck", "head"],
        ["neck", "nose"],
        ["neck", "L_shoulder"],
        ["neck", "R_shoulder"],
        ["neck", "L_ear"],
        ["neck", "R_ear"],
        ["nose", "L_eye"],
        ["nose", "R_eye"],
        ["nose", "L_ear"],
        ["nose", "R_ear"],
        ["head", "L_ear"],
        ["head", "R_ear"],
        ["head", "L_eye"],
        ["head", "R_eye"],
    ],
    "reduced": [
        ["S_tail", "E_tail"],
        ["S_tail", "L_hip"],
        ["S_tail", "R_hip"],
        ["R_knee", "R_hip"],
        ["R_knee", "R_ankle"],
        ["L_knee", "L_hip"],
        ["L_knee", "L_ankle"],
        ["R_elbow", "R_shoulder"],
        ["R_elbow", "R_wrist"],
        ["L_elbow", "L_shoulder"],
        ["L_elbow", "L_wrist"],
        ["neck", "S_tail"],
        ["neck", "nose"],
        ["neck", "L_shoulder"],
        ["neck", "R_shoulder"],
        ["neck", "L_ear"],
        ["neck", "R_ear"],
        ["nose", "L_ear"],
        ["nose", "R_ear"],
    ],
    "extended": [
        ["M_tail", "E_tail"],
        ["M_tail", "S_tail"],
        ["S_tail", "spine"],
        ["S_tail", "L_hip"],
        ["S_tail", "R_hip"],
        ["R_upperLeg", "R_hip"],
        ["R_upperLeg", "R_knee"],
        ["R_lowerLeg", "R_knee"],
        ["R_lowerLeg", "R_ankle"],
        ["L_upperLeg", "L_hip"],
        ["L_upperLeg", "L_knee"],
        ["L_lowerLeg", "L_knee"],
        ["L_lowerLeg", "L_ankle"],
        ["R_upperArm", "R_shoulder"],
        ["R_upperArm", "R_elbow"],
        ["R_lowerArm", "R_elbow"],
        ["R_lowerArm", "R_wrist"],
        ["L_upperArm", "L_shoulder"],
        ["L_upperArm", "L_elbow"],
        ["L_lowerArm", "L_elbow"],
        ["L_lowerArm", "L_wrist"],
        ["neck", "spine"],
        ["neck", "head"],
        ["neck", "nose"],
        ["neck", "L_shoulder"],
        ["neck", "R_shoulder"],
        ["neck", "L_ear"],
        ["neck", "R_ear"],
        ["nose", "L_eye"],
        ["nose", "R_eye"],
        ["nose", "L_ear"],
        ["nose", "R_ear"],
        ["head", "L_ear"],
        ["head", "R_ear"],
        ["head", "L_eye"],
        ["head", "R_eye"],
    ],
}

# todo: fix this
JOINT_ANGLES_IDXS = {
    "gbyk": [
        ["l_shoulder", "neck", "l_elbow"],
        ["r_shoulder", "neck", "r_elbow"],
        ["l_hip", "s_tail", "l_knee"],
        ["r_hip", "s_tail", "r_knee"],
        ["l_elbow", "l_wrist", "l_hand"],
        ["r_elbow", "r_wrist", "r_hand"],
    ],
    "pg": [
        ["l_shoulder", "neck", "l_elbow"],
        ["r_shoulder", "neck", "r_elbow"],
        ["l_hip", "s_tail", "l_knee"],
        ["r_hip", "s_tail", "r_knee"],
        ["l_elbow", "l_wrist", "l_hand"],
        ["r_elbow", "r_wrist", "r_hand"],
    ],
}


def get_neckless_skeleton():
    """
    Return the skeleton with neck connections to head, nose, and ears removed.

    Returns
    -------
    list
        Skeleton edges with specified neck connections removed.
    """
    remove_edges = [
        ["neck", "head"],
        ["neck", "nose"],
        ["neck", "l_ear"],
        ["neck", "r_ear"],
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


def list_modalities(modalities):
    """
    Return a list of modalities based on the input string.

    Parameters
    ----------
    modalities : str
        Modality type or 'all'.

    Returns
    -------
    list
        List of modality strings.

    Raises
    ------
    ValueError
        If an unknown modality is provided.
    """
    if modalities == "all":
        return ["poses", "speed", "acceleration", "spike_count"]
    elif modalities == "spike_count":
        return ["spike_count"]
    elif modalities == "poses":
        return ["poses"]
    elif modalities == "kinematics":
        return ["poses", "speed", "acceleration"]
    elif modalities == "poses_spike_count":
        return ["poses", "spike_count"]
    elif modalities == "position":
        return ["position"]
    else:
        raise ValueError(f"unknown modalities: {modalities}.")


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
    intervals: list[list[int]] | None = None,
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
        data_ = np.concatenate(
            [data[s:e] for s, e in intervals],
            axis=0,
        )
    else:
        data_ = data

    if divide_variance:
        data_ -= data_.mean(axis=0)
        data_ /= data_.std(axis=0)

    pca = PCA(n_components=data_.shape[-1])
    pca.fit(data_)
    data_ = pca.transform(data)
    return data_, pca


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
    lengths = np.array([end - start + 1 for start, end in intervals])
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
