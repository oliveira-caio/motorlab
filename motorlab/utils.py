import random

import numpy as np
import torch

from sklearn.decomposition import PCA

from motorlab.intervals import LabeledInterval


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def compute_tile_distribution(tiles, intervals=None):
    n_of_tiles = 15
    if intervals:
        tiles = np.concatenate([tiles[s:e] for s, e in intervals])
    counts = np.array([(tiles == t).sum() for t in range(n_of_tiles)])
    distr = counts / counts.sum()
    return distr


def list_modalities(modalities):
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


def compute_weights(tiles, intervals):
    # i'm using the poses and the valid intervals instead of the tiles itself because the distribution will probably change considerably when we compare the full session and only the valid time points.
    tile_distr = compute_tile_distribution(tiles, intervals)
    weights = torch.tensor(
        np.where(tile_distr > 0.0, 1.0 - tile_distr, 0.0), dtype=torch.float32
    ).to(device)
    return weights


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_params(model):
    # Core
    core_params = sum(p.numel() for p in model.core.parameters())

    # Embedding per session
    embedding_params = sum(
        sum(p.numel() for p in v.parameters())
        for v in model.embedding.linear.values()
    )

    # Readout per session
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
    # Core (shared across sessions)
    core_param_count = sum(p.numel() for p in model.core.parameters())

    # Embedding and readout per session
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
    intervals: list[LabeledInterval] = None,
    divide_variance: bool = False,
):
    """
    - data: usually a np.array with shape (n_frames, n_features)
    - intervals: if not provided, projects to the PCAs using the entire session. if provided, selects only the frames contained in the intervals and stack them. necessary for training to not leak information, for example. provide only the training intervals in such case.

    important: i fit on the intervals (training) and then use the components to project the entire dataset.
    """
    if intervals:
        data_ = np.concatenate(
            [data[interval.start : interval.end] for interval in intervals],
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
