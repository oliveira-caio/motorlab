from pathlib import Path

import numpy as np
import yaml

from mymodels import poses, room, utils


def load_from_memmap(MODALITY_DIR):
    MODALITY_DIR = Path(MODALITY_DIR)
    modality = MODALITY_DIR.name

    with open(MODALITY_DIR / "meta.yaml", "r") as f:
        meta = yaml.safe_load(f)
        mm_data = np.memmap(
            MODALITY_DIR / "data.mem",
            dtype=meta["dtype"],
            mode="r",
            shape=(meta["n_timestamps"], meta["n_signals"]),
        )

    if modality == "poses":
        fte = len(mm_data) - (len(mm_data) % 5)
        mm_data = mm_data[:fte].reshape(-1, 5, meta["n_signals"]).mean(axis=1)
    elif modality == "responses":
        fte = len(mm_data) - (len(mm_data) % 50)
        mm_data = mm_data[:fte].reshape(-1, 5, meta["n_signals"]).sum(axis=1)

    return mm_data.astype(np.float32)


def load_all(config):
    DATA_DIR = Path(config["DATA_DIR"])
    in_modalities = utils.list_modalities(config["in_modalities"])
    out_modalities = utils.list_modalities(config["out_modalities"])
    modalities = in_modalities + out_modalities
    data = {session: dict() for session in config["sessions"]}

    if "poses" in modalities:
        poses_raw = dict()
        for session in config["sessions"]:
            POSES_DIR = DATA_DIR / session / "poses"
            poses_raw[session] = load_from_memmap(POSES_DIR)
            data[session]["poses"] = poses.change_repr(
                poses_raw[session], config["experiment"], config["body_repr"]
            )

    if "tiles" in modalities:
        for session in config["sessions"]:
            poses_tiles = (
                poses_raw[session]
                if "poses" in modalities
                else load_from_memmap(DATA_DIR / session / "poses")
            )
            com = poses.compute_com(poses_tiles, config["experiment"])
            data[session]["tiles"] = room.extract_tiles(com)

    if "speed" in modalities:
        for session in config["sessions"]:
            data[session]["speed"] = poses.compute_speed(
                data[session]["poses"], config["experiment"], config["speed_repr"]
            )
            data[session]["acceleration"] = poses.compute_acceleration(
                data[session]["speed"]
            )

    if "spike_count" in modalities:
        for session in config["sessions"]:
            SPK_CNT_DIR = DATA_DIR / session / "spike_count"
            data[session]["spike_count"] = load_from_memmap(SPK_CNT_DIR)

    if "spikes" in modalities:
        for session in config["sessions"]:
            RESP_DIR = DATA_DIR / session / "spikes"
            data[session]["spikes"] = load_from_memmap(RESP_DIR)

    return data
