import warnings

from pathlib import Path

import numpy as np
import yaml

from motorlab import utils


def load(
    session_dir: Path | str,
    modality: str,
    sampling_rate: int,
) -> np.ndarray:
    """
    Load spikes or spike count data for a session.

    Parameters
    ----------
    session_dir : Path or str
        Path to the session directory
    modality : str
        Modality to load (e.g., "spikes" or "spike_count")
    sampling_rate : int
        Sampling rate for the data

    Returns
    -------
    np.ndarray
        Modality data array
    """
    session_dir = Path(session_dir)
    modality_dir = session_dir / modality

    with open(modality_dir / "meta.yml", "r") as f:
        meta_data = yaml.safe_load(f)

    if "is_mem_mapped" in meta_data and meta_data["is_mem_mapped"]:
        modality_data = np.memmap(
            modality_dir / "data.mem",
            dtype=meta_data["dtype"],
            mode="r",
            shape=(meta_data["n_timestamps"], meta_data["n_signals"]),
        )
        modality_data = np.array(modality_data).astype(meta_data["dtype"])
    else:
        modality_data = np.load(modality_dir / "data.npy")

    if sampling_rate <= meta_data["sampling_rate"]:
        factor = meta_data["sampling_rate"] // sampling_rate
        duration = meta_data["n_timestamps"] - (
            meta_data["n_timestamps"] % factor
        )
        modality_data = modality_data[:duration].reshape(
            -1, factor, meta_data["n_signals"]
        )
        modality_data = np.nansum(modality_data, axis=1)
    else:
        warnings.warn(
            f"Upsampling not implemented. Original sampling rate is: {meta_data['sampling_rate']}"
        )

    return modality_data.astype(meta_data["dtype"])


def preprocess(spikes_data, spikes_config):
    pass


def extract_area(
    spikes: np.ndarray, area: str, directory: str | Path
) -> np.ndarray:
    """
    Extract spikes for a specific brain area or return all spikes.

    Parameters
    ----------
    spikes : np.ndarray
        Spike data array.
    area : str
        Brain area to extract ('all' for all areas).
    directory : str or Path
        Directory containing meta/areas.npy.

    Returns
    -------
    np.ndarray
        Spikes for the specified area(s).
    """
    if area == "all":
        return spikes

    dir_path = Path(directory)
    areas_path = dir_path / "meta" / "areas.npy"
    with areas_path.open("rb") as f:
        areas_array = np.load(f)
        indices = np.where(
            np.char.find(np.char.lower(areas_array), np.char.lower(area)) >= 0
        )[0]

    return spikes[..., indices]
