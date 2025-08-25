import warnings

from pathlib import Path

import numpy as np
import yaml


def load(session_dir: Path | str, modality: str, sampling_rate: int):
    session_dir = Path(session_dir)
    modality_dir = session_dir / modality

    with open(modality_dir / "meta.yml", "r") as f:
        meta = yaml.safe_load(f)

    modality_data = np.memmap(
        modality_dir / "data.mem",
        dtype=meta["dtype"],
        mode="r",
        shape=(meta["n_timestamps"], meta["n_signals"]),
    )

    if sampling_rate <= meta["sampling_rate"]:
        factor = meta["sampling_rate"] // sampling_rate
        duration = meta["n_timestamps"] - (meta["n_timestamps"] % factor)
        modality_data = modality_data[:duration].reshape(
            -1, factor, meta["n_signals"]
        )
        modality_data = np.nansum(modality_data, axis=1)
    else:
        warnings.warn(
            f"Upsampling not implemented. Original sampling rate is: {meta['sampling_rate']}"
        )

    return modality_data.astype(meta["dtype"])


def extract_areas(
    data: np.ndarray,
    areas: str | list[str],
    session_dir: str | Path,
) -> np.ndarray:
    """
    Extract spikes for a specific brain area or return all spikes.

    Parameters
    ----------
    spikes : np.ndarray
        Spike data array.
    areas : str | list[str]
        Brain area(s) to extract ('all' for all areas).
    session_dir : str or Path
        Directory containing meta/areas.npy.

    Returns
    -------
    np.ndarray
        Spikes for the specified area(s).
    """
    if areas == "all":
        return data

    if isinstance(areas, str):
        areas = [areas]

    session_dir = Path(session_dir)
    areas_path = session_dir / "spikes" / "meta" / "areas.npy"
    with areas_path.open("rb") as f:
        areas_array = np.load(f)

    indices = []

    for area in areas:
        indices.extend(
            np.where(
                np.char.find(np.char.lower(areas_array), np.char.lower(area))
                >= 0
            )[0]
        )

    return data[..., indices]


def preprocess(
    data: np.ndarray,
    config: dict,
    session_dir: str | Path,
) -> np.ndarray:
    if not config:
        warnings.warn("spikes.preprocess got an empty config")
        return data

    preprocessed_data = data.copy()

    if "brain_area" in config:
        preprocessed_data = extract_areas(
            preprocessed_data, config["brain_area"], session_dir
        )

    return preprocessed_data


def get_areas_array(session_dir: str | Path) -> np.ndarray:
    session_dir = Path(session_dir)
    areas_path = session_dir / "spikes" / "meta" / "areas.npy"
    with areas_path.open("rb") as f:
        areas_array = np.load(f)
    return areas_array
