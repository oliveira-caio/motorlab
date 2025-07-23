from pathlib import Path

import numpy as np


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
