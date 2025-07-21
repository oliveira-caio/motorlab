from pathlib import Path

import numpy as np


def extract_area(spikes, area, directory):
    if area == "all":
        return spikes

    directory = Path(directory)
    areas_path = directory / "meta" / "areas.npy"
    with open(areas_path, "rb") as f:
        areas = np.load(f)
        indices = np.where(
            np.char.find(np.char.lower(areas), np.char.lower(area)) >= 0
        )[0]

    return spikes[..., indices]
