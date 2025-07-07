from pathlib import Path

import numpy as np


def extract_area(spikes, area, SPK_CNT_DIR):
    if area == "all":
        return spikes

    SPK_CNT_DIR = Path(SPK_CNT_DIR)
    AREAS_PATH = SPK_CNT_DIR / "meta" / "areas.npy"
    with open(AREAS_PATH, "rb") as f:
        areas = np.load(f)
        indices = np.where(
            np.char.find(np.char.lower(areas), np.char.lower(area)) >= 0
        )[0]

    return spikes[..., indices]
