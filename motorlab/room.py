import numpy as np

# room size
X_SIZE = 2.595
Y_SIZE = 4.325

# tiles range from 0 to X_DIVISIONS * Y_DIVISIONS - 1
X_DIVISIONS = 3
Y_DIVISIONS = 5


def get_size() -> tuple[float, float]:
    """Get the size of the room.

    Returns
    -------
    tuple[float, float]
        The width and height of the room.
    """
    return X_SIZE, Y_SIZE


def get_divisions() -> tuple[int, int]:
    """Get the number of divisions in the room.

    Returns
    -------
    tuple[int, int]
        The number of divisions along the x and y axes, respectively.
    """
    return X_DIVISIONS, Y_DIVISIONS


def compute_tiles(
    xs: float | np.ndarray,
    ys: float | np.ndarray,
) -> np.ndarray:
    """
    Compute the tile number for given x and y coordinates in the room.

    Parameters
    ----------
    xs : float or np.ndarray
        xs coordinates (meters).
    ys : float or np.ndarray
        ys coordinates (meters).

    Returns
    -------
    float or np.ndarray
        Tile number(s) corresponding to the input coordinates.
        Returns float for scalar inputs, np.ndarray for array inputs.
    """
    if isinstance(xs, np.ndarray) and isinstance(ys, np.ndarray):
        assert len(xs) == len(ys), "inputs must have the same length"
    elif isinstance(xs, np.ndarray) or isinstance(ys, np.ndarray):
        raise ValueError("inputs must have the same type (float or np.ndarray)")

    tile_width = X_SIZE / X_DIVISIONS
    tile_height = Y_SIZE / Y_DIVISIONS

    col = xs // tile_width
    row = ys // tile_height

    # Clamp the indices to stay within bounds.
    col = np.clip(col, 0, X_DIVISIONS - 1)
    row = np.clip(row, 0, Y_DIVISIONS - 1)

    # Tiles are numbered left-to-right, bottom-to-top.
    tile_number = row * X_DIVISIONS + col

    return tile_number
