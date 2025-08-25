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
