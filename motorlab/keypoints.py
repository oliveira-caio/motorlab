SKELETON = {
    "normal": [
        ("s_tail", "e_tail"),
        ("s_tail", "l_hip"),
        ("s_tail", "r_hip"),
        ("r_knee", "r_hip"),
        ("r_knee", "r_ankle"),
        ("l_knee", "l_hip"),
        ("l_knee", "l_ankle"),
        ("r_elbow", "r_shoulder"),
        ("r_elbow", "r_wrist"),
        ("l_elbow", "l_shoulder"),
        ("l_elbow", "l_wrist"),
        ("neck", "s_tail"),
        ("neck", "head"),
        ("neck", "nose"),
        ("neck", "l_shoulder"),
        ("neck", "r_shoulder"),
        ("neck", "l_ear"),
        ("neck", "r_ear"),
        ("nose", "l_eye"),
        ("nose", "r_eye"),
        ("nose", "l_ear"),
        ("nose", "r_ear"),
        ("head", "l_ear"),
        ("head", "r_ear"),
        ("head", "l_eye"),
        ("head", "r_eye"),
    ],
    "reduced": [
        ("s_tail", "e_tail"),
        ("s_tail", "l_hip"),
        ("s_tail", "r_hip"),
        ("r_knee", "r_hip"),
        ("r_knee", "r_ankle"),
        ("l_knee", "l_hip"),
        ("l_knee", "l_ankle"),
        ("r_elbow", "r_shoulder"),
        ("r_elbow", "r_wrist"),
        ("l_elbow", "l_shoulder"),
        ("l_elbow", "l_wrist"),
        ("neck", "s_tail"),
        ("neck", "nose"),
        ("neck", "l_shoulder"),
        ("neck", "r_shoulder"),
        ("neck", "l_ear"),
        ("neck", "r_ear"),
        ("nose", "l_ear"),
        ("nose", "r_ear"),
    ],
    "extended": [
        ("m_tail", "e_tail"),
        ("m_tail", "s_tail"),
        ("s_tail", "spine"),
        ("s_tail", "l_hip"),
        ("s_tail", "r_hip"),
        ("r_upperleg", "r_hip"),
        ("r_upperleg", "r_knee"),
        ("r_lowerleg", "r_knee"),
        ("r_lowerleg", "r_ankle"),
        ("l_upperleg", "l_hip"),
        ("l_upperleg", "l_knee"),
        ("l_lowerleg", "l_knee"),
        ("l_lowerleg", "l_ankle"),
        ("r_upperarm", "r_shoulder"),
        ("r_upperarm", "r_elbow"),
        ("r_lowerarm", "r_elbow"),
        ("r_lowerarm", "r_wrist"),
        ("l_upperarm", "l_shoulder"),
        ("l_upperarm", "l_elbow"),
        ("l_lowerarm", "l_elbow"),
        ("l_lowerarm", "l_wrist"),
        ("neck", "spine"),
        ("neck", "head"),
        ("neck", "nose"),
        ("neck", "l_shoulder"),
        ("neck", "r_shoulder"),
        ("neck", "l_ear"),
        ("neck", "r_ear"),
        ("nose", "l_eye"),
        ("nose", "r_eye"),
        ("nose", "l_ear"),
        ("nose", "r_ear"),
        ("head", "l_ear"),
        ("head", "r_ear"),
        ("head", "l_eye"),
        ("head", "r_eye"),
    ],
}


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

HEAD = [
    "neck",
    "head",
    "l_ear",
    "l_eye",
    "r_eye",
    "r_ear",
    "nose",
]

TRUNK = [
    "neck",
    "l_hip",
    "l_upperleg",
    "l_knee",
    "l_lowerleg",
    "l_ankle",
    "l_shoulder",
    "l_upperarm",
    "l_elbow",
    "l_lowerarm",
    "l_wrist",
    "r_hip",
    "r_upperleg",
    "r_knee",
    "r_lowerleg",
    "r_ankle",
    "r_shoulder",
    "r_upperarm",
    "r_elbow",
    "r_lowerarm",
    "r_wrist",
]


def get(experiment: str) -> dict[str, int]:
    """
    Get the keypoints of a specific type.

    Parameters
    ----------
    experiment : str
        Options: pg, old_gbyk, gbyk

    Returns
    -------
    dict
        Keypoints for the specified experiment.
    """
    return KEYPOINTS[experiment]


def to_idx(keypoint_name: str, experiment: str) -> int:
    """
    Map a keypoint name to its index in the keypoints array.

    Parameters
    ----------
    keypoint_name : str
        Name of the keypoint.
    experiment : str
        Experiment name (for keypoint selection).

    Returns
    -------
    int
        Index of the keypoint in the keypoints array.
    """
    return KEYPOINTS[experiment][keypoint_name]


def get_skeleton(skeleton_type: str) -> list[tuple[str, str]]:
    """
    Get the skeleton connections for a specific experiment.

    Parameters
    ----------
    skeleton_type : str
        Skeleton type (for keypoint selection).

    Returns
    -------
    list
        Skeleton edges for the specified experiment.
    """
    return SKELETON[skeleton_type]


def get_neckless_skeleton(skeleton_type: str) -> list[tuple[str, str]]:
    """
    Return the skeleton with neck connections to head, nose, and ears removed.

    Returns
    -------
    list
        Skeleton edges with specified neck connections removed.
    """
    to_remove = [
        ("neck", "head"),
        ("neck", "nose"),
        ("neck", "l_ear"),
        ("neck", "r_ear"),
    ]
    return [edge for edge in SKELETON[skeleton_type] if edge not in to_remove]


def get_keypoints_for_angles() -> list[tuple[str, str, str]]:
    return [
        ("l_hip", "l_knee", "l_ankle"),
        ("r_hip", "r_knee", "r_ankle"),
        ("l_shoulder", "l_elbow", "l_wrist"),
        ("r_shoulder", "r_elbow", "r_wrist"),
    ]
