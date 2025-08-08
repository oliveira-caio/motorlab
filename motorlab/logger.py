"""
Logging utilities for the motorlab package.
"""

import logging
import sys

from pathlib import Path


def create(
    log_dir: Path | str,
    uid: str,
    logger_name: str = "motorlab",
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up a logger that writes to both console and file with buffering.

    Parameters
    ----------
    log_dir : str
        Directory where log files should be saved
    uid : str
        Unique identifier for this run (used in log filename)
    logger_name : str, optional
        Name of the logger, by default "motorlab"
    level : int, optional
        Logging level, by default logging.INFO
    console_output : bool, optional
        Whether to output to console in addition to file, by default True

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    log_file = log_dir / f"{uid}.log"

    file_obj = open(log_file, "w", buffering=8192)
    file_handler = logging.StreamHandler(file_obj)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get(logger_name: str = "motorlab") -> logging.Logger:
    """
    Get an existing logger instance.

    Parameters
    ----------
    logger_name : str, optional
        Name of the logger, by default "motorlab"

    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(logger_name)
