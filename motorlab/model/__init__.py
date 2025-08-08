from .orchestration import train, evaluate
from .core import setup
from .training import (
    iterate,
    iterate_entire_trials,
    loop,
    track,
    format_metrics,
)
from .factory import (
    create,
    load_model,
    load_checkpoint_metadata,
    save_checkpoint,
    compute_mean,
    dump_outputs,
    compute_dimensions,
    register_model,
    get_available_architectures,
)

__all__ = [
    # Main API
    "train",
    "evaluate",
    "setup",
    # Training functions
    "iterate",
    "iterate_entire_trials",
    "loop",
    "track",
    "format_metrics",
    # Factory functions
    "create",
    "load_model",
    "load_checkpoint_metadata",
    "save_checkpoint",
    "compute_mean",
    "compute_dimensions",
    "dump_outputs",
    "register_model",
    "get_available_architectures",
]
