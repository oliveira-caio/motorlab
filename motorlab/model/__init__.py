# Import main API functions
from .core import train, evaluate, setup

# Import training functions
from .training import iterate, iterate_entire_trials, loop, track, format_metrics

# Import factory functions  
from .factory import create, load, save_checkpoint, save_config, compute_mean, dump_outputs

# Keep the same API - these are the main entry points
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
    "load", 
    "save_checkpoint", 
    "save_config", 
    "compute_mean", 
    "dump_outputs"
]
