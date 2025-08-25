"""
Distributed training utilities for multi-GPU support.
Handles automatic initialization and configuration of distributed training.
"""

import os

import torch
import torch.distributed as dist

from motorlab import logger


def is_available() -> bool:
    """Check if distributed training is available and beneficial."""
    return torch.cuda.is_available() and torch.cuda.device_count() > 1


def is_initialized() -> bool:
    """Check if distributed training is already initialized."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get the total number of processes (GPUs)."""
    if is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get the rank (ID) of the current process."""
    if is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed() -> bool:
    """
    Set up distributed training using torchrun environment variables.

    Returns
    -------
    bool
        True if distributed training was set up, False otherwise
    """
    if not is_available():
        return False

    if is_initialized():
        return True

    # Check if launched with torchrun (has RANK environment variable)
    if "RANK" in os.environ:
        return _setup_manual_distributed()
    else:
        # Single machine, multiple GPUs available but not launched with torchrun
        if is_main_process():
            run_logger = logger.get()
            if run_logger.handlers:
                run_logger.info(
                    f"Multiple GPUs detected ({torch.cuda.device_count()})"
                )
                run_logger.info(
                    "For multi-GPU training, launch with: torchrun --nproc_per_node={} script.py".format(
                        torch.cuda.device_count()
                    )
                )
        return False


def _setup_manual_distributed() -> bool:
    """Set up distributed training using torchrun environment variables."""
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )

        if is_main_process():
            run_logger = logger.get()
            if run_logger.handlers:
                run_logger.info(
                    f"Distributed training initialized with {world_size} GPUs"
                )
                run_logger.info("Using NCCL backend for communication")

        return True

    except Exception as e:
        run_logger = logger.get()
        if run_logger.handlers:
            run_logger.error(f"Failed to setup distributed training: {e}")
        return False


def cleanup_distributed():
    """Clean up distributed training."""
    if is_initialized():
        dist.destroy_process_group()


def all_reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform all-reduce operation on a tensor across all processes.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to reduce across all processes

    Returns
    -------
    torch.Tensor
        Reduced tensor (averaged across all processes)
    """
    if not is_initialized():
        return tensor

    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
    reduced_tensor /= get_world_size()
    return reduced_tensor


def barrier():
    """Synchronize all processes."""
    if is_initialized():
        dist.barrier()


def broadcast_object(obj, src: int = 0):
    """Broadcast an object from src rank to all other ranks."""
    if not is_initialized():
        return obj

    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]
