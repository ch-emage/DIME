# anomaly_engine/distrib/dist_utils.py
import os
import socket
from datetime import timedelta
import torch
import torch.distributed as dist

def env_default_port() -> str:
    # Pick a stable fallback port
    return os.environ.get("MASTER_PORT", "29500")

def init_distributed(backend: str = "gloo", timeout_sec: int = 1800):
    """
    Safe DDP init that works on a SINGLE GPU. All ranks bind to cuda:0.
    Use torchrun --nproc_per_node=R ... with CUDA_VISIBLE_DEVICES=0.
    """
    if dist.is_initialized():
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", env_default_port())
    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

    # torchrun sets these:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Force all ranks to the SAME GPU (index 0)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=timeout_sec),
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    )

def get_rank_world():
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()

def is_main_process():
    r, _ = get_rank_world()
    return r == 0

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def gather_object_all(obj):
    """
    Gather a Python object from all ranks to all ranks.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    out = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(out, obj)
    return out
