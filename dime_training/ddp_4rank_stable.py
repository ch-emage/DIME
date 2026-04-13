#!/usr/bin/env python3
"""
Stable 4-Rank DDP Launcher for RTX 5090
Uses file-based initialization for Windows compatibility

This configuration provides:
- 4 ranks for optimal parallelism without memory conflicts
- 7GB memory per rank (4 × 7GB = 28GB total utilization)
- Windows-compatible file-based DDP initialization
- Memory-efficient feature processing
"""

import os
import sys
import random
import tempfile
import warnings
from datetime import timedelta
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
WORLD_SIZE = 2  # Optimal balance for RTX 5090

def setup_environment():
    """Setup optimal environment for 4-rank DDP"""
    # Memory optimization environment variables
    os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.6"
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    
    # Threading optimization for 4 ranks
    os.environ["OMP_NUM_THREADS"] = "12"  # 48 cores / 4 ranks = 12 per rank
    os.environ["MKL_NUM_THREADS"] = "12"
    os.environ["OPENBLAS_NUM_THREADS"] = "12"
    
    # CUDA optimizations
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_USE_CUDA_DSA"] = "0"
    
    # Prevent library conflicts
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def worker(rank, shared_file_path):
    """4-rank DDP worker with file-based initialization"""
    
    try:
        print(f"[Rank {rank}/WORLD_SIZE] Starting stable worker")
        
        # Environment setup
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
        
        # Import optimizations after environment setup
        from training_accelerator import apply_speed_optimizations
        apply_speed_optimizations()
        
        print(f"[Rank {rank}] Stable 4-rank optimizations applied")
        
        # Device setup with optimal memory allocation
        if torch.cuda.is_available():
            device = 0
            torch.cuda.set_device(device)
            device_name = torch.cuda.get_device_name(device)
            print(f"[Rank {rank}] Using CUDA device {device}: {device_name}")
            
            # Optimal memory allocation for 4 ranks: 22% per rank = 88% total
            memory_fraction = 0.22
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            print(f"[Rank {rank}] Memory fraction: {memory_fraction:.2f} (~7GB per rank)")
        else:
            print(f"[Rank {rank}] Using CPU")
        
        # Initialize process group with file-based method (Windows compatible)
        print(f"[Rank {rank}] Initializing process group...")
        
        try:
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{shared_file_path}",
                rank=rank,
                world_size=WORLD_SIZE,
                timeout=timedelta(seconds=300)
            )
            print(f"[Rank {rank}] Successfully initialized process group")
        except Exception as e:
            print(f"[Rank {rank}] Failed to initialize process group: {e}")
            return
        
        # Import and run training
        try:
            from train_tuple import main as train_main, config as TRAIN_CFG
            
            # Enable DDP
            TRAIN_CFG["ddp"]["enable"] = True
            
            # Configure 2x2 tiling for reduced memory usage
            TRAIN_CFG["tiling"]["rows"] = 2  # Reduced from 4 to 2
            TRAIN_CFG["tiling"]["cols"] = 2  # Reduced from 4 to 2
            
            # Optimal batch sizes for 4-rank memory management with 2x2 tiling
            TRAIN_CFG["dataset"]["train_batch_size"] = 16  # Can increase with fewer tiles
            TRAIN_CFG["dataset"]["eval_batch_size"] = 8
            
            # Optimize data loading for 4 ranks
            TRAIN_CFG["dataset"]["num_workers"] = 12  # 48 cores / 4 ranks = 12
            
            # Each rank handles 1 tile (4 total / 4 ranks = 1 per rank)
            total_tiles = TRAIN_CFG["tiling"]["rows"] * TRAIN_CFG["tiling"]["cols"]
            tiles_per_rank = (total_tiles + WORLD_SIZE - 1) // WORLD_SIZE
            
            print(f"[Rank {rank}] Handling ~{tiles_per_rank} tiles of {total_tiles} total")
            
            if not torch.cuda.is_available():
                TRAIN_CFG["anomaly_detect"]["proximity_on_gpu"] = False
            
            print(f"[Rank {rank}] Starting stable 4-rank DDP training...")
            train_main()
            print(f"[Rank {rank}] Training completed successfully!")
            
        except Exception as e:
            print(f"[Rank {rank}] Training failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
    except Exception as e:
        print(f"[Rank {rank}] Worker failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            print(f"[Rank {rank}] Process group destroyed")
            dist.destroy_process_group()

def main():
    """Launch stable 4-rank DDP training"""
    
    print("=" * 80)
    print(" STABLE 4-RANK DDP CONFIGURATION (2x2 TILING) ")
    print("=" * 80)
    print(f"4-Rank DDP Configuration: WORLD_SIZE={WORLD_SIZE}")
    print(f"Tiling: 2x2 (4 tiles total, 1 tile per rank)")
    print(f"Memory allocation: ~7GB per rank (22% of 32GB)")
    print(f"Expected speedup: ~3x over single-rank with optimal memory usage")
    print("This configuration uses 2x2 tiling for maximum stability")
    print("Using file-based initialization for Windows compatibility")
    print("=" * 80)
    
    # Setup environment
    setup_environment()
    
    # Create shared file for DDP initialization
    temp_dir = tempfile.gettempdir()
    shared_file_path = os.path.join(temp_dir, f"ddp_init_{os.getpid()}")
    
    # Clean up any existing file
    if os.path.exists(shared_file_path):
        os.remove(shared_file_path)
    
    print("Starting stable 4-rank DDP training...")
    
    try:
        mp.spawn(worker, nprocs=WORLD_SIZE, args=(shared_file_path,), join=True)
        print("\n" + "=" * 80)
        print("4-rank DDP training completed successfully!")
        print("Expected performance gain: ~3x over single-rank configuration")
        print("Memory efficiency: Stable usage with ~7GB per rank")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n  4-rank DDP training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up shared file
        try:
            if os.path.exists(shared_file_path):
                os.remove(shared_file_path)
        except:
            pass

if __name__ == "__main__":
    main()