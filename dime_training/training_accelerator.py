# training_accelerator.py - Advanced training optimizations for RTX 5090
import os
import torch
import warnings

def get_optimal_training_config():
    """Get optimal training configuration for RTX 5090"""
    
    # Get GPU memory info
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Memory: {gpu_memory_gb:.1f} GB")
        
        # RTX 5090 has 32GB VRAM - we can be aggressive with batch sizes
        if gpu_memory_gb > 20:  # RTX 5090 or similar high-end GPU
            return {
                "batch_size": 8,  # Increase from 2 to 8
                "num_workers": 8,  # Increase from 2 to 8
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 4,
                "mixed_precision": True,
                "compile_model": True,
                "gradient_accumulation_steps": 2
            }
        else:
            return {
                "batch_size": 4,
                "num_workers": 6, 
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 2,
                "mixed_precision": True,
                "compile_model": False,
                "gradient_accumulation_steps": 1
            }
    else:
        return {
            "batch_size": 2,
            "num_workers": 4,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "mixed_precision": False,
            "compile_model": False,
            "gradient_accumulation_steps": 1
        }

def apply_speed_optimizations():
    """Apply comprehensive speed optimizations"""
    
    # Advanced memory optimizations for RTX 5090
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.6,expandable_segments:True"
    
    # Enhanced threading for high-end systems
    os.environ["OMP_NUM_THREADS"] = "8"  # Increase for RTX 5090 system
    os.environ["MKL_NUM_THREADS"] = "8"
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
    
    # CUDA optimizations for maximum performance
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_USE_CUDA_DSA"] = "0"  # Disable for speed
    os.environ["CUDA_CACHE_DISABLE"] = "0"
    
    # Set optimal CUDA settings
    if torch.cuda.is_available():
        # Enable all performance features
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        
        # Enable TF32 for RTX 5090 (massive speedup)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable optimized attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except:
            pass
        
        # Set memory fraction conservatively for multi-rank DDP
        # Don't set here - let launcher manage per-rank allocation
        # torch.cuda.set_per_process_memory_fraction(0.95)
        
        print("Applied RTX 5090 optimizations:")
        print("- TF32 enabled for 1.5-2x speedup")
        print("- CuDNN benchmark enabled")
        print("- Flash attention enabled")
        print("- Memory optimizations applied")

def optimize_dataloader_args(base_config):
    """Get optimized DataLoader arguments"""
    optimal_config = get_optimal_training_config()
    
    return {
        "batch_size": optimal_config["batch_size"],
        "num_workers": optimal_config["num_workers"],
        "pin_memory": optimal_config["pin_memory"],
        "persistent_workers": optimal_config["persistent_workers"],
        "prefetch_factor": optimal_config["prefetch_factor"],
        "drop_last": True,  # For better DDP performance
        "shuffle": True,
    }

def setup_mixed_precision():
    """Setup mixed precision training"""
    optimal_config = get_optimal_training_config()
    
    if optimal_config["mixed_precision"] and torch.cuda.is_available():
        print("Mixed precision training enabled - expect 30-50% speedup")
        return True
    return False

if __name__ == "__main__":
    apply_speed_optimizations()
    config = get_optimal_training_config()
    print("Optimal training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")