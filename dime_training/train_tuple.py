import os
# Robust defaults for Windows rendezvous
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

import contextlib
import csv
import json
import logging
import pickle
import glob
import time
from typing import List, Tuple, Iterable, Union, Dict, Any, Optional
from collections import defaultdict
from time import perf_counter
import faiss
import cv2
import numpy as np
from PIL import Image
import pathlib
import torch
import torch.distributed as dist
from torch.amp import autocast
import torch.backends.cudnn as cudnn  # [PERF] fast kernels / autotune
import argparse
import shutil
from anomaly_engine.core.motion_utils import MotionStats, MotionSpeedAnalyzer, extract_motion_features_with_speed

# from training_accelerator import apply_speed_optimizations
# apply_speed_optimizations()

# [PERF] enable fast kernels + TF32 on Ampere+ (no logic change)
cudnn.benchmark = True
# try:
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     if hasattr(torch, "set_float32_matmul_precision"):
#         torch.set_float32_matmul_precision("high")
# except Exception:
#     pass

# ---- [INTEGRATION] import your external DDP + sharding helpers ----
# All ranks will bind to cuda:0 (single-GPU DDP).
from anomaly_engine.distrib.dist_utils import (
    init_distributed as ext_init_distributed,
    get_rank_world as ext_get_rank_world,
    is_main_process as ext_is_main_process,
    barrier as ext_barrier,
    gather_object_all as ext_gather_object_all,  # kept available if you want it
)
from anomaly_engine.tiling.position_shard import shard_positions as ext_shard_positions

# ---- Anomaly-Engine imports ----
import anomaly_engine.core
import anomaly_engine.core.core_utils
import anomaly_engine.datasets
import anomaly_engine.models
import anomaly_engine.selectors
from anomaly_engine.core.anomaly_net import AnomalyNet as Anomaly_net_eng
from anomaly_engine.core.position_models import PositionAwareAnomalyNet
from datetime import datetime
import psutil
import GPUtil
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass


def update_existing_model(model_path, new_data_loader, config):
    """
    Update an existing model with new data
    """
    from anomaly_engine.core.anomaly_net import AnomalyNet
    from anomaly_engine.core.core_utils import ApproximateProximitySearcher
    
    device = anomaly_engine.core.core_utils.set_torch_device(config["gpu"])
    
    try:
        # Load existing model
        anomaly_net = AnomalyNet(device)
        anomaly_net.load_from_path(
            model_path, 
            device, 
            nn_method=ApproximateProximitySearcher(False, 4)
        )
        
        # Update with new data
        success = anomaly_net.update_with_new_frames(new_data_loader, update_percentage=0.15)
        
        if success:
            # Save updated model
            updated_path = os.path.join(model_path, "updated")
            anomaly_net.save_to_path(updated_path)
            LOGGER.info(f"Model updated successfully at: {updated_path}")
            return True
            
    except Exception as e:
        LOGGER.error(f"Failed to update model: {e}")
        
    return False

# ---------------------------------------------
# Configuration
# ---------------------------------------------
config = {
    "results_path": "MODEL/",
    "gpu": [0],
    "seed": 0,
    "log_group": "SLC",
    "log_project": "CAMERA_1",
    "save_anomaly_maps": False,
    "save_model": True,
    "dataset": {
        "name": "anomaly",
        "data_path": "data",
        "subdatasets": [ "blister"],
        "train_val_split": 0.8,
        "batch_size": 4, # default 1
        "num_workers": 1,
        "resize": None,   # (H, W) of original frame
        "imagesize": None,      # auto-computed from tiler.compute_tile_coords_hw
        "augment": True,
        "video_to_frames": True,
        "frame_interval": 1,    # Fixed frame interval issue
        "roi_selection_mode": True,
        "enable_object_detection": False,
        "object_detection_roi": None,
    },
    "selector": {
        "name": "approx_greedy_coreset",
        "percentage": 0.02
    },
    "anomaly_detect": {
        "backbone_names": ["wideresnet50"],
        "layers_to_extract_from": ["layer2","layer3"],
        "pretrain_embed_dimension": 512,
        "target_embed_dimension": 512,
        "preprocessing": "mean",
        "aggregation": "mean",
        "anomaly_scorer_num_nn": 2,
        "feature_window": 1,
        "window_step": 1,
        "window_score": "mean",
        "window_overlap": 0.0,
        "window_aggregate": [],
        "proximity_on_gpu": False,
        "proximity_num_workers": 12
    },
    "motion_detection": {
        "enable": False,
        "method": "farneback",
        "save_motion_maps": True,
        "flow_threshold": 0,
        "motion_embed_dim": 128,
        "speed_analysis": True,           # NEW: Enable speed analysis
        "speed_threshold": 0.3,           # NEW: 30% slowdown threshold
        "roi_for_motion": None,           # NEW: ROI for conveyor belt (optional)
        "motion_weight": 0.3,             # NEW: Weight in final anomaly score
        "min_speed_samples": 10,          # NEW: Min samples for baseline
        "speed_history_size": 100         # NEW: History buffer size
    },
    "sequence_model": {
        "enable": False,
        "model_type": "lstm",
        "sequence_length": 16,
        "hidden_size": 256,
        "num_layers": 2,
        "reconstruction_loss_weight": 0.5
    },
    "object_detection": {
        "enable": False,
        "model": "yolov8n",
        "confidence_threshold": 0.3,
        "roi_based": True,
        "roi_config": {
            "interactive_selection": True,
            "save_roi": True,
            "roi_expansion_factor": 0.2
        }
    },
    # PatchCore-style flags
    "ddp": {
        "enable": True,     # enable distributed for tiled train + inference
        "backend": "gloo",  # 'gloo' on Windows/CPU, 'nccl' on Linux+CUDA
    },
    "tiling": {
        "enable": True,
        "rows": 1,
        "cols": 1,
        "overlap": 0.10,     # 0..1 (fraction of base tile)
        "hann_blend": True,
        "save_tile_outputs": False,
    },
    "position_models": {
        "enable": True,
        "models_root": "",
    }
}

# ---------------------------------------------
# Logging
# ---------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)

_DATASETS = {"anomaly": ["anomaly_engine.datasets.dataset_loader", "VideoAnomalyDataset"]}


motion_stats = MotionStats()


# ---------------- CLI: build parser for ALL config fields ----------------
def build_arg_parser():
    p = argparse.ArgumentParser("Tile/DDP trainer (full CLI)")

    # ===== top-level =====
    p.add_argument("--results_path", type=str)
    p.add_argument("--gpu", type=str, help='GPU indices (e.g. "0,1")')
    p.add_argument("--seed", type=int)
    p.add_argument("--log_group", type=str)
    p.add_argument("--log_project", type=str)
    p.add_argument("--save-anomaly-maps", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--roi-count", type=int)

    # ===== DDP =====
    p.add_argument("--ddp", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable distributed")
    p.add_argument("--ddp-backend", type=str, choices=["nccl", "gloo"])
    p.add_argument("--ddp-init-method", type=str, help='e.g. "file:///... or tcp://ip:port"')
    p.add_argument("--world-size", type=int)
    p.add_argument("--rank", type=int)
    p.add_argument("--local-rank", type=int)

    # ===== Tiling =====
    p.add_argument("--tiling-rows", type=int)
    p.add_argument("--tiling-cols", type=int)
    p.add_argument("--tiling-overlap", type=float)

    # ===== Dataset =====
    p.add_argument("--dataset-name", type=str)
    p.add_argument("--data_path", type=str)
    p.add_argument("--subdatasets", type=str, help='Comma/space separated: "a,b" or "a b"')
    p.add_argument("--train-val-split", type=float)
    p.add_argument("--batch_size", type=int)          # training batch size
    p.add_argument("--num_workers", type=int)
    p.add_argument("--resize", type=str, help='H,W (e.g. "1024,1024")')
    p.add_argument("--imagesize", type=str, help='H,W (e.g. "1024,1024")')
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--video-to-frames", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--frame_interval", type=int)

    # ===== Selector =====
    p.add_argument("--selector-name", type=str, choices=["identity", "greedy_coreset", "approx_greedy_coreset"])
    p.add_argument("--selector-percentage", type=float)

    # ===== Anomaly detection =====
    p.add_argument("--backbone-names", type=str, help='Comma/space separated')
    p.add_argument("--layers-to-extract-from", type=str, help='Comma/space separated (e.g. "layer2,layer3")')
    p.add_argument("--pretrain-embed-dimension", type=int)
    p.add_argument("--target-embed-dimension", type=int)
    p.add_argument("--preprocessing", type=str)
    p.add_argument("--aggregation", type=str)
    p.add_argument("--anomaly-scorer-num-nn", type=int)
    p.add_argument("--feature-window", type=int)
    p.add_argument("--window-step", type=int)
    p.add_argument("--window-score", type=str)
    p.add_argument("--window-overlap", type=float)
    p.add_argument("--window-aggregate", type=str)
    p.add_argument("--proximity-on-gpu", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--proximity-num-workers", type=int)

    # ===== Motion detection =====
    p.add_argument("--motion-enable", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--motion-method", type=str)
    p.add_argument("--motion-save-motion-maps", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--motion-flow-threshold", type=float)
    p.add_argument("--motion-embed-dim", type=int)
    p.add_argument("--speed-analysis", action=argparse.BooleanOptionalAction, default=None)  # NEW
    p.add_argument("--speed-threshold", type=float)  # NEW
    p.add_argument("--motion-weight", type=float)  # NEW
    p.add_argument("--min-speed-samples", type=int)  # NEW
    p.add_argument("--speed-history-size", type=int)  # NEW

    # ===== Sequence model =====
    p.add_argument("--sequence-enable", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--sequence-model-type", type=str)
    p.add_argument("--sequence-length", type=int)
    p.add_argument("--sequence-hidden-size", type=int)
    p.add_argument("--sequence-num-layers", type=int)
    p.add_argument("--sequence-reconstruction-loss-weight", type=float)

    # ===== Config file (optional) =====
    p.add_argument("--config-file", type=str, help="JSON config to merge (then CLI overrides)")

    return p


# ---------------- helpers ----------------
def _split_list(s):
    if s is None: return None
    parts = [p for chunk in s.split(",") for p in chunk.split()]
    return [p for p in (p.strip() for p in parts) if p]

def _split_ints(s):
    vals = _split_list(s)
    return None if vals is None else [int(x) for x in vals]

def _parse_hw(s):
    if not s: return None
    h, w = [int(x) for x in _split_list(s)]
    if len([h, w]) != 2:
        raise ValueError("--imagesize/--resize must be exactly two integers (H,W)")
    return [h, w]


# ---------------- merge: defaults → file → CLI ----------------
def update_config_from_args(cfg, args):
    # 1) optional config file as baseline
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, "r") as f:
            file_cfg = json.load(f)
        def deep_update(dst, src):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    deep_update(dst[k], v)
                else:
                    dst[k] = v
        deep_update(cfg, file_cfg)

    # 2) CLI overrides
    # top-level
    if args.results_path is not None: cfg["results_path"] = args.results_path
    if args.gpu is not None:          cfg["gpu"] = [int(x) for x in _split_list(args.gpu)]
    if args.seed is not None:         cfg["seed"] = args.seed
    if args.log_group is not None:    cfg["log_group"] = args.log_group
    if args.log_project is not None:  cfg["log_project"] = args.log_project
    if args.save_anomaly_maps is not None: cfg["save_anomaly_maps"] = args.save_anomaly_maps
    if args.save_model is not None:        cfg["save_model"] = args.save_model
    if args.roi_count is not None:         cfg["roi_count"] = args.roi_count

    # ddp
    ddp = cfg.get("ddp", {})
    if args.ddp is not None:            ddp["enable"] = args.ddp
    if args.ddp_backend is not None:    ddp["backend"] = args.ddp_backend
    if args.ddp_init_method is not None:ddp["init_method"] = args.ddp_init_method
    if args.world_size is not None:     ddp["world_size"] = args.world_size
    if args.rank is not None:           ddp["rank"] = args.rank
    if args.local_rank is not None:     ddp["local_rank"] = args.local_rank
    cfg["ddp"] = ddp

    # tiling
    til = cfg.get("tiling", {})
    if args.tiling_rows is not None:    til["rows"] = args.tiling_rows
    if args.tiling_cols is not None:    til["cols"] = args.tiling_cols
    if args.tiling_overlap is not None: til["overlap"] = args.tiling_overlap
    cfg["tiling"] = til

    # dataset
    ds = cfg.get("dataset", {})
    if args.dataset_name is not None:   ds["name"] = args.dataset_name
    if args.data_path is not None:      ds["data_path"] = args.data_path
    if args.subdatasets is not None:    ds["subdatasets"] = _split_list(args.subdatasets)
    if args.train_val_split is not None:ds["train_val_split"] = args.train_val_split
    if args.batch_size is not None:     ds["batch_size"] = args.batch_size
    if args.num_workers is not None:    ds["num_workers"] = args.num_workers
    if args.resize is not None:         ds["resize"] = _parse_hw(args.resize)
    if args.imagesize is not None:      ds["imagesize"] = _parse_hw(args.imagesize)
    if args.augment is not None:        ds["augment"] = args.augment
    if args.video_to_frames is not None:ds["video_to_frames"] = args.video_to_frames
    if args.frame_interval is not None: ds["frame_interval"] = args.frame_interval
    cfg["dataset"] = ds

    # selector
    sel = cfg.get("selector", {})
    if args.selector_name is not None:       sel["name"] = args.selector_name
    if args.selector_percentage is not None: sel["percentage"] = args.selector_percentage
    cfg["selector"] = sel

    # anomaly_detect
    an = cfg.get("anomaly_detect", {})
    if args.backbone_names is not None:          an["backbone_names"] = _split_list(args.backbone_names)
    if args.layers_to_extract_from is not None:  an["layers_to_extract_from"] = _split_list(args.layers_to_extract_from)
    if args.pretrain_embed_dimension is not None:an["pretrain_embed_dimension"] = args.pretrain_embed_dimension
    if args.target_embed_dimension is not None:  an["target_embed_dimension"] = args.target_embed_dimension
    if args.preprocessing is not None:           an["preprocessing"] = args.preprocessing
    if args.aggregation is not None:             an["aggregation"] = args.aggregation
    if args.anomaly_scorer_num_nn is not None:   an["anomaly_scorer_num_nn"] = args.anomaly_scorer_num_nn
    if args.feature_window is not None:          an["feature_window"] = args.feature_window
    if args.window_step is not None:             an["window_step"] = args.window_step
    if args.window_score is not None:            an["window_score"] = args.window_score
    if args.window_overlap is not None:          an["window_overlap"] = args.window_overlap
    if args.window_aggregate is not None:        an["window_aggregate"] = args.window_aggregate
    if args.proximity_on_gpu is not None:        an["proximity_on_gpu"] = args.proximity_on_gpu
    if args.proximity_num_workers is not None:   an["proximity_num_workers"] = args.proximity_num_workers
    cfg["anomaly_detect"] = an

    # motion_detection
    mo = cfg.get("motion_detection", {})
    if args.motion_enable is not None:           mo["enable"] = args.motion_enable
    if args.motion_method is not None:           mo["method"] = args.motion_method
    if args.motion_save_motion_maps is not None: mo["save_motion_maps"] = args.motion_save_motion_maps
    if args.motion_flow_threshold is not None:   mo["flow_threshold"] = args.motion_flow_threshold
    if args.motion_embed_dim is not None:        mo["motion_embed_dim"] = args.motion_embed_dim
    if args.speed_analysis is not None:          mo["speed_analysis"] = args.speed_analysis  # NEW
    if args.speed_threshold is not None:         mo["speed_threshold"] = args.speed_threshold  # NEW
    if args.motion_weight is not None:           mo["motion_weight"] = args.motion_weight  # NEW
    if args.min_speed_samples is not None:       mo["min_speed_samples"] = args.min_speed_samples  # NEW
    if args.speed_history_size is not None:      mo["speed_history_size"] = args.speed_history_size  # NEW
    cfg["motion_detection"] = mo

    # sequence_model
    sm = cfg.get("sequence_model", {})
    if args.sequence_enable is not None:                 sm["enable"] = args.sequence_enable
    if args.sequence_model_type is not None:             sm["model_type"] = args.sequence_model_type
    if args.sequence_length is not None:                 sm["sequence_length"] = args.sequence_length
    if args.sequence_hidden_size is not None:            sm["hidden_size"] = args.sequence_hidden_size
    if args.sequence_num_layers is not None:             sm["num_layers"] = args.sequence_num_layers
    if args.sequence_reconstruction_loss_weight is not None:
        sm["reconstruction_loss_weight"] = args.sequence_reconstruction_loss_weight
    cfg["sequence_model"] = sm

# --- ROI meta copy helper (small + safe) ---
def _copy_roi_meta_to_output(subdataset: str, data_path: str, dest_dir: str):
    """
    Copy roi_meta.json from <data_path>/<subdataset>/train/good/ to <dest_dir>/roi_meta.json
    if it exists. Logs a warning if not found. No-op if already the same.
    """
    src = os.path.join(data_path, subdataset, "train", "good", "roi_meta.json")
    dst = os.path.join(dest_dir, "roi_meta.json")
    if os.path.exists(src):
        os.makedirs(dest_dir, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            LOGGER.info("Copied ROI meta → %s", dst)
        except Exception as e:
            LOGGER.warning("Failed to copy ROI meta from %s → %s: %s", src, dst, e)
    else:
        LOGGER.warning("ROI meta not found at %s (skipping).")
        # List what's actually in the directory to help debug
        parent_dir = os.path.dirname(src)
        if os.path.exists(parent_dir):
            files = os.listdir(parent_dir)
            LOGGER.warning("[ROI_META_COPY]   Files in %s: %s", parent_dir, files)

def infer_frame_hw_from_dataset(ds) -> Tuple[int,int]:
    # Pull one sample to get its true H,W after any ds-level resizing
    tmp_loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    try:
        batch = next(iter(tmp_loader))
    except StopIteration:
        raise RuntimeError("Empty dataset when inferring frame size.")
    t = batch["image"] if isinstance(batch, dict) else batch
    assert t.ndim == 4, "Expected BCHW"
    _, _, H, W = t.shape
    return int(H), int(W)

# ---------------------------------------------
# Motion Speed Analysis for Linear Conveyor Belt
# ---------------------------------------------
class ConveyorMotionAnalyzer:
    """Analyze motion speed for linear conveyor belt scenarios"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.speed_analyzers = {}  # Per-ROI analyzers
        self.normal_speed_baselines = {}
        self.speed_history = []
        
    def initialize_roi_analyzer(self, roi_key, method="farneback"):
        """Initialize analyzer for specific ROI"""
        if roi_key not in self.speed_analyzers:
            self.speed_analyzers[roi_key] = MotionSpeedAnalyzer(
                method=method,
                roi_mask=None
            )
            LOGGER.info(f"Initialized motion analyzer for ROI: {roi_key}")
    
    def analyze_frame_pair(self, prev_frame, current_frame, roi_key=None, roi_coords=None):
        """Analyze motion between two frames"""
        if roi_key not in self.speed_analyzers:
            self.initialize_roi_analyzer(roi_key, self.config.get("method", "farneback"))
        
        analyzer = self.speed_analyzers[roi_key]
        
        # Convert to numpy arrays if needed
        if isinstance(prev_frame, torch.Tensor):
            prev_np = prev_frame.permute(1, 2, 0).cpu().numpy()
            prev_np = (prev_np * 255).astype(np.uint8)
        else:
            prev_np = prev_frame
            
        if isinstance(current_frame, torch.Tensor):
            current_np = current_frame.permute(1, 2, 0).cpu().numpy()
            current_np = (current_np * 255).astype(np.uint8)
        else:
            current_np = current_frame
        
        # Extract motion features with speed analysis
        motion_data = extract_motion_features_with_speed(
            current_np,
            prev_np,
            method=self.config.get("method", "farneback"),
            roi=roi_coords
        )
        
        return motion_data
    
    def update_speed_baseline(self, speeds, roi_key="default"):
        """Update normal speed baseline from training data"""
        if roi_key not in self.normal_speed_baselines:
            self.normal_speed_baselines[roi_key] = {
                'mean': np.mean(speeds) if len(speeds) > 0 else 0,
                'std': np.std(speeds) if len(speeds) > 0 else 0,
                'percentile_75': np.percentile(speeds, 75) if len(speeds) > 0 else 0,
                'percentile_25': np.percentile(speeds, 25) if len(speeds) > 0 else 0,
                'samples': len(speeds)
            }
            LOGGER.info(f"Speed baseline for {roi_key}: "
                       f"mean={self.normal_speed_baselines[roi_key]['mean']:.4f}, "
                       f"std={self.normal_speed_baselines[roi_key]['std']:.4f}")
    
    def detect_motion_anomaly(self, current_speed, roi_key="default"):
        """Detect if current speed is anomalous"""
        if roi_key not in self.normal_speed_baselines:
            return False, 0.0
        
        baseline = self.normal_speed_baselines[roi_key]
        threshold = baseline['percentile_25']  # Lower quartile as threshold
        speed_threshold = self.config.get("speed_threshold", 0.3)
        
        # Calculate slowdown ratio
        if baseline['mean'] > 0:
            slowdown_ratio = max(0, (baseline['mean'] - current_speed) / baseline['mean'])
        else:
            slowdown_ratio = 0.0
        
        # Check if slowdown exceeds threshold
        is_anomalous = (current_speed < threshold) or (slowdown_ratio > speed_threshold)
        
        # Calculate anomaly score (0-1)
        if baseline['std'] > 0:
            z_score = abs(current_speed - baseline['mean']) / baseline['std']
            anomaly_score = min(1.0, z_score / 3.0)  # Cap at 1.0
        else:
            anomaly_score = 1.0 if is_anomalous else 0.0
        
        return is_anomalous, anomaly_score
    
    def reset(self):
        """Reset all analyzers"""
        for analyzer in self.speed_analyzers.values():
            analyzer.reset()
        self.speed_history = []

# ---------------------------------------------
# [TIMING] Phase timer to dissect training steps (rank-aware)
# ---------------------------------------------
class PhaseTimer:
    def __init__(self):
        self.sum = defaultdict(float)
        self.count = defaultdict(int)

    def add(self, name: str, dt: float):
        self.sum[name] += float(dt)
        self.count[name] += 1

    @contextlib.contextmanager
    def time(self, name: str):
        t0 = perf_counter()
        try:
            yield
        finally:
            self.add(name, perf_counter() - t0)

    def snapshot(self) -> dict:
        return {"sum": dict(self.sum), "count": dict(self.count)}

    def merge_inplace(self, snap: dict):
        for k, v in snap.get("sum", {}).items():
            self.sum[k] += float(v)
        for k, v in snap.get("count", {}).items():
            self.count[k] += int(v)

    def report_lines(self, header: str = "") -> str:
        lines = []
        if header:
            lines.append(header)
        keys = sorted(self.sum.keys())
        for k in keys:
            s = self.sum[k]
            c = max(1, self.count[k])
            lines.append(f"{k:<28} total={s:8.3f}s | count={c:6d} | avg={s/c:8.6f}s")
        return "\n".join(lines)

TRAIN_TIMERS = PhaseTimer()  # global instance per rank

# ---------------------------------------------
# Metrics helper with motion tracking
# ---------------------------------------------
class TrainingMetrics:
    """Class to track training metrics and timing with motion analysis"""
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.start_time = time.time()
        self.metrics = {
            "dataset_name": dataset_name,
            "start_time": datetime.now().isoformat(),
            "total_training_time": 0,
            "frame_counts": {
                "total_training_frames": 0,
                "total_validation_frames": 0,
                "total_training_tiles": 0,
                "total_validation_tiles": 0,
                "training_videos_count": 0,
                "validation_videos_count": 0
            },
            "process_timing": {
                "data_loading_time": 0,
                "model_training_time": 0,
                "threshold_computation_time": 0,
                "anomaly_map_generation_time": 0,
                "model_saving_time": 0
            },
            "hardware_usage": {
                "max_cpu_usage_percent": 0,
                "max_ram_usage_gb": 0,
                "max_gpu_memory_usage_mb": 0
            },
            "motion_analysis": {  # NEW: Motion analysis metrics
                "normal_speed_mean": 0,
                "normal_speed_std": 0,
                "speed_threshold": 0,
                "motion_anomalies_detected": 0,
                "max_slowdown_ratio": 0
            },
            "model_info": {
                "backbone_models": [],
                "total_parameters": 0,
                "frames_processed_per_second": 0
            },
            "training_config": {
                "batch_size": config["dataset"]["batch_size"],
                "image_size": config["dataset"]["imagesize"],
                "backbone_models": config["anomaly_detect"]["backbone_names"],
                "feature_layers": config["anomaly_detect"]["layers_to_extract_from"],
                "object_detection_enabled": config["object_detection"]["enable"],
                "motion_detection_enabled": config["motion_detection"]["enable"],
                "motion_speed_analysis": config["motion_detection"].get("speed_analysis", False),  # NEW
                "sequence_model_enabled": config["sequence_model"]["enable"]
            }
        }
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.speed_history = []  # NEW: Track speeds for baseline
        self.motion_analyzer = None  # NEW: Motion analyzer instance

    def init_motion_analyzer(self, motion_config):
        """Initialize motion analyzer"""
        if motion_config.get("enable", False) and motion_config.get("speed_analysis", False):
            self.motion_analyzer = ConveyorMotionAnalyzer(motion_config)
            LOGGER.info("Motion analyzer initialized for speed analysis")
    
    def update_motion_stats(self, speed, is_anomalous=False, slowdown_ratio=0):
        """Update motion statistics"""
        if speed > 0:
            self.speed_history.append(speed)
            self.metrics["motion_analysis"]["normal_speed_mean"] = np.mean(self.speed_history)
            self.metrics["motion_analysis"]["normal_speed_std"] = np.std(self.speed_history)
            
            # Update threshold (25th percentile)
            if len(self.speed_history) >= 10:
                self.metrics["motion_analysis"]["speed_threshold"] = np.percentile(self.speed_history, 25)
            
            if is_anomalous:
                self.metrics["motion_analysis"]["motion_anomalies_detected"] += 1
                self.metrics["motion_analysis"]["max_slowdown_ratio"] = max(
                    self.metrics["motion_analysis"]["max_slowdown_ratio"],
                    slowdown_ratio
                )
    
    def start_process(self, process_name):
        setattr(self, f"{process_name}_start", time.time())

    def end_process(self, process_name):
        if hasattr(self, f"{process_name}_start"):
            duration = time.time() - getattr(self, f"{process_name}_start")
            self.metrics["process_timing"][f"{process_name}_time"] = duration
            LOGGER.info(f"{process_name.replace('_', ' ').title()} took: {duration:.2f} seconds")

    def update_hardware_usage(self):
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_usage.append(cpu_percent)
        self.metrics["hardware_usage"]["max_cpu_usage_percent"] = max(self.cpu_usage)

        ram_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        self.ram_usage.append(ram_gb)
        self.metrics["hardware_usage"]["max_ram_usage_gb"] = max(self.ram_usage)

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_memory = sum([gpu.memoryUsed for gpu in gpus])
                self.gpu_usage.append(gpu_memory)
                self.metrics["hardware_usage"]["max_gpu_memory_usage_mb"] = max(self.gpu_usage)
        except Exception as e:
            LOGGER.warning(f"Could not get GPU usage: {e}")

    def count_frames(self, train_dataloader, val_dataloader):
        try:
            # training
            train_frame_count, train_video_count = 0, 0
            if hasattr(train_dataloader.dataset, 'data_to_iterate'):
                train_frame_count = len(train_dataloader.dataset.data_to_iterate)
                video_paths = set([item[0] if isinstance(item, (list, tuple)) and len(item) > 0 else str(item)
                                   for item in train_dataloader.dataset.data_to_iterate])
                train_video_count = len(video_paths)

            # validation
            val_frame_count, val_video_count = 0, 0
            if hasattr(val_dataloader.dataset, 'data_to_iterate'):
                val_frame_count = len(val_dataloader.dataset.data_to_iterate)
                video_paths = set([item[0] if isinstance(item, (list, tuple)) and len(item) > 0 else str(item)
                                   for item in val_dataloader.dataset.data_to_iterate])
                val_video_count = len(video_paths)

            tiles_per_frame = config["tiling"]["rows"] * config["tiling"]["cols"]
            self.metrics["frame_counts"].update({
                "total_training_frames": train_frame_count,
                "total_validation_frames": val_frame_count,
                "total_training_tiles": train_frame_count * tiles_per_frame,
                "total_validation_tiles": val_frame_count * tiles_per_frame,
                "training_videos_count": train_video_count,
                "validation_videos_count": val_video_count
            })
            LOGGER.info(f"Training frames: {train_frame_count}, Videos: {train_video_count}")
            LOGGER.info(f"Validation frames: {val_frame_count}, Videos: {val_video_count}")
        except Exception as e:
            LOGGER.warning(f"Could not count frames accurately: {e}")

    def calculate_throughput(self, total_frames, total_time):
        if total_time > 0:
            fps = total_frames / total_time
            self.metrics["model_info"]["frames_processed_per_second"] = round(fps, 2)
            return fps
        return 0

    def finalize(self, anomaly_net_list=None):
        self.metrics["total_training_time"] = time.time() - self.start_time
        self.metrics["end_time"] = datetime.now().isoformat()
        total_frames = (self.metrics["frame_counts"]["total_training_frames"] +
                        self.metrics["frame_counts"]["total_validation_frames"])
        self.calculate_throughput(total_frames, self.metrics["total_training_time"])

        if anomaly_net_list:
            self.metrics["model_info"]["backbone_models"] = [
                net.backbone.name if hasattr(net, 'backbone') else 'unknown'
                for net in anomaly_net_list
            ]
            try:
                total_params = sum(
                    sum(p.numel() for p in net.parameters())
                    for net in anomaly_net_list if hasattr(net, 'parameters')
                )
                self.metrics["model_info"]["total_parameters"] = total_params
            except Exception as e:
                LOGGER.warning(f"Could not calculate model parameters: {e}")

        # Finalize motion analysis
        if self.speed_history:
            LOGGER.info(f"Motion Analysis Summary:")
            LOGGER.info(f"  Normal speed: mean={self.metrics['motion_analysis']['normal_speed_mean']:.4f}, "
                       f"std={self.metrics['motion_analysis']['normal_speed_std']:.4f}")
            LOGGER.info(f"  Speed threshold: {self.metrics['motion_analysis']['speed_threshold']:.4f}")
            LOGGER.info(f"  Motion anomalies detected: {self.metrics['motion_analysis']['motion_anomalies_detected']}")

        LOGGER.info(f"Total training time for {self.dataset_name}: {self.metrics['total_training_time']:.2f} seconds")
        LOGGER.info(f"Frames processed per second: {self.metrics['model_info']['frames_processed_per_second']}")

    def save_to_json(self, filepath):
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        metrics_serializable = convert_numpy_types(self.metrics)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        LOGGER.info(f"Training metrics saved to: {filepath}")

# ---------------------------------------------
# DDP helpers
# ---------------------------------------------
def _resolve_backend(requested: str) -> str:
    backend = (requested or "nccl").lower()
    if os.name == "nt" or not torch.cuda.is_available():
        backend = "gloo"
    return backend

def init_distributed(backend: str) -> bool:
    ext_init_distributed(backend=_resolve_backend(backend))
    return dist.is_available() and dist.is_initialized()

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass

def get_rank_world() -> Tuple[int, int]:
    return ext_get_rank_world()

def is_main_process() -> bool:
    return ext_is_main_process()

def broadcast_str(value: str, src: int = 0) -> str:
    if dist.is_available() and dist.is_initialized():
        obj = [value]
        dist.broadcast_object_list(obj, src=src)
        return obj[0]
    return value

def broadcast_obj(obj: Any, src: int = 0) -> Any:
    if dist.is_available() and dist.is_initialized():
        buf = [obj]
        dist.broadcast_object_list(buf, src=src)
        return buf[0]
    return obj

def _barrier():
    ext_barrier()

def _bcast_batch_from_rank0(t: Optional[torch.Tensor], tag: int = 2000) -> Optional[torch.Tensor]:
    """
    Rank-0 sends a BCHW float32 tensor; other ranks receive a tensor with the same shape.
    Returns None when the sender signals end-of-iteration.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return t  # single-process path

    rank = dist.get_rank()
    shape_hdr = torch.empty(4, dtype=torch.int64)

    if rank == 0:
        if t is None:
            shape_hdr[:] = torch.tensor([-1, -1, -1, -1], dtype=torch.int64)
            dist.broadcast(shape_hdr, src=0)
            return None
        shape = torch.tensor(list(t.shape), dtype=torch.int64)
        dist.broadcast(shape, src=0)
        dist.broadcast(t.contiguous(), src=0)
        return t
    else:
        dist.broadcast(shape_hdr, src=0)
        s0, s1, s2, s3 = [int(x) for x in shape_hdr.tolist()]
        if s0 == -1:
            return None
        recv = torch.empty((s0, s1, s2, s3), dtype=torch.float32)
        dist.broadcast(recv, src=0)
        return recv

# ---------------------------------------------
# Tiling utilities (unified for train + infer)
# ---------------------------------------------
_TILE_COORDS_CACHE: Dict[Tuple[int,int,int,int,float], List[Tuple[int,int,int,int]]] = {}

def compute_tile_coords_hw(H: int, W: int, rows: int, cols: int, ov: float) -> List[Tuple[int, int, int, int]]:
    if not (0.0 <= float(ov) < 1.0):
        raise ValueError(f"tile_overlap must be in [0,1); got {ov}")
    R, Cn = int(rows), int(cols)
    base_h = max(1, H // R)
    base_w = max(1, W // Cn)
    overlap_h = int(round(base_h * ov))
    overlap_w = int(round(base_w * ov))
    stride_y = max(1, base_h - overlap_h)
    stride_x = max(1, base_w - overlap_w)
    actual_h = base_h + overlap_h
    actual_w = base_w + overlap_w

    coords = []
    for r in range(R):
        y0 = r * stride_y
        y1 = y0 + actual_h
        if y1 > H:
            y1 = H
            y0 = max(0, y1 - actual_h)
        for c in range(Cn):
            x0 = c * stride_x
            x1 = x0 + actual_w
            if x1 > W:
                x1 = W
                x0 = max(0, x1 - actual_w)
            coords.append((int(x0), int(y0), int(x1), int(y1)))
    assert len(coords) == R * Cn, f"Expected {R*Cn} tiles, got {len(coords)}"
    return coords

def get_tile_coords_cached(H: int, W: int, rows: int, cols: int, ov: float) -> List[Tuple[int,int,int,int]]:
    key = (H, W, int(rows), int(cols), float(ov))
    if key in _TILE_COORDS_CACHE:
        return _TILE_COORDS_CACHE[key]
    coords = compute_tile_coords_hw(H, W, rows, cols, ov)
    _TILE_COORDS_CACHE[key] = coords
    return coords

def hann_weight(h: int, w: int) -> np.ndarray:
    wy = np.hanning(max(h, 2))[:, None]
    wx = np.hanning(max(w, 2))[None, :]
    w2d = wy * wx
    m = float(w2d.max()) if w2d.size else 1.0
    return (w2d / (m + 1e-8)).astype(np.float32)

def stitch_tiles_blend(
    pos_outputs: Dict[int, np.ndarray],
    coords: List[Tuple[int, int, int, int]],
    H: int,
    W: int,
    use_hann: bool = True,
    eps: float = 1e-8,
    weight_cache: Optional[Dict[Tuple[int,int], np.ndarray]] = None,
) -> np.ndarray:
    acc  = np.zeros((H, W), dtype=np.float32)  # weighted sum
    wsum = np.zeros((H, W), dtype=np.float32)  # weight sum
    if weight_cache is None:
        weight_cache = {}

    for pos, (x0, y0, x1, y1) in enumerate(coords):
        seg = pos_outputs.get(pos, None)
        if seg is None:
            continue

        th, tw = (y1 - y0), (x1 - x0)
        if seg.shape[:2] != (th, tw):
            seg = cv2.resize(seg, (tw, th), interpolation=cv2.INTER_LINEAR)

        # keep original scale; normalize later only for visualization if needed
        seg_n = seg.astype(np.float32)

        if use_hann:
            key = (th, tw)
            if key not in weight_cache:
                wy = np.hanning(max(th, 2))[:, None]
                wx = np.hanning(max(tw, 2))[None, :]
                w  = (wy * wx).astype(np.float32)
                w /= (w.max() + eps)
                weight_cache[key] = w
            w = weight_cache[key]
        else:
            w = np.ones((th, tw), dtype=np.float32)

        acc[y0:y1, x0:x1]  += seg_n * w
        wsum[y0:y1, x0:x1] += w

    wsum[wsum == 0] = 1.0
    stitched = acc / wsum
    return stitched

# ---------------------------------------------
# Data / selector / model helpers
# ---------------------------------------------
def get_dataloaders(seed: int, dataset_config: Dict[str, Any], model_result_dir: str,
                    extract_on_this_rank: bool, disable_interactive_on_this_rank: bool) -> List[Dict[str, Any]]:
    """Create train/val loaders. On non-rank0, skip expensive extraction & interactive ROI."""
    dataset_info = _DATASETS[dataset_config["name"]]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1], "DatasetSplit"])  # type: ignore
    dataloaders = []

    for subdataset in dataset_config["subdatasets"]:
        cfg_local = dict(dataset_config)
        if not extract_on_this_rank:
            cfg_local["video_to_frames"] = False
        if disable_interactive_on_this_rank:
            cfg_local["roi_selection_mode"] = False
            cfg_local["enable_object_detection"] = False

        train_dataset = dataset_library.__dict__[dataset_info[1]](
            source=cfg_local["data_path"],
            classname=subdataset,
            resize=cfg_local["resize"],
            train_val_split=cfg_local["train_val_split"],
            imagesize=cfg_local["imagesize"],
            split=dataset_library.DatasetSplit.TRAIN,
            seed=seed,
            augment=cfg_local["augment"],
            video_to_frames=cfg_local["video_to_frames"],
            frame_interval=cfg_local["frame_interval"],
            model_result_dir=model_result_dir,
            motion_config=config["motion_detection"],
            sequence_config=config["sequence_model"],
            object_detection_config=config["object_detection"],
            roi_selection_mode=cfg_local.get("roi_selection_mode", False),
            enable_object_detection=cfg_local.get("enable_object_detection", False),
        )
        val_dataset = dataset_library.__dict__[dataset_info[1]](
            source=cfg_local["data_path"],
            classname=subdataset,
            resize=cfg_local["resize"],
            train_val_split=cfg_local["train_val_split"],
            imagesize=cfg_local["imagesize"],
            split=dataset_library.DatasetSplit.VAL,
            seed=seed,
            augment=False,
            video_to_frames=cfg_local["video_to_frames"],
            frame_interval=cfg_local["frame_interval"],
            model_result_dir=model_result_dir,
            motion_config=config["motion_detection"],
            sequence_config=config["sequence_model"],
            object_detection_config=config["object_detection"],
            roi_selection_mode=cfg_local.get("roi_selection_mode", False),
            enable_object_detection=cfg_local.get("enable_object_detection", False),
        )

        # Train loader: rank-aware workers/pinning. Keep B=1 when tiling.
        nw = cfg_local["num_workers"]
        tr_bs = cfg_local["batch_size"]
        pin = True
        persistent = False
        extra_dl_kwargs: Dict[str, Any] = {}

        tiling_on = config.get("tiling", {}).get("enable", False)
        if tiling_on:
            tr_bs = 1

        import platform

        # ---- Windows-safe policy for training loader ----
        if platform.system() == "Windows":
            nw = 4
            pin = True
            persistent = False
            extra_dl_kwargs["prefetch_factor"] = 4
        else:
            if disable_interactive_on_this_rank:
                nw = 0
                pin = False
                persistent = False
            else:
                if tiling_on:
                    nw = max(nw, max(4, (os.cpu_count() or 8)//2))
                persistent = (nw > 0)
                if nw > 0:
                    extra_dl_kwargs["prefetch_factor"] = 4

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=tr_bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=False,
            persistent_workers=False,
            multiprocessing_context=("spawn" if nw > 0 else None),
            **extra_dl_kwargs
        )
        # Validation loader: only main rank uses workers/pin to avoid duplication
        val_nw = 0
        val_pin = False
        if is_main_process():
            val_nw = max(2, (os.cpu_count() or 8)//4)
            val_pin = True
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg_local["batch_size"],
            shuffle=False,
            num_workers=val_nw,
            pin_memory=False,
            persistent_workers=(val_nw > 0)
        )

        train_dataloader.name = cfg_local["name"]
        if subdataset is not None:
            train_dataloader.name += "_" + subdataset

        dataloaders.append({"training": train_dataloader, "validation": val_dataloader})
    return dataloaders

def get_selector(selector_config: Dict[str, Any], device: torch.device):
    name = selector_config["name"]
    if name == "identity":
        return anomaly_engine.selectors.feature_selector.IdentitySelector()
    elif name == "greedy_coreset":
        return anomaly_engine.selectors.feature_selector.GreedyCoresetSelector(
            selector_config["percentage"], device)
    elif name == "approx_greedy_coreset":
        return anomaly_engine.selectors.feature_selector.ApproximateGreedyCoresetSelector(
            selector_config["percentage"], device)

def _split_layers_for_backbones(backbone_names: List[str], layers_to_extract_from: List[str]) -> List[List[str]]:
    if len(backbone_names) > 1:
        coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            lyr = ".".join(layer.split(".")[1:])
            coll[idx].append(lyr)
        return coll
    else:
        return [layers_to_extract_from]

def _normalize_input_shape(shape) -> Tuple[int, int, int]:
    import numpy as _np
    def _ii(x):
        return int(x) if isinstance(x, (int, _np.integer)) else int(x)

    if isinstance(shape, (int, _np.integer)):
        s = _ii(shape)
        return (3, s, s)

    if not isinstance(shape, tuple):
        raise TypeError(f"Unsupported input_shape format: {shape!r}")

    if len(shape) == 2 and all(isinstance(v, (int, _np.integer)) for v in shape):
        h, w = _ii(shape[0]), _ii(shape[1])
        return (3, h, w)

    if len(shape) == 3 and all(isinstance(v, (int, _np.integer)) for v in shape):
        c, h, w = _ii(shape[0]), _ii(shape[1]), _ii(shape[2])
        return (c, h, w)

    if len(shape) == 2 and isinstance(shape[1], tuple):
        c, hw = shape
        if len(hw) != 2:
            raise TypeError(f"Unsupported nested shape: {shape!r}")
        h, w = _ii(hw[0]), _ii(hw[1])
        return (_ii(c), h, w)

    if len(shape) == 3 and isinstance(shape[1], tuple) and isinstance(shape[2], tuple):
        c, hw1, hw2 = shape
        if len(hw1) != 2 or len(hw2) != 2:
            raise TypeError(f"Unsupported nested shape: {shape!r}")
        h1, w1 = _ii(hw1[0]), _ii(hw1[1])
        h2, w2 = _ii(hw2[0]), _ii(hw2[1])
        if (h1, w1) != (h2, w2):
            LOGGER.warning("input_shape has two different (H,W): %s vs %s; using the first.", (h1, w1), (h2, w2))
        return (_ii(c), h1, w1)

    flat: list[int] = []
    def _flatten(x):
        if isinstance(x, (int, _np.integer)):
            flat.append(_ii(x))
        elif isinstance(x, tuple):
            for y in x:
                _flatten(y)
        else:
            raise TypeError(f"Unsupported nested type in input_shape: {type(x)}")
    _flatten(shape)
    if len(flat) >= 2:
        h, w = flat[-2], flat[-1]
        c = flat[0] if len(flat) >= 3 else 3
        return (int(c), int(h), int(w))
    raise TypeError(f"Unsupported input_shape format: {shape!r}.")

def get_anomaly_net(input_shape: Union[int, Tuple[int, int]], selector, device: torch.device,
                    anomaly_config: Dict[str, Any], motion_config: Dict[str, Any],
                    sequence_config: Dict[str, Any], object_detection_config: Dict[str, Any]) -> List[Anomaly_net_eng]:
    loaded = []
    norm_input_shape = _normalize_input_shape(input_shape)
    LOGGER.info("[INPUT_SHAPE] normalized to (C,H,W)=%s from %s", norm_input_shape, input_shape)

    backbone_names = anomaly_config["backbone_names"]
    layers_coll = _split_layers_for_backbones(backbone_names, anomaly_config["layers_to_extract_from"]).copy()

    for backbone_name, layers_to_extract_from in zip(backbone_names, layers_coll):
        backbone_seed = None
        if ".seed-" in backbone_name:
            backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(backbone_name.split("-")[-1])
        backbone = anomaly_engine.models.network_models.load(backbone_name)
        backbone.name, backbone.seed = backbone_name, backbone_seed

        nn_method = anomaly_engine.core.core_utils.ApproximateProximitySearcher(
            anomaly_config["proximity_on_gpu"],
            anomaly_config["proximity_num_workers"],
        )

        model = Anomaly_net_eng(device)
        model.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=device,
            input_shape=norm_input_shape,
            pretrain_embed_dimension=anomaly_config["pretrain_embed_dimension"],
            target_embed_dimension=anomaly_config["target_embed_dimension"],
            feature_window=anomaly_config["feature_window"],
            window_step=anomaly_config["window_step"],
            featuresampler=selector,
            anomaly_score_num_nn=anomaly_config["anomaly_scorer_num_nn"],
            nn_method=nn_method,
            motion_config=motion_config,
            sequence_config=sequence_config,
            object_detection_config=object_detection_config,
        )
        loaded.append(model)
    return loaded

# ---------------------------------------------
# FAISS per-position save / index
# ---------------------------------------------
def save_faiss_position(anomaly_net: Anomaly_net_eng, models_root: str, dataset_name: str, pos_id: int, extra_meta: dict = None) -> str:
    r = get_rank_world()[0]
    pos_dir = os.path.join(models_root, dataset_name, f"rank{r}", f"pos{pos_id}")
    os.makedirs(pos_dir, exist_ok=True)

    saved = False
    nn = getattr(anomaly_net, "nn_method", None)
    if nn is not None and hasattr(nn, "save"):
        nn.save(os.path.join(pos_dir, "nnscorer_search_index.faiss"))
        saved = True

    if not saved:
        scorer = getattr(anomaly_net, "anomaly_scorer", None)
        if scorer is not None and hasattr(scorer, "save"):
            scorer.save(os.path.join(pos_dir, "nnscorer_search_index.faiss"))
            saved = True

    if not saved and hasattr(anomaly_net, "save_to_path"):
        anomaly_net.save_to_path(pos_dir, prepend="")
        saved = True

    if not saved:
        raise RuntimeError("Could not locate FAISS exporter on anomaly_net (nn_method.save / anomaly_scorer.save / save_to_path)")

    proj = {}
    for k in ("mean", "std", "proj_W", "proj_b"):
        if hasattr(anomaly_net, k):
            v = getattr(anomaly_net, k)
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            proj[k] = v
    if proj:
        np.savez(os.path.join(pos_dir, "projector.npz"), **proj)

    meta = {
        "saved_by_rank": r,
        "pos_id": pos_id,
        "ts": time.time(),
        "dataset": dataset_name,
        "engine": "AnomalyNet",
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(pos_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    LOGGER.info("[SAVE] rank=%s pos=%s → %s", r, pos_id, pos_dir)
    return pos_dir

def build_models_index(models_root: str, dataset_name: str) -> Dict[str, List[str]]:
    root = os.path.join(models_root, dataset_name)
    index: Dict[str, List[str]] = {}
    for path in glob.glob(os.path.join(root, "rank*", "pos*")):
        base = os.path.basename(path)
        if not base.startswith("pos"):
            continue
        try:
            pos_id = int(base[3:])
        except ValueError:
            continue
        index.setdefault(str(pos_id), []).append(os.path.relpath(path, root))
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    LOGGER.info("[INDEX] models index written at %s", os.path.join(root, "index.json"))
    return index

# ---------------------------------------------
# Tiled per-image INFERENCE with DDP sharding + Hann blend + Motion
# ---------------------------------------------
@torch.no_grad()
def infer_image_tiled_with_motion(
    anomaly_model: Anomaly_net_eng,
    img_tensor_1BCHW: torch.Tensor,
    device: torch.device,
    til_cfg: dict,
    is_ddp: bool,
    dataset_name: str,
    motion_analyzer: Optional[ConveyorMotionAnalyzer] = None,
    prev_frame: Optional[torch.Tensor] = None,
    roi_coords: Optional[Tuple[int, int, int, int]] = None,
    pos_loader: Optional[PositionAwareAnomalyNet] = None,
    out_dir_for_tiles: Optional[str] = None,
    cached_coords: Optional[List[Tuple[int,int,int,int]]] = None,
    hann_cache: Optional[Dict[Tuple[int,int], np.ndarray]] = None,
):
    """
    Run tiled inference with motion analysis on a single image tensor (shape 1xCxHxW).
    - Returns a stitched float32 heatmap of shape (H, W) and motion analysis results.
    """
    # Extract H,W from tensor
    _, _, H, W = img_tensor_1BCHW.shape

    # Precompute coords once
    coords: List[Tuple[int, int, int, int]] = (
        cached_coords
        if cached_coords is not None
        else get_tile_coords_cached(H, W, til_cfg["rows"], til_cfg["cols"], til_cfg["overlap"])
    )

    # Motion analysis
    motion_results = None
    if motion_analyzer and prev_frame is not None:
        try:
            motion_data = motion_analyzer.analyze_frame_pair(
                prev_frame, img_tensor_1BCHW[0],
                roi_key=dataset_name,
                roi_coords=roi_coords
            )
            
            # Get motion anomaly detection
            if motion_data:
                is_anomalous, motion_score = motion_analyzer.detect_motion_anomaly(
                    motion_data['speed'],
                    roi_key=dataset_name
                )
                
                motion_results = {
                    'speed': motion_data['speed'],
                    'is_slow': motion_data.get('is_slow', False),
                    'slowdown_ratio': motion_data.get('slowdown_ratio', 0.0),
                    'motion_anomaly_score': motion_score,
                    'is_motion_anomalous': is_anomalous,
                    'flow_magnitude': motion_data.get('flow_magnitude'),
                    'flow_angle': motion_data.get('flow_angle')
                }
        except Exception as e:
            LOGGER.warning(f"Motion analysis failed: {e}")

    # Rank helpers
    def _rank_world() -> Tuple[int, int]:
        if is_ddp and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _is_main() -> bool:
        r, _ = _rank_world()
        return r == 0

    def _my_positions(npos: int) -> List[int]:
        r, w = _rank_world()
        if w <= 1:
            return list(range(npos))
        # Use the exact same sharding scheme as training
        try:
            return ext_shard_positions(til_cfg["rows"], til_cfg["cols"], w, r)
        except Exception:
            # Fallback: round-robin (shouldn't happen)
            return [p for p in range(npos) if (p % w) == r]

    my_positions = _my_positions(len(coords))
    local: Dict[int, np.ndarray] = {}

    # ---- Per-tile inference on my shard ----
    for pos, (x0, y0, x1, y1) in enumerate(coords):
        if pos not in my_positions:
            continue

        tile_t = img_tensor_1BCHW[:, :, y0:y1, x0:x1].to(device, non_blocking=True)
        c = tile_t.shape[1]
        if c == 1:
            tile_t = tile_t.repeat(1, 3, 1, 1)
        elif c == 4:
            tile_t = tile_t[:, :3, :, :]
        elif c != 3:
            raise ValueError(f"Unexpected channel count C={c}; expected 1, 3, or 4.")

        if pos_loader is not None:
            out = pos_loader.infer_tile(pos, tile_t)
            heat = out["heatmap"] if isinstance(out, dict) else out
        else:
            scores, maps = anomaly_model._predict(tile_t, None)
            heat = maps[0]
            if isinstance(heat, torch.Tensor):
                heat = heat.detach().cpu().numpy()

        local[pos] = np.asarray(heat, dtype=np.float32)

    # ---- DDP gather ----
    if is_ddp and dist.is_initialized():
        _, world = _rank_world()
        gathered: List[Any] = [None for _ in range(world)]

        # sanitize outbound payload (int -> float32 ndarray)
        outbound: Dict[int, np.ndarray] = {}
        for k, v in local.items():
            try:
                kk = int(k)
            except Exception:
                continue
            outbound[kk] = np.asarray(v, dtype=np.float32)

        dist.all_gather_object(gathered, outbound)

        pos_outputs: Dict[int, np.ndarray] = {}
        for i, d in enumerate(gathered):
            if isinstance(d, dict):
                for kk, vv in d.items():
                    if isinstance(kk, (int, np.integer)):
                        pos_outputs[int(kk)] = np.asarray(vv, dtype=np.float32)
                continue

            if isinstance(d, list):
                merged: Dict[int, np.ndarray] = {}
                try:
                    # Case A: list of dicts
                    if all(isinstance(x, dict) for x in d):
                        for dd in d:
                            for kk, vv in dd.items():
                                if isinstance(kk, (int, np.integer)):
                                    merged[int(kk)] = np.asarray(vv, dtype=np.float32)
                    # Case B: list of (pos, array)-pairs
                    elif all(isinstance(x, (list, tuple)) and len(x) == 2 for x in d):
                        for kk, vv in d:
                            if isinstance(kk, (int, np.integer)):
                                merged[int(kk)] = np.asarray(vv, dtype=np.float32)
                except Exception:
                    merged = {}

                if merged:
                    pos_outputs.update(merged)
                    continue

            if d is not None:
                LOGGER.warning(
                    "infer_image_tiled: gathered[%d] type=%s (not coercible); skipping.",
                    i, type(d).__name__,
                )
    else:
        pos_outputs = local

    expected = len(coords)
    if len(pos_outputs) != expected:
        LOGGER.warning("infer_image_tiled: got %d/%d tiles after gather.", len(pos_outputs), expected)

    # Optional debug dump of tiles
    if _is_main() and out_dir_for_tiles and til_cfg.get("save_tile_outputs", False):
        os.makedirs(out_dir_for_tiles, exist_ok=True)
        for pos, heat in pos_outputs.items():
            dpos = os.path.join(out_dir_for_tiles, f"pos{pos}")
            os.makedirs(dpos, exist_ok=True)
            np.save(os.path.join(dpos, "tile.npy"), heat)
            hmin, hmax = float(np.min(heat)), float(np.max(heat))
            vis = (255 * (heat - hmin) / (hmax - hmin + 1e-8)).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(dpos, "tile.png"), vis)

    # ---- Blend tiles with Hann weights (if enabled) ----
    stitched = stitch_tiles_blend(
        pos_outputs=pos_outputs,
        coords=coords,
        H=H,
        W=W,
        use_hann=bool(til_cfg.get("hann_blend", True)),
        weight_cache=hann_cache,
    )
    
    return stitched, motion_results

# ---------------------------------------------
# Main
# ---------------------------------------------
def main(argv=None):
    # Parse CLI and merge into config
    args = build_arg_parser().parse_args(argv)
    update_config_from_args(config, args)
    overall_start = time.time()
    ddp_cfg = config.get("ddp", {})
    if ddp_cfg.get("enable", False):
        backend_req = ddp_cfg.get("backend", None)
        # your wrapper resolves to gloo on Windows/CPU automatically
        dist_inited = init_distributed(backend_req or "nccl")
    else:
        dist_inited = False
    til_cfg = config.get("tiling", {})
    pos_cfg = config.get("position_models", {})

    is_ddp = bool(ddp_cfg.get("enable", False))
    use_tiling = bool(til_cfg.get("enable", False))

    LOGGER.info(
        "FLAGS → tiling=%s (rows=%s, cols=%s, overlap=%s) | position_models=%s | ddp=%s",
        use_tiling, til_cfg.get("rows"), til_cfg.get("cols"), til_cfg.get("overlap"),
        pos_cfg.get("enable", False), is_ddp
    )
    
    # Log motion settings
    motion_cfg = config.get("motion_detection", {})
    if motion_cfg.get("enable", False):
        LOGGER.info("MOTION → Enabled with method=%s, speed_analysis=%s", 
                   motion_cfg.get("method", "farneback"),
                   motion_cfg.get("speed_analysis", False))

    if is_ddp:
        LOGGER.info(
            "DDP requested | env RANK=%s WORLD_SIZE=%s LOCAL_RANK=%s",
            os.environ.get("RANK"), os.environ.get("WORLD_SIZE"), os.environ.get("LOCAL_RANK")
        )

    # ---- Init DDP ----
    dist_inited = False
    if is_ddp:
        LOGGER.info("Initializing DDP backend=%s …", ddp_cfg.get("backend", "nccl"))
        dist_inited = init_distributed(ddp_cfg.get("backend", "nccl"))
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    rank, world = get_rank_world()
    LOGGER.info(
        "RUNTIME → dist_inited=%s | rank=%s | world=%s | backend=%s",
        dist_inited, rank, world, (_resolve_backend(ddp_cfg.get("backend", "nccl")) if is_ddp else "NA")
    )

    # FAISS info (rank0)
    if is_main_process():
        try:
            try:
                faiss.omp_set_num_threads(max(1, (os.cpu_count() or 8)//2))
            except Exception:
                pass
            LOGGER.info("FAISS OMP threads=%s | CUDA_VISIBLE_DEVICES=%s",
                        getattr(faiss, "omp_get_max_threads", lambda: "NA")(),
                        os.environ.get("CUDA_VISIBLE_DEVICES"))
        except Exception:
            pass

    # ---- Device ----
    device = anomaly_engine.core.core_utils.set_torch_device(config["gpu"])
    device_context = (
        torch.cuda.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else contextlib.suppress()
    )
    if torch.cuda.is_available():
        try:
            LOGGER.info(
                "CUDA → current_device=%s name=%s device_count=%s",
                torch.cuda.current_device(),
                torch.cuda.get_device_name(torch.cuda.current_device()),
                torch.cuda.device_count(),
            )
        except Exception:
            pass
    else:
        LOGGER.info("CUDA not available; running on CPU.")

    # ---- Create storage folder (rank0 then broadcast) ----
    if is_main_process():
        run_save_path = anomaly_engine.core.core_utils.create_storage_folder(
            config["results_path"], config["log_project"], config["log_group"], mode="iterate"
        )
    else:
        run_save_path = None

    if dist_inited:
        run_save_path = broadcast_str(run_save_path, src=0)
        if not is_main_process():
            os.makedirs(run_save_path, exist_ok=True)
        dist.barrier()

    stats_csv_path = os.path.join(run_save_path, "category_stats.csv")
    if is_main_process():
        os.makedirs(os.path.dirname(stats_csv_path), exist_ok=True)
        with open(stats_csv_path, mode="w", newline="") as csv_file:
            csv.writer(csv_file).writerow(["Category", "Average Score", "Max Score < 1"])
        cfg_snapshot = json.loads(json.dumps(config))
        with open(os.path.join(run_save_path, "training_config.json"), "w") as f:
            json.dump(cfg_snapshot, f, indent=2)
        LOGGER.info("Training configuration saved to %s", os.path.join(run_save_path, "training_config.json"))

    models_root = os.path.join(run_save_path, "models")

    # ---- DDP-safe dataloaders: only rank0 does extraction / interactive bits ----
    extract_on_this_rank = (not dist_inited) or is_main_process()
    disable_interactive_on_this_rank = (dist_inited and not is_main_process())

    if extract_on_this_rank:
        list_of_dataloaders = get_dataloaders(
            config["seed"], config["dataset"], models_root,
            extract_on_this_rank=True, disable_interactive_on_this_rank=False
        )
    _barrier()
    if not extract_on_this_rank:
        list_of_dataloaders = get_dataloaders(
            config["seed"], config["dataset"], models_root,
            extract_on_this_rank=False, disable_interactive_on_this_rank=True
        )
    _barrier()

    pos_loader: Optional[PositionAwareAnomalyNet] = None

    # ===================== per dataset loop =====================
    for dl_idx, dls in enumerate(list_of_dataloaders):
        dataset_name = dls["training"].name
        LOGGER.info("Processing dataset [%s] (%s/%s)…", dataset_name, dl_idx + 1, len(list_of_dataloaders))

        metrics = TrainingMetrics(dataset_name)
        metrics.init_motion_analyzer(config["motion_detection"])  # Initialize motion analyzer
        metrics.start_process("data_loading")

        anomaly_engine.core.core_utils.fix_seeds(config["seed"], device)

        with device_context:
            torch.cuda.empty_cache()
            metrics.count_frames(dls["training"], dls["validation"])
            metrics.update_hardware_usage()
            metrics.end_process("data_loading")

            # --- Per-subdataset frame/tile size inference ---
            rows = int(config["tiling"]["rows"])
            cols = int(config["tiling"]["cols"])
            ov   = float(config["tiling"]["overlap"])

            # Infer the actual frame size for THIS subdataset
            frame_h, frame_w = infer_frame_hw_from_dataset(dls["training"].dataset)

            # Build tile coords and compute the tile size for THIS subdataset
            coords_ds = get_tile_coords_cached(frame_h, frame_w, rows, cols, ov)
            x0, y0, x1, y1 = coords_ds[0]
            tile_h, tile_w = (y1 - y0), (x1 - x0)

            # Make sure the dataset and model see the right per-dataset imagesize
            dls["training"].dataset.imagesize = (tile_h, tile_w)
            dls["validation"].dataset.imagesize = (tile_h, tile_w)

            LOGGER.info(
                "[SIZE/%s] frame(H,W)=%s → tile(H,W)=%s (rows=%d, cols=%d, ov=%.3f)",
                dls["training"].name, (frame_h, frame_w), (tile_h, tile_w), rows, cols, ov
            )

            imagesize = (tile_h, tile_w)  # used below by get_anomaly_net(...)

            selector = get_selector(config["selector"], device)

            # Limit proximity workers per rank when DDP sharing one CPU
            anomaly_detect_cfg = config["anomaly_detect"].copy()
            if dist_inited:
                cpu_cnt = os.cpu_count() or 8
                _, world = get_rank_world()
                anomaly_detect_cfg["proximity_num_workers"] = max(2, cpu_cnt // max(2, 2*world))

            anomaly_net_list = get_anomaly_net(
                imagesize, selector, device,
                anomaly_detect_cfg,
                config["motion_detection"],
                config["sequence_model"],
                config["object_detection"],
            )
            if len(anomaly_net_list) > 1:
                LOGGER.info("Using AnomalyNet Ensemble (N=%s).", len(anomaly_net_list))

            # ---- TRAIN with motion analysis ----
            metrics.start_process("model_training")
            speed_history = []  # Collect speeds for baseline
            
            for i, anomaly_net in enumerate(anomaly_net_list):
                torch.cuda.empty_cache()
                if getattr(anomaly_net.backbone, "seed", None) is not None:
                    anomaly_engine.core.core_utils.fix_seeds(anomaly_net.backbone.seed, device)
                LOGGER.info("Training model (%s/%s)", i + 1, len(anomaly_net_list))

                train_source = dls["training"]
                if use_tiling:
                    tiled_iter = TiledTrainIterator(
                        loader=train_source,
                        rows=config["tiling"]["rows"],
                        cols=config["tiling"]["cols"],
                        overlap=config["tiling"]["overlap"],
                        ddp_enabled=dist_inited,
                        timers=TRAIN_TIMERS,
                    )
                    with TRAIN_TIMERS.time("train.fit_total"):
                        with torch.inference_mode(), autocast("cuda"):
                            # Collect motion data during training
                            prev_frame = None
                            for batch in tiled_iter:
                                images = batch["image"] if isinstance(batch, dict) else batch
                                
                                # If motion analysis is enabled, collect speed data
                                if config["motion_detection"]["enable"] and config["motion_detection"].get("speed_analysis", False):
                                    if prev_frame is not None:
                                        # Analyze motion between frames
                                        motion_data = metrics.motion_analyzer.analyze_frame_pair(
                                            prev_frame[0] if prev_frame.dim() == 4 else prev_frame,
                                            images[0] if images.dim() == 4 else images,
                                            roi_key=dataset_name,
                                            roi_coords=config["motion_detection"].get("roi_for_motion")
                                        )
                                        if motion_data:
                                            speed_history.append(motion_data['speed'])
                                            metrics.update_motion_stats(
                                                motion_data['speed'],
                                                motion_data.get('is_slow', False),
                                                motion_data.get('slowdown_ratio', 0)
                                            )
                                    
                                    prev_frame = images
                                
                            # Reset iterator for actual training
                            tiled_iter = TiledTrainIterator(
                                loader=train_source,
                                rows=config["tiling"]["rows"],
                                cols=config["tiling"]["cols"],
                                overlap=config["tiling"]["overlap"],
                                ddp_enabled=dist_inited,
                                timers=TRAIN_TIMERS,
                            )
                            anomaly_net.fit(tiled_iter)
                else:
                    with TRAIN_TIMERS.time("train.fit_total"):
                        with torch.inference_mode(), autocast("cuda"):
                            # Collect motion data for non-tiled training
                            if config["motion_detection"]["enable"] and config["motion_detection"].get("speed_analysis", False):
                                prev_frame = None
                                for batch in train_source:
                                    images = batch["image"] if isinstance(batch, dict) else batch
                                    
                                    if prev_frame is not None:
                                        motion_data = metrics.motion_analyzer.analyze_frame_pair(
                                            prev_frame[0] if prev_frame.dim() == 4 else prev_frame,
                                            images[0] if images.dim() == 4 else images,
                                            roi_key=dataset_name,
                                            roi_coords=config["motion_detection"].get("roi_for_motion")
                                        )
                                        if motion_data:
                                            speed_history.append(motion_data['speed'])
                                            metrics.update_motion_stats(
                                                motion_data['speed'],
                                                motion_data.get('is_slow', False),
                                                motion_data.get('slowdown_ratio', 0)
                                            )
                                    
                                    prev_frame = images
                            
                            anomaly_net.fit(train_source)

                metrics.update_hardware_usage()
            
            # Update motion analyzer with collected speeds
            if speed_history:
                metrics.motion_analyzer.update_speed_baseline(speed_history, dataset_name)
                LOGGER.info(f"Collected {len(speed_history)} speed samples for {dataset_name}")
                LOGGER.info(f"Normal speed baseline: mean={np.mean(speed_history):.4f}, std={np.std(speed_history):.4f}")
            
            metrics.end_process("model_training")

            # ---- SAVE FAISS per-position (for my shard only) ----
            if use_tiling:
                tmp_iter = TiledTrainIterator(
                    loader=dls["training"],
                    rows=config["tiling"]["rows"],
                    cols=config["tiling"]["cols"],
                    overlap=config["tiling"]["overlap"],
                    ddp_enabled=dist_inited,
                    timers=TRAIN_TIMERS,
                )
                my_positions = list(tmp_iter.my_positions)
            else:
                my_positions = [0]

            for pos_id in my_positions:
                with TRAIN_TIMERS.time("train.save_pos"):
                    save_faiss_position(
                        anomaly_net_list[0], models_root=models_root,
                        dataset_name=dataset_name, pos_id=pos_id,
                        extra_meta={"backbone": anomaly_net_list[0].backbone.name},
                    )

            t_b0 = perf_counter(); _barrier(); TRAIN_TIMERS.add("sync.barrier", perf_counter() - t_b0)

            if is_main_process():
                with TRAIN_TIMERS.time("train.index_build"):
                    build_models_index(models_root, dataset_name)

            t_b1 = perf_counter(); _barrier(); TRAIN_TIMERS.add("sync.barrier", perf_counter() - t_b1)

            # ---- POSITION LOADER ----
            if config["position_models"]["enable"]:
                dev_str = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
                pos_loader = PositionAwareAnomalyNet(
                    base_cfg=config.get("anomaly_detect", {}),
                    models_root=models_root,
                    device=dev_str,
                    dataset_name=dataset_name,
                )
                LOGGER.info("Position models ENABLED | models_root=%s | device=%s", models_root, dev_str)
            else:
                LOGGER.info("Position models DISABLED (single model for all tiles).")

            # ---- VALIDATION (combined threshold + maps + motion) ----
            metrics.start_process("validation_combined")
            LOGGER.info("Validation (combined threshold + maps + motion) starting ...")

            cached_coords = coords_ds   # uses the (frame_h, frame_w) of THIS subdataset
            hann_cache: Dict[Tuple[int,int], np.ndarray] = {}
            
            # Get ROI for motion analysis if specified
            motion_roi = None
            if config["motion_detection"].get("roi_for_motion"):
                motion_roi = config["motion_detection"]["roi_for_motion"]
                if isinstance(motion_roi, dict):
                    motion_roi = (motion_roi["x"], motion_roi["y"], motion_roi["w"], motion_roi["h"])

            # rank-0 iterates validation; other ranks receive the broadcasted batch
            if is_main_process():
                validation_dataloader = torch.utils.data.DataLoader(
                    dls["validation"].dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=max(1, (os.cpu_count() or 8)//8),
                    pin_memory=False,
                    persistent_workers=True,
                    prefetch_factor=4,
                )
            else:
                validation_dataloader = None

            local_scores: List[float] = []
            local_maps: List[np.ndarray] = []
            local_motion_scores: List[float] = []  # NEW: Store motion anomaly scores
            local_motion_flags: List[bool] = []    # NEW: Store motion anomaly flags
            
            prev_frame_tensor = None  # For motion analysis between frames
            
            with torch.no_grad(), autocast("cuda"):
                if is_main_process():
                    val_iter = iter(validation_dataloader)
                while True:
                    if is_main_process():
                        try:
                            batch = next(val_iter)
                        except StopIteration:
                            _ = _bcast_batch_from_rank0(None)  # termination signal
                            break
                        images = (batch["image"] if isinstance(batch, dict) else batch).to(torch.float32)
                    else:
                        images = None

                    # BCHW broadcast (B==1 here)
                    images = _bcast_batch_from_rank0(images)
                    if images is None:
                        break
                    images = images.to(device, non_blocking=True)

                    B = images.shape[0]
                    
                    for b in range(B):
                        # Run inference with motion analysis
                        stitched, motion_results = infer_image_tiled_with_motion(
                            anomaly_model=anomaly_net_list[0],
                            img_tensor_1BCHW=images[b:b+1],
                            device=device,
                            til_cfg=config["tiling"],
                            is_ddp=dist_inited,
                            dataset_name=dataset_name,
                            motion_analyzer=metrics.motion_analyzer if config["motion_detection"].get("speed_analysis", False) else None,
                            prev_frame=prev_frame_tensor,
                            roi_coords=motion_roi,
                            pos_loader=pos_loader,
                            cached_coords=cached_coords,
                            hann_cache=hann_cache,
                        )
                        
                        local_maps.append(stitched)
                        
                        # Calculate appearance-based score
                        score_b = float(np.max(stitched))
                        local_scores.append(score_b)
                        
                        # Process motion results
                        if motion_results:
                            motion_score = motion_results.get('motion_anomaly_score', 0.0)
                            is_motion_anomalous = motion_results.get('is_motion_anomalous', False)
                            local_motion_scores.append(motion_score)
                            local_motion_flags.append(is_motion_anomalous)
                            
                            # Log motion anomaly if detected
                            if is_motion_anomalous:
                                LOGGER.warning(f"Motion anomaly detected: speed={motion_results['speed']:.4f}, "
                                             f"slowdown={motion_results['slowdown_ratio']:.1%}")
                        
                        # Update previous frame for next iteration
                        prev_frame_tensor = images[b:b+1].clone()

            # Gather results
            if dist_inited:
                gathered_scores = [None for _ in range(world)]
                gathered_maps = [None for _ in range(world)]
                gathered_motion_scores = [None for _ in range(world)]  # NEW
                gathered_motion_flags = [None for _ in range(world)]   # NEW
                
                dist.all_gather_object(gathered_scores, local_scores)
                dist.all_gather_object(gathered_maps, local_maps)
                dist.all_gather_object(gathered_motion_scores, local_motion_scores)  # NEW
                dist.all_gather_object(gathered_motion_flags, local_motion_flags)    # NEW
            else:
                gathered_scores = [local_scores]
                gathered_maps = [local_maps]
                gathered_motion_scores = [local_motion_scores]  # NEW
                gathered_motion_flags = [local_motion_flags]    # NEW

            # Small summary (rank0) of contributions
            if dist_inited and is_main_process():
                for r, sub in enumerate(gathered_maps):
                    t = type(sub).__name__
                    n = len(sub) if isinstance(sub, (list, tuple)) else 0
                    LOGGER.info("[VAL] rank %d contributed %d map(s), container=%s", r, n, t)

            # ---- Rank0: compute combined threshold & write maps ----
            if is_main_process():
                # Combine appearance and motion scores
                motion_weight = config["motion_detection"].get("motion_weight", 0.3)
                appearance_weight = 1.0 - motion_weight
                
                # Flatten all scores
                all_appearance_scores = []
                all_motion_scores = []
                all_combined_scores = []
                
                for i in range(len(gathered_scores)):
                    if gathered_scores[i]:
                        all_appearance_scores.extend(gathered_scores[i])
                    
                    if gathered_motion_scores[i] and len(gathered_motion_scores[i]) == len(gathered_scores[i]):
                        all_motion_scores.extend(gathered_motion_scores[i])
                    else:
                        # Pad with zeros if motion scores missing
                        all_motion_scores.extend([0.0] * len(gathered_scores[i]))
                
                # Combine scores
                for i in range(len(all_appearance_scores)):
                    combined = (appearance_weight * all_appearance_scores[i] + 
                               motion_weight * all_motion_scores[i])
                    all_combined_scores.append(combined)
                
                # Calculate dynamic threshold on combined scores
                if all_combined_scores:
                    scores_np = np.asarray(all_combined_scores, dtype=np.float32)
                    dynamic_threshold = float(np.percentile(scores_np, 95))
                    
                    LOGGER.info("Dynamic threshold (combined) for %s: %.4f", dataset_name, dynamic_threshold)
                    LOGGER.info("Appearance scores: N=%d, mean=%.4f", len(all_appearance_scores), np.mean(all_appearance_scores))
                    LOGGER.info("Motion scores: N=%d, mean=%.4f", len(all_motion_scores), np.mean(all_motion_scores))
                    LOGGER.info("Combined scores: N=%d, mean=%.4f", len(all_combined_scores), np.mean(all_combined_scores))
                    
                    # Count motion anomalies
                    motion_anomaly_count = sum(sum(flags) if flags else 0 for flags in gathered_motion_flags)
                    if motion_anomaly_count > 0:
                        LOGGER.warning(f"Detected {motion_anomaly_count} motion anomalies in validation set")
                else:
                    dynamic_threshold = 0.0
                    LOGGER.warning("No valid scores collected; defaulting dynamic threshold to 0.0")

                model_ds_dir = os.path.join(models_root, dataset_name)
                os.makedirs(model_ds_dir, exist_ok=True)
                with open(os.path.join(model_ds_dir, "dynamic_threshold.pkl"), "wb") as f:
                    pickle.dump(dynamic_threshold, f)
                
                # Save motion baseline if available
                if metrics.motion_analyzer and dataset_name in metrics.motion_analyzer.normal_speed_baselines:
                    baseline_file = os.path.join(model_ds_dir, "motion_baseline.json")
                    with open(baseline_file, 'w') as f:
                        json.dump(metrics.motion_analyzer.normal_speed_baselines[dataset_name], f, indent=2)
                    LOGGER.info(f"Motion baseline saved to {baseline_file}")

                # Process and save anomaly maps
                def _coerce_map(x) -> Optional[np.ndarray]:
                    try:
                        a = np.asarray(x)
                        if a.ndim == 2 and a.dtype.kind not in ("U", "S", "O"):
                            return a.astype(np.float32, copy=False)
                    except Exception:
                        pass
                    return None

                all_maps_raw = [m for sub in gathered_maps if sub for m in sub]
                all_maps: List[np.ndarray] = []
                for idx, m in enumerate(all_maps_raw):
                    a = _coerce_map(m)
                    if a is None:
                        LOGGER.warning("Skipping non-numeric/invalid map #%d of type=%s", idx, type(m).__name__)
                        continue
                    all_maps.append(a)

                if len(all_maps):
                    mmin = float(min(np.min(m) for m in all_maps))
                    mmax = float(max(np.max(m) for m in all_maps))
                    denom = (mmax - mmin) if (mmax > mmin) else 1.0
                    vis_maps = [((m - mmin) / denom).astype(np.float32) for m in all_maps]

                    dataset_save_path = os.path.join(run_save_path, "anomaly_maps", dataset_name)
                    os.makedirs(dataset_save_path, exist_ok=True)

                    # Try to align with (up to) K image paths
                    image_paths = []
                    try:
                        image_paths = [x[2] for x in dls["validation"].dataset.data_to_iterate]
                    except Exception:
                        pass

                    K = min(len(image_paths), len(vis_maps)) if image_paths else len(vis_maps)
                    stats_rows = anomaly_engine.core.core_utils.plot_anomaly_maps(
                        savefolder=dataset_save_path,
                        image_paths=image_paths[:K] if image_paths else None,
                        anomaly_maps=np.stack(vis_maps[:K], axis=0),
                        categories=[dataset_name] * K,
                        anomaly_scores=None,
                        image_transform=lambda x: x,
                    )
                    # Append CSV
                    try:
                        with open(stats_csv_path, mode="a", newline="") as csv_file:
                            writer = csv.writer(csv_file)
                            writer.writerows(stats_rows)
                        for cat, avg_score, max_lt1 in stats_rows:
                            LOGGER.info("Category: %s, Avg Score: %.2f, Max < 1: %.2f", cat, avg_score, max_lt1)
                    except Exception as e:
                        LOGGER.warning("Could not write stats CSV: %s", e)
                else:
                    LOGGER.warning("No images or valid anomaly maps found for dataset %s.", dataset_name)

            metrics.end_process("validation_combined")

            # ---- SAVE FINAL MODELS ----
            if config["save_model"]:
                metrics.start_process("model_saving")
                model_save_path = os.path.join(run_save_path, "models", dataset_name)
                os.makedirs(model_save_path, exist_ok=True)
                for i, anomaly_net in enumerate(anomaly_net_list):
                    prepend = f"Ensemble-{i + 1}-{len(anomaly_net_list)}_" if len(anomaly_net_list) > 1 else ""
                    anomaly_net.save_to_path(model_save_path, prepend)
                metrics.end_process("model_saving")

            # ---- METRICS JSON (rank0) ----
            metrics.finalize(anomaly_net_list)
            if is_main_process():
                metrics_json_path = os.path.join(models_root, dataset_name, "training_metrics.json")
                os.makedirs(os.path.dirname(metrics_json_path), exist_ok=True)
                metrics.save_to_json(metrics_json_path)
                # NEW: copy roi_meta.json from the good input image folder to this subdataset's output folder
                subdataset_name = getattr(dls["training"].dataset, "classname", dataset_name.split("_", 1)[-1])
                _copy_roi_meta_to_output(subdataset_name, config["dataset"]["data_path"], os.path.join(models_root, dataset_name))

            # ---- TIMING REPORT (rank0) ----
            try:
                local_snap = TRAIN_TIMERS.snapshot()
                if dist_inited:
                    gathered_t = [None for _ in range(world)]
                    dist.all_gather_object(gathered_t, local_snap)
                else:
                    gathered_t = [local_snap]
                if is_main_process():
                    combined = PhaseTimer()
                    for snap in gathered_t:
                        combined.merge_inplace(snap)
                    print("\n" + combined.report_lines(header=f"=== TIMING REPORT (dataset={dataset_name}) ===") + "\n")
            except Exception as _e:
                LOGGER.warning("Timing aggregation failed: %s", _e)

        LOGGER.info("-----")

    if dist_inited:
        cleanup_distributed()
    overall_end = time.time()
    LOGGER.info("TOTAL training+validation time: %.2f seconds (%.2f minutes)",
                overall_end - overall_start, (overall_end - overall_start)/60)
    LOGGER.info("Run complete. dist_inited=%s | world=%s",
                dist_inited, (get_rank_world()[1] if dist_inited else 1))

# ---------------------------------------------
# TRAIN TILED ITERATOR (tile-scatter DDP)
# ---------------------------------------------
class TiledTrainIterator(Iterable):
    """
        Wraps a dataloader and yields tiles instead of full images.
        DDP: rank 0 iterates the DataLoader, slices tiles once, and sends only the
        tiles belonging to each rank. Other ranks receive tile packs directly.
        """
    def __init__(
        self,
        loader,
        rows: int,
        cols: int,
        overlap: float,
        ddp_enabled: bool,
        timers=None,
        **_ignored,
    ):
        self.loader = loader
        self.rows = int(rows)
        self.cols = int(cols)
        self.overlap = float(overlap)
        self.ddp_enabled = bool(ddp_enabled)
        self.timers = timers
        self.rank, self.world = get_rank_world()
        try:
            self.my_positions = ext_shard_positions(self.rows, self.cols, self.world, self.rank) if ddp_enabled else list(range(self.rows * self.cols))
        except Exception:
            self.my_positions = [p for p in range(self.rows * self.cols) if (p % max(1, self.world)) == self.rank] if ddp_enabled else list(range(self.rows * self.cols))

        LOGGER.info(
            "[TRAIN-TILING] ddp=%s | rank=%s/%s | rows=%s cols=%s ov=%.3f | my_positions=%s",
            ddp_enabled, self.rank, self.world, self.rows, self.cols, self.overlap,
            self.my_positions[:min(8, len(self.my_positions))]
        )

    def __len__(self):
        try:
            bs = getattr(self.loader, "batch_size", 1) or 1
            tiles_share = max(1, len(self.my_positions))
            return len(self.loader) * bs * tiles_share
        except Exception:
            return len(self.loader)

    # ---------- helpers for scatter-by-tile ----------
    def _rank_tiles_map(self) -> Dict[int, List[int]]:
        m = {r: [] for r in range(self.world)}
        for r in range(self.world):
            try:
                m[r] = ext_shard_positions(self.rows, self.cols, self.world, r)
            except Exception:
                m[r] = [p for p in range(self.rows * self.cols) if (p % max(1, self.world)) == r]
        return m

    def _pack_tiles_for_positions(self, images: torch.Tensor, coords, pos_list: List[int]) -> torch.Tensor:
        if not pos_list:
            # Empty pack encoded as shape (0, C, th, tw) logically → we'll send (0,0,0,0)
            return torch.empty((0, 0, 0, 0), dtype=images.dtype)
        B, C, H, W = images.shape
        x0, y0, x1, y1 = coords[0]
        th, tw = (y1 - y0), (x1 - x0)
        out = torch.empty((B * len(pos_list), C, th, tw), dtype=images.dtype).pin_memory()
        idx = 0
        for b in range(B):
            for pos in pos_list:
                x0, y0, x1, y1 = coords[pos]
                out[idx].copy_(images[b, :, y0:y1, x0:x1])
                idx += 1
        return out

    def _send_tensor(self, tensor: torch.Tensor, dst: int, tag: int):
        # Always send a 4-int64 shape header
        if tensor is None:
            shape4 = torch.tensor([-1, -1, -1, -1], dtype=torch.int64)  # termination
            dist.send(shape4, dst=dst, tag=tag)
            return
        if tensor.numel() == 0 or tensor.ndim != 4:
            shape4 = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
            dist.send(shape4, dst=dst, tag=tag)
            return
        shape4 = torch.tensor(list(tensor.shape), dtype=torch.int64)
        dist.send(shape4, dst=dst, tag=tag)
        dist.send(tensor.view(-1), dst=dst, tag=tag + 1)

    def _recv_tensor(self, src: int, dtype: torch.dtype, tag: int) -> Optional[torch.Tensor]:
        shape4 = torch.empty((4,), dtype=torch.int64)
        dist.recv(shape4, src=src, tag=tag)
        s0, s1, s2, s3 = [int(x) for x in shape4.tolist()]
        if s0 == -1 and s1 == -1:
            return None  # termination
        if s0 == 0:
            return torch.empty((0, 0, 0, 0), dtype=dtype).pin_memory()
        t = torch.empty((s0, s1, s2, s3), dtype=dtype).pin_memory()
        dist.recv(t.view(-1), src=src, tag=tag + 1)
        return t

    def _iter_local_pack(self, pack: torch.Tensor, is_dict: bool):
        for i in range(pack.shape[0]):
            tile = pack[i:i+1]  # 1,C,th,tw
            yield {"image": tile} if is_dict else tile

    # ---------- main iterator ----------
    def __iter__(self):
        first_logged = False
        tag = 1000  # message tag space for this iterator

        # Non-main ranks: receive tile packs from rank0
        if self.ddp_enabled and self.world > 1 and self.rank != 0:
            while True:
                pack = self._recv_tensor(src=0, dtype=torch.float32, tag=tag)
                if pack is None:
                    break  # termination
                if pack.numel() == 0:
                    continue  # empty pack for this batch
                if not first_logged:
                    first_logged = True
                    _, C, th, tw = pack.shape
                    LOGGER.info("[TRAIN-TILING] recv pack: N=%d, C=%d, tile=%sx%s", pack.shape[0], C, th, tw)
                for item in self._iter_local_pack(pack, is_dict=True):
                    yield item
            return

        # Rank 0: iterate dataloader, slice once, scatter tiles
        tile_map = self._rank_tiles_map()
        for batch in self.loader:
            images = batch["image"] if isinstance(batch, dict) else batch
            assert torch.is_tensor(images) and images.dim() == 4, "Expected images tensor BxCxHxW"

            B, C, H, W = images.shape
            coords = get_tile_coords_cached(H, W, self.rows, self.cols, self.overlap)

            if not first_logged:
                first_logged = True
                LOGGER.info(
                    "[TRAIN-TILING] sample batch shape BxCxHxW = %s | first tile=(x0=%s,y0=%s,x1=%s,y1=%s) tile_size=%sx%s",
                    (B, C, H, W), *coords[0], coords[0][3]-coords[0][1], coords[0][2]-coords[0][0]
                )

            # send per-rank packs
            for r in range(1, self.world):
                pack_r = self._pack_tiles_for_positions(images, coords, tile_map[r])
                self._send_tensor(pack_r, dst=r, tag=tag)

            # yield local tiles for rank0
            local_pack = self._pack_tiles_for_positions(images, coords, tile_map[0])
            if local_pack.numel() > 0:
                for item in self._iter_local_pack(local_pack, is_dict=isinstance(batch, dict)):
                    yield item

        # termination signals
        if self.ddp_enabled and self.world > 1 and self.rank == 0:
            for r in range(1, self.world):
                self._send_tensor(None, dst=r, tag=tag)

# ---------------------------------------------
# Entrypoint
# ---------------------------------------------
if __name__ == "__main__":
    if not (dist.is_available() and dist.is_initialized()) or get_rank_world()[0] == 0:
        LOGGER.info("Starting anomaly detection with configuration")
    main()