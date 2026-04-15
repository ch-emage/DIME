#!/usr/bin/env python3
"""
Anomaly Detection Inference Script (Parallel Tile Processing)
- Optimized for external calling with frame-by-frame processing
- No DDP, uses parallel tile processing instead
- Returns coordinates of all anomaly areas in respect to full frame
"""

import os
import sys
import argparse
import logging
import time
import pickle
import glob
import datetime
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision
from collections import deque
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict, Any

# monitoring
import json
import psutil
import GPUtil

# Parallel processing
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Critical CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Import your custom modules
# try:
from dime_v2.anomaly_engine.core.anomaly_net import AnomalyNet
from dime_v2.anomaly_engine.core.core_utils import ProximitySearcher, FeatureExtractor
# except Exception as e:
#     print(f"Error importing anomaly detection modules: {e}")
#     print("Please ensure the anomaly_engine module is in your Python path")
    # sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------ Resource Monitor ------------------------------------------------------
class ResourceMonitor:
    """Monitor system resources during processing"""
    def __init__(self):
        self.cpu_percentages = []
        self.ram_usages = []
        self.gpu_usages = []
        self.gpu_memories = []
        self.start_time = None
        self.end_time = None

    def start_monitoring(self):
        self.start_time = time.time()
        self.cpu_percentages.clear()
        self.ram_usages.clear()
        self.gpu_usages.clear()
        self.gpu_memories.clear()

    def take_sample(self):
        self.cpu_percentages.append(psutil.cpu_percent())
        process = psutil.Process()
        ram_info = process.memory_info()
        self.ram_usages.append(ram_info.rss / (1024 * 1024))
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_usages.append(gpus[0].load * 100)
                self.gpu_memories.append(gpus[0].memoryUsed)
        except Exception:
            pass

    def stop_monitoring(self):
        self.end_time = time.time()
        stats = {
            'monitoring_duration_seconds': self.end_time - self.start_time if self.start_time else 0.0,
            'cpu_usage_percent': {
                'average': float(np.mean(self.cpu_percentages)) if self.cpu_percentages else 0.0,
                'max': float(np.max(self.cpu_percentages)) if self.cpu_percentages else 0.0,
                'samples': len(self.cpu_percentages)
            },
            'ram_usage_mb': {
                'average': float(np.mean(self.ram_usages)) if self.ram_usages else 0.0,
                'max': float(np.max(self.ram_usages)) if self.ram_usages else 0.0,
                'samples': len(self.ram_usages)
            }
        }
        if self.gpu_usages:
            stats['gpu_usage_percent'] = {
                'average': float(np.mean(self.gpu_usages)),
                'max': float(np.max(self.gpu_usages)),
                'samples': len(self.gpu_usages)
            }
            stats['gpu_memory_mb'] = {
                'average': float(np.mean(self.gpu_memories)),
                'max': float(np.max(self.gpu_memories)),
                'samples': len(self.gpu_memories)
            }
        return stats

# ------ Tiling + Stitching ----------------------------------------------------
def _compute_tile_coords(W, H, tile_rows, tile_cols, tile_overlap=0.0):
    assert 0.0 <= tile_overlap < 1.0
    R, Cn = int(tile_rows), int(tile_cols)
    base_h = max(1, H // R)
    base_w = max(1, W // Cn)
    overlap_h = int(round(base_h * tile_overlap))
    overlap_w = int(round(base_w * tile_overlap))
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
            x0 = int(max(0, min(x0, W - 1)))
            y0 = int(max(0, min(y0, H - 1)))
            x1 = int(max(x0 + 1, min(x1, W)))
            y1 = int(max(y0 + 1, min(y1, H)))
            coords.append((x0, y0, x1, y1))
    return coords

def _stitch_tiles(tile_list, H, W, eps=1e-8):
    acc  = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    weight_cache = {}
    for t in tile_list:
        x0, y0, x1, y1 = t["coords"]
        seg = t["seg"]
        th, tw = y1 - y0, x1 - x0
        if seg.shape[:2] != (th, tw):
            seg = cv2.resize(seg, (tw, th), interpolation=cv2.INTER_LINEAR)
        key = (th, tw)
        if key not in weight_cache:
            wy = np.hanning(max(th, 2))[:, None]
            wx = np.hanning(max(tw, 2))[None, :]
            w  = (wy * wx).astype(np.float32)
            w /= (w.max() + eps)
            weight_cache[key] = w
        w = weight_cache[key]
        acc[y0:y1, x0:x1]  += seg * w
        wsum[y0:y1, x0:x1] += w
    wsum[wsum == 0] = 1.0
    stitched = acc / wsum
    return stitched

# ------ Parallel Tile Processing ----------------------------------------------
def _process_single_tile_worker(args):
    """Worker function for processing a single tile in parallel"""
    tile_data, model_path, device_str, imagesize, use_compile = args
    tile_rgb, coords, pos = tile_data
    
    try:
        import torch
        import cv2
        import numpy as np
        from .anomaly_engine.core.anomaly_net import AnomalyNet
        from .anomaly_engine.core.core_utils import ProximitySearcher
        
        device = torch.device(device_str)
        
        # Load model for this worker
        nn_method = ProximitySearcher(False, num_workers=2)
        model = AnomalyNet(device=device)
        model.load_from_path(model_path, device=device, nn_method=nn_method)
        
        if torch.cuda.is_available():
            model = model.half()
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        if use_compile and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
            except Exception:
                pass
        
        # Process tile
        Ht, Wt = tile_rgb.shape[:2]
        
        if imagesize is not None:
            tH, tW = imagesize
            tile_np = cv2.resize(tile_rgb, (int(tW), int(tH)), interpolation=cv2.INTER_LINEAR)
        else:
            tile_np = tile_rgb
        
        # Preprocess
        img_float = tile_np.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(device, non_blocking=True)
        
        # Inference
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, segmentations = model._predict(tensor)
        
        seg = segmentations[0]
        if torch.is_tensor(seg):
            seg = seg.detach().float().cpu().numpy()
        seg = np.asarray(seg, dtype=np.float32)
        if seg.ndim == 3:
            seg = np.mean(seg, axis=0)
        
        if seg.shape[:2] != (Ht, Wt):
            seg = cv2.resize(seg, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
        
        return {"pos": pos, "coords": coords, "seg": seg}
        
    except Exception as e:
        print(f"Error in tile worker {pos}: {e}")
        return {"pos": pos, "coords": coords, "seg": np.zeros((coords[3]-coords[1], coords[2]-coords[0]), dtype=np.float32)}

class ParallelTileProcessor:
    """Multi-process tile processor for parallel inference"""
    def __init__(self, model_path, device, imagesize=None, use_compile=True, num_workers=4):
        self.model_path = model_path
        self.device = device
        self.imagesize = imagesize
        self.use_compile = use_compile
        self.num_workers = num_workers
        
        mp.set_start_method('spawn', force=True)
        self.pool = ProcessPoolExecutor(max_workers=num_workers)
        
        logger.info(f"[ParallelTileProcessor] Initialized with {num_workers} workers")

    def process_tiles_parallel(self, frame_rgb, coords):
        """Process all tiles in parallel using worker processes"""
        tile_data_list = []
        for pos, (x0, y0, x1, y1) in enumerate(coords):
            tile_rgb = frame_rgb[y0:y1, x0:x1, :].copy()
            tile_data_list.append((tile_rgb, (x0, y0, x1, y1), pos))
        
        worker_args = [
            (tile_data, self.model_path, str(self.device), self.imagesize, self.use_compile)
            for tile_data in tile_data_list
        ]
        
        futures = [self.pool.submit(_process_single_tile_worker, args) for args in worker_args]
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30.0)
                results.append(result)
            except Exception as e:
                logger.error(f"Tile processing failed: {e}")
                if 'coords' in locals():
                    x0, y0, x1, y1 = coords
                    results.append({
                        "pos": len(results),
                        "coords": (x0, y0, x1, y1),
                        "seg": np.zeros((y1-y0, x1-x0), dtype=np.float32)
                    })
        
        return results

    def __del__(self):
        if hasattr(self, 'pool'):
            self.pool.shutdown(wait=True)

def _is_multi_model_root(model_root: str) -> bool:
    try:
        subdirs = [d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))]
    except Exception:
        return False
    hits = 0
    for d in subdirs:
        p = os.path.join(model_root, d)
        if os.path.isfile(os.path.join(p, "roi_meta.json")) and (
           os.path.isfile(os.path.join(p, "dime_params.pkl")) or
           os.path.isfile(os.path.join(p, "nnscorer_search_index.faiss"))):
            hits += 1
    return hits >= 1

def _read_roi_meta(detector_dir: str):
    meta_path = os.path.join(detector_dir, "roi_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    rect = meta.get("rectangle") or {}
    poly = meta.get("polygon") or []
    imsz = meta.get("image_size") or {}
    Wm, Hm = int(imsz.get("width", 0)), int(imsz.get("height", 0))
    return rect, poly, (Wm, Hm)

def _polygon_mask_for_rect(rect_xywh, poly_scaled, H_full, W_full):
    x, y, w, h = rect_xywh
    if w <= 0 or h <= 0:
        return np.zeros((1, 1), np.uint8)
    local = np.zeros((h, w), np.uint8)
    if poly_scaled:
        pts = np.array([ [px - x, py - y] for (px,py) in poly_scaled ], dtype=np.int32)
        cv2.fillPoly(local, [pts], color=1)
    else:
        local[:] = 1
    return local

def _masked_max_inside_polygon(seg_map_rect_f32: np.ndarray, poly_mask_uint8: np.ndarray) -> float:
    if seg_map_rect_f32 is None or seg_map_rect_f32.size == 0:
        return 0.0
    if poly_mask_uint8 is None or poly_mask_uint8.size == 0:
        return float(np.max(seg_map_rect_f32))
    m = (poly_mask_uint8.astype(bool))
    if seg_map_rect_f32.shape[:2] != m.shape[:2]:
        seg_map_rect_f32 = cv2.resize(seg_map_rect_f32, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_LINEAR)
    if not np.any(m):
        return 0.0
    return float(np.max(seg_map_rect_f32[m]))

# --------- Simple profiler ----------
import collections
from contextlib import contextmanager
import csv

ANOM_PROF = bool(int(os.environ.get("ANOM_PROF", "1")))

def _sync_if_cuda():
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

class FrameProfiler:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.sections = collections.OrderedDict()

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        _sync_if_cuda()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync_if_cuda()
            dt = (time.perf_counter() - t0) * 1000.0
            self.sections[name] = self.sections.get(name, 0.0) + dt

    def merge(self, child: "FrameProfiler", prefix: str = None):
        if not (self.enabled and child and child.sections):
            return
        for k, v in child.sections.items():
            name = f"{prefix}{k}" if prefix else k
            self.sections[name] = self.sections.get(name, 0.0) + v

    def dump(self, prefix=""):
        if not self.enabled or not self.sections:
            return
        parts = [f"{k}={v:.2f}ms" for k, v in self.sections.items()]
        logger.info(f"{prefix}[PROF] " + " | ".join(parts))

# --------- NEW: Anomaly Detection Result Class ----------
class AnomalyDetectionResult:
    """Container for anomaly detection results with coordinates"""
    def __init__(self, 
                 processed_frame: np.ndarray,
                 anomaly_score: float,
                 is_anomaly: bool,
                 anomaly_areas: List[Dict[str, Any]],
                 segmentation_map: Optional[np.ndarray] = None):
        self.processed_frame = processed_frame
        self.anomaly_score = anomaly_score
        self.is_anomaly = is_anomaly
        self.anomaly_areas = anomaly_areas  # List of dicts with coordinates and metadata
        self.segmentation_map = segmentation_map
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for easy serialization"""
        return {
            'anomaly_score': self.anomaly_score,
            'is_anomaly': self.is_anomaly,
            'anomaly_areas': self.anomaly_areas,
            'frame_shape': self.processed_frame.shape if self.processed_frame is not None else None
        }

def _extract_anomaly_areas(segmentation_map: np.ndarray, threshold: float, 
                          min_area: int = 50, min_aspect_ratio: float = 0.125, 
                          max_aspect_ratio: float = 8.0, min_std: float = 0.05) -> List[Dict[str, Any]]:
    """
    Extract anomaly areas from segmentation map with filtering
    Returns list of dictionaries with coordinates and metadata
    """
    if segmentation_map is None or segmentation_map.size == 0:
        return []
    
    # Create binary mask
    binary_mask = (segmentation_map >= threshold).astype(np.uint8) * 255
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    anomaly_areas = []
    
    for i in range(1, num_labels):  # Skip background (label 0)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter by area
        if area < min_area:
            continue
            
        # Filter by aspect ratio
        aspect_ratio = w / max(h, 1)
        if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
            continue
            
        # Filter by intensity variation
        patch = segmentation_map[y:y+h, x:x+w]
        if patch.size > 0 and np.std(patch) < min_std:
            continue
        
        # Calculate confidence (max value in this region)
        confidence = float(np.max(segmentation_map[y:y+h, x:x+w]))
        
        anomaly_areas.append({
            'bbox': [x, y, w, h],  # [x, y, width, height]
            'area': area,
            'confidence': confidence,
            'centroid': [int(centroids[i][0]), int(centroids[i][1])],
            'aspect_ratio': aspect_ratio
        })
    
    # Sort by confidence (highest first)
    anomaly_areas.sort(key=lambda x: x['confidence'], reverse=True)
    
    return anomaly_areas

class AnomalyInference:
    def __init__(self, model_path, threshold=None, imagesize=None,
                skip_frames=3, mask_alpha=0.9, proximity_on_gpu=False,
                tile_rows=1, tile_cols=1, tile_overlap=0.0,
                parallel_tiles=True, num_workers=4, same_gpu=False,
                resize=None, resource_sample_interval=2.0,
                output_path: Optional[str] = None,
                # NEW: Anomaly area extraction parameters
                min_anomaly_area: int = 50,
                min_aspect_ratio: float = 0.125,
                max_aspect_ratio: float = 8.0,
                min_intensity_std: float = 0.05):

        # Store parameters
        self.model_path = model_path
        self.threshold = threshold
        self.imagesize = None
        self.skip_frames = int(skip_frames)
        self.mask_alpha = float(mask_alpha)
        self.proximity_on_gpu = bool(proximity_on_gpu)
        self.tile_rows, self.tile_cols = int(tile_rows), int(tile_cols)
        self.tile_overlap = float(tile_overlap)
        self.output_path = output_path
        self.resource_sample_interval = float(resource_sample_interval)
        self.resize = resize
        
        # NEW: Anomaly area extraction parameters
        self.min_anomaly_area = min_anomaly_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_intensity_std = min_intensity_std

        try:
            # Only adjust if this is NOT a multi-ROI root
            if not _is_multi_model_root(self.model_path):
                # Does current dir already look like a model folder?
                has_model_here = (
                    os.path.isfile(os.path.join(self.model_path, "nnscorer_search_index.faiss")) or
                    bool(glob.glob(os.path.join(self.model_path, "*.faiss"))) or
                    os.path.isfile(os.path.join(self.model_path, "dime_params.pkl"))
                )
                if not has_model_here:
                    # Look exactly one level down for a unique child that looks like a model folder
                    subs = [
                        os.path.join(self.model_path, d)
                        for d in os.listdir(self.model_path)
                        if os.path.isdir(os.path.join(self.model_path, d))
                    ]
                    candidates = [
                        d for d in subs
                        if (
                            os.path.isfile(os.path.join(d, "nnscorer_search_index.faiss")) or
                            glob.glob(os.path.join(d, "*.faiss")) or
                            os.path.isfile(os.path.join(d, "dime_params.pkl"))
                        )
                    ]
                    if len(candidates) == 1:
                        logger.warning(f"[single-detector] Adjusting model_path: {self.model_path} -> {candidates[0]}")
                        self.model_path = candidates[0]
        except Exception:
            # Fail-safe: leave path unchanged if anything odd happens
            pass
        
        # Parallel processing settings (can be modified after initialization)
        self.parallel_tiles = parallel_tiles
        self.num_workers = num_workers
        self.parallel_processor = None
        
        # Optimization settings (can be modified after initialization)
        self.use_compile = True
        self.use_pinned_memory = False
        self.resolution_scale = 1
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using device: {self.device}")
        else:
            self.device = torch.device("cpu")
            logger.info(f"Using device: {self.device}")

        self.resource_monitor = ResourceMonitor()

        if not torch.cuda.is_available() and self.proximity_on_gpu:
            logger.warning("CUDA not available, falling back to CPU for FAISS")
            self.proximity_on_gpu = False

        # transforms
        self.tfm = self.preprocess_image()

        # Load model parameters + threshold
        self.load_model_params(threshold)

        # temporal
        self.temporal_buffer = deque(maxlen=5)
        
        # Model holder
        self.model = None

        # single-frame mode state
        self._sf_frame_idx = 0
        self._sf_last_segmentation = None

        # profiling state
        self._prof_enabled = ANOM_PROF

    def load_model_params(self, threshold):
        params_file = os.path.join(self.model_path, "dime_params.pkl")
        if not os.path.exists(params_file):
            cand = glob.glob(os.path.join(self.model_path, "**", "pos*", "dime_params.pkl"), recursive=True)
            if cand:
                params_file = cand[0]
        if os.path.exists(params_file):
            with open(params_file, "rb") as f:
                params = pickle.load(f)
            inp = params.get("input_shape", None)
            if isinstance(inp, (tuple, list)):
                if len(inp) == 3:
                    _, H, W = inp
                elif len(inp) == 2:
                    H, W = inp
                else:
                    H = W = None
                if H is not None and W is not None:
                    self.imagesize = (int(H), int(W))
                    logger.info(f"Using training image size from params: {self.imagesize}")
        else:
            self.imagesize = None
            logger.warning("Params file not found; imagesize will be derived per-tile.")

        if threshold is None:
            threshold_file = os.path.join(self.model_path, "dynamic_threshold.pkl")
            if os.path.exists(threshold_file):
                with open(threshold_file, "rb") as f:
                    self.threshold = pickle.load(f) * 1.25
                    # self.threshold = 170
                logger.info(f"Loaded dynamic threshold: {self.threshold:.4f}")
            else:
                self.threshold = 0.5
                logger.warning(f"Dynamic threshold not found; using default: {self.threshold:.4f}")
        else:
            self.threshold = threshold
            logger.info(f"Using provided threshold: {self.threshold:.4f}")

    def load_model(self):
        """Load the model - call this after initialization before processing frames"""
        if self.model is not None:
            return self.model
            
        try:
            # Try per-position models first
            pos_models = self._load_models_by_position()
            if pos_models is not None:
                self.model = pos_models
                logger.info(f"Loaded per-position models for positions: {sorted(self.model.keys())}")
                return self.model

            # Monolithic fallback
            faiss_index_path = os.path.join(self.model_path, "nnscorer_search_index.faiss")
            if not os.path.exists(faiss_index_path):
                found = glob.glob(os.path.join(self.model_path, "*.faiss"))
                if not found:
                    raise FileNotFoundError(f"No FAISS index found in {self.model_path}")
                    
            nn_method = ProximitySearcher(self.proximity_on_gpu, num_workers=4)
            model = AnomalyNet(device=self.device)
            model.load_from_path(self.model_path, device=self.device, nn_method=nn_method)
            
            if torch.cuda.is_available():
                model = model.half()
                logger.info("Using FP16 precision for faster inference")
            
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            
            if hasattr(torch, 'compile') and self.use_compile:
                try:
                    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                    logger.info("Model compiled with torch.compile for faster inference")
                except Exception as e:
                    logger.warning(f"torch.compile failed, continuing without it: {e}")
            
            self.model = model
            
            # Initialize parallel processor if enabled
            if self.parallel_tiles:
                self.parallel_processor = ParallelTileProcessor(
                    model_path=self.model_path,
                    device=self.device,
                    imagesize=self.imagesize,
                    use_compile=self.use_compile,
                    num_workers=self.num_workers
                )
                logger.info(f"Parallel tile processor initialized with {self.num_workers} workers")
            
            # Warmup
            if torch.cuda.is_available():
                try:
                    if self.imagesize:
                        dummy_input = torch.randn(1, 3, self.imagesize[0], self.imagesize[1], 
                                                 device=self.device, dtype=torch.float16)
                        with torch.no_grad():
                            with torch.amp.autocast("cuda"):
                                _ = model._predict(dummy_input)
                        torch.cuda.synchronize()
                        logger.info("Model warmup completed")
                except Exception as e:
                    logger.warning(f"Model warmup failed: {e}")
            
            logger.info(f"Loaded model from: {self.model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_models_by_position(self):
        pos_dirs = [p for p in glob.glob(os.path.join(self.model_path, "**", "pos*"), recursive=True) if os.path.isdir(p)]
        if not pos_dirs:
            return None
        models = {}
        # self.layers_to_extract_from = ["layer2", "layer3", "layer4"]
        # self.backbone = eval("torchvision.models.wide_resnet101_2(weights='IMAGENET1K_V2')")
        # feature_extractor = FeatureExtractor(
        #     self.backbone, self.layers_to_extract_from, self.device
        # ).half().to(self.device)
        # feature_extractor = None
        for p in pos_dirs:
            base = os.path.basename(p)
            if base.startswith("pos") and base[3:].isdigit():
                pos_idx = int(base[3:])
                nn_method = ProximitySearcher(self.proximity_on_gpu, num_workers=4)
                m = AnomalyNet(device=self.device)
                m.load_from_path(p, device=self.device, nn_method=nn_method)
                if torch.cuda.is_available():
                    m = m.half()
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
                if hasattr(torch, 'compile') and self.use_compile:
                    try:
                        m = torch.compile(m, mode="reduce-overhead", fullgraph=True)
                    except Exception:
                        pass
                models[pos_idx] = m
        return models or None

    @staticmethod
    def preprocess_image(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    def _fast_preprocess(self, img_rgb_np):
        img_float = img_rgb_np.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).contiguous()
        if self.use_pinned_memory:
            tensor = tensor.pin_memory()
        return tensor.unsqueeze(0)

    # def add_annotations(self, frame, score, processing_time_ms, fps=None, anomaly_areas=None):
    #     h, w = frame.shape[:2]
    #     is_anom = (score >= self.threshold)

    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     big_scale, big_th = 1.1, 3
    #     small_scale, small_th = 0.7, 2
    #     pad = 12
    #     fg_ok  = (0, 255, 0)
    #     fg_bad = (0, 0, 255)
    #     fg_txt = (255, 255, 255)
    #     bg     = (0, 0, 0)

    #     # Left-top: STATUS (big)
    #     status = "ANOMALY" if is_anom else "NORMAL"
    #     size = cv2.getTextSize(status, font, big_scale, big_th)[0]
    #     xL, yT = pad, pad + size[1]
    #     cv2.rectangle(frame, (xL - 6, yT - size[1] - 6), (xL + size[0] + 6, yT + 6), bg, -1)
    #     cv2.putText(frame, status, (xL, yT), font, big_scale, fg_bad if is_anom else fg_ok, big_th, cv2.LINE_AA)

    #     # Top-middle: FPS (small)
    #     if fps is None:
    #         fps = 1000.0 / processing_time_ms if processing_time_ms > 1e-6 else 0.0
    #     fps_text = f"FPS: {fps:.1f}"
    #     size = cv2.getTextSize(fps_text, font, small_scale, small_th)[0]
    #     xM = (w - size[0]) // 2
    #     yM = pad + size[1]
    #     cv2.rectangle(frame, (xM - 6, yM - size[1] - 6), (xM + size[0] + 6, yM + 6), bg, -1)
    #     cv2.putText(frame, fps_text, (xM, yM), font, small_scale, fg_txt, small_th, cv2.LINE_AA)

    #     # Right-top: Processing ms (big)
    #     proc = f"{processing_time_ms:.1f} ms"
    #     size = cv2.getTextSize(proc, font, big_scale, big_th)[0]
    #     xR = w - size[0] - pad
    #     yR = pad + size[1]
    #     cv2.rectangle(frame, (xR - 6, yR - size[1] - 6), (xR + size[0] + 6, yR + 6), bg, -1)
    #     cv2.putText(frame, proc, (xR, yR), font, big_scale, fg_txt, big_th, cv2.LINE_AA)

    #     # NEW: Draw anomaly areas if provided
    #     if anomaly_areas and is_anom:
    #         for i, area in enumerate(anomaly_areas):
    #             x, y, w_bbox, h_bbox = area['bbox']
    #             confidence = area['confidence']
                
    #             # Draw bounding box
    #             # cv2.rectangle(frame, (x, y), (x + w_bbox, y + h_bbox), (0, 0, 255), 2)
    #             cv2.rectangle(frame, (max(0, x-100), max(0, y-100)), (min(frame.shape[1], x+w_bbox+100), min(frame.shape[0], y+h_bbox+100)), (0, 0, 255), 2)

                
    #             # Draw label with confidence
    #             label = f"Anom {i+1}: {confidence:.2f}"
    #             label_size = cv2.getTextSize(label, font, 0.5, 1)[0]
    #             cv2.rectangle(frame, (x, y - label_size[1] - 5), 
    #                          (x + label_size[0], y), (0, 0, 255), -1)
    #             cv2.putText(frame, label, (x, y - 5), font, 0.5, (255, 255, 255), 1)

    #     return frame


    def add_annotations(self, frame, score, processing_time_ms, fps=None, anomaly_areas=None):
        h, w = frame.shape[:2]
        is_anom = (score >= self.threshold)

        font = cv2.FONT_HERSHEY_SIMPLEX
        big_scale, big_th = 1.1, 3
        small_scale, small_th = 0.7, 2
        pad = 12

        fg_ok  = (0, 255, 0)
        fg_bad = (0, 0, 255)
        fg_txt = (255, 255, 255)
        bg     = (0, 0, 0)

        # ===================== STATUS (TOP-LEFT) =====================
        status = "ANOMALY" if is_anom else "NORMAL"
        size = cv2.getTextSize(status, font, big_scale, big_th)[0]
        xL, yT = pad, pad + size[1]
        cv2.rectangle(frame, (xL - 6, yT - size[1] - 6), (xL + size[0] + 6, yT + 6), bg, -1)
        cv2.putText(frame, status, (xL, yT), font, big_scale,
                    fg_bad if is_anom else fg_ok, big_th, cv2.LINE_AA)

        # ===================== FPS (TOP-CENTER) =====================
        if fps is None:
            fps = 1000.0 / processing_time_ms if processing_time_ms > 1e-6 else 0.0

        fps_text = f"FPS: {fps:.1f}"
        size = cv2.getTextSize(fps_text, font, small_scale, small_th)[0]
        xM = (w - size[0]) // 2
        yM = pad + size[1]
        cv2.rectangle(frame, (xM - 6, yM - size[1] - 6), (xM + size[0] + 6, yM + 6), bg, -1)
        cv2.putText(frame, fps_text, (xM, yM), font, small_scale, fg_txt, small_th, cv2.LINE_AA)

        # ===================== PROCESSING TIME (TOP-RIGHT) =====================
        proc = f"{processing_time_ms:.1f} ms"
        size = cv2.getTextSize(proc, font, big_scale, big_th)[0]
        xR = w - size[0] - pad
        yR = pad + size[1]
        cv2.rectangle(frame, (xR - 6, yR - size[1] - 6), (xR + size[0] + 6, yR + 6), bg, -1)
        cv2.putText(frame, proc, (xR, yR), font, big_scale, fg_txt, big_th, cv2.LINE_AA)

        # ===================== ANOMALY BOUNDING BOXES =====================
        if anomaly_areas and is_anom:
            for i, area in enumerate(anomaly_areas):
                x, y, bw, bh = area['bbox']
                confidence = area['confidence']

                # Expand bounding box by 100px on each side
                pad_box = 100
                x1 = max(0, x - pad_box)
                y1 = max(0, y - pad_box)
                x2 = min(w, x + bw + pad_box)
                y2 = min(h, y + bh + pad_box)

                # Draw expanded bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Label
                label = f"Anom {i+1}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, font, 0.5, 1)[0]

                ly1 = max(0, y1 - label_size[1] - 6)
                ly2 = y1
                lx1 = x1
                lx2 = min(w, x1 + label_size[0] + 6)

                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), -1)
                cv2.putText(frame, label, (lx1 + 3, ly2 - 4),
                            font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame


    def _predict_tile(self, model, tile_rgb_np):
        tile_prof = FrameProfiler(self._prof_enabled)

        with tile_prof.section("prep_pil"):
            Ht, Wt = tile_rgb_np.shape[:2]
            tile_np = tile_rgb_np

        with tile_prof.section("resize_to_train"):
            if self.imagesize is not None:
                tH, tW = self.imagesize
                scale = getattr(self, 'resolution_scale', 1.0)
                if scale != 1.0:
                    tH, tW = int(tH * scale), int(tW * scale)
                tile_np = cv2.resize(tile_np, (int(tW), int(tH)), interpolation=cv2.INTER_LINEAR)

        with tile_prof.section("tfm"):
            x = self._fast_preprocess(tile_np)

        with tile_prof.section("to_device"):
            x = x.to(self.device, non_blocking=True)

        with tile_prof.section("forward"):
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _, segmentations = model._predict(x)

        with tile_prof.section("to_numpy+reduce"):
            seg = segmentations[0]
            if torch.is_tensor(seg):
                seg = seg.detach().float().cpu().numpy()
            seg = np.asarray(seg, dtype=np.float32)
            if seg.ndim == 3:
                seg = np.mean(seg, axis=0)

        with tile_prof.section("resize_back"):
            if seg.shape[:2] != (Ht, Wt):
                seg = cv2.resize(seg, (Wt, Ht), interpolation=cv2.INTER_LINEAR)

        return seg, tile_prof


    def process_frame(self, frame, frame_idx=None, last_segmentation=None):
        """
        Main method to process a single frame
        Returns: AnomalyDetectionResult object with coordinates
        """
        if frame_idx is None:
            frame_idx = self._sf_frame_idx
            
        prof = FrameProfiler(self._prof_enabled)

        # recompute or reuse?
        recompute = int((frame_idx % (self.skip_frames + 1) == 0) or (last_segmentation is None))

        # ===== SKIP-FRAME FAST PATH =====
        if not recompute:
            with prof.section("skip.prep"):
                frame_start_time = time.time()
                original_frame = frame.copy()
                H, W = original_frame.shape[:2]
                current_segmentation = last_segmentation
                
                # IMPORTANT: Get the last anomaly areas
                # Store anomaly areas in the instance for skip frames
                if not hasattr(self, '_sf_last_anomaly_areas'):
                    self._sf_last_anomaly_areas = []

            with prof.section("skip.temporal_smooth"):
                self.temporal_buffer.append(current_segmentation)
                smoothed_map = (np.mean(self.temporal_buffer, axis=0)
                                if len(self.temporal_buffer) > 0 else current_segmentation)
                anomaly_score = float(np.max(smoothed_map)) if smoothed_map is not None else 0.0

            with prof.section("skip.threshold"):
                if smoothed_map is None:
                    m = None
                else:
                    m = smoothed_map.copy()
                    m[m < self.threshold] = 0
                    m[m >= self.threshold] = 1
                    if m.ndim == 3:
                        m = m.mean(axis=0)

            with prof.section("skip.mask_resize"):
                mask = np.zeros((H, W), dtype=bool) if m is None else \
                    cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

            # NEW: Use last anomaly areas instead of recalculating
            with prof.section("skip.reuse_areas"):
                # If we have stored anomaly areas from last compute frame, use them
                if hasattr(self, '_sf_last_anomaly_areas') and self._sf_last_anomaly_areas:
                    anomaly_areas = self._sf_last_anomaly_areas.copy()  # Use copy to avoid modification
                else:
                    # Fallback: calculate from mask (original behavior)
                    anomaly_areas = []
                    if np.any(mask):
                        mask_uint8 = (mask * 255).astype(np.uint8)
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                        
                        for i in range(1, num_labels):  # Skip background
                            x = stats[i, cv2.CC_STAT_LEFT]
                            y = stats[i, cv2.CC_STAT_TOP]
                            w = stats[i, cv2.CC_STAT_WIDTH]
                            h = stats[i, cv2.CC_STAT_HEIGHT]
                            area = stats[i, cv2.CC_STAT_AREA]
                            
                            if area >= self.min_anomaly_area:
                                patch = current_segmentation[y:y+h, x:x+w] if current_segmentation is not None else None
                                confidence = float(np.max(patch)) if patch is not None and patch.size > 0 else 1.0
                                
                                anomaly_areas.append({
                                    'bbox': [x, y, w, h],
                                    'area': area,
                                    'confidence': confidence,
                                    'centroid': [int(centroids[i][0]), int(centroids[i][1])]
                                })

            with prof.section("skip.overlay"):
                frame_out = original_frame
                if np.any(mask):
                    overlay = frame_out.copy()
                    overlay[mask] = [0, 0, 255]
                    frame_out = cv2.addWeighted(overlay, self.mask_alpha, frame_out, 1 - self.mask_alpha, 0)

            with prof.section("skip.annotate"):
                processing_time_ms = (time.time() - frame_start_time) * 1000.0
                inst_fps = 1000.0 / processing_time_ms if processing_time_ms > 1e-6 else 0.0
                
                # IMPORTANT: Show annotations using the preserved anomaly areas
                frame_out = self.add_annotations(frame_out, anomaly_score, processing_time_ms, inst_fps, anomaly_areas)

            prof.dump(prefix=f"[f{frame_idx:05d}] ")
            
            return AnomalyDetectionResult(
                processed_frame=frame_out,
                anomaly_score=anomaly_score,
                is_anomaly=anomaly_score >= self.threshold,
                anomaly_areas=anomaly_areas,  # Use preserved/recalculated areas
                segmentation_map=current_segmentation
            )

        # ===== COMPUTE-FRAME PATH =====
        with prof.section("prep.copy"):
            frame_start_time = time.time()
            original_frame = frame.copy()
            H, W = original_frame.shape[:2]

        with prof.section("prep.cvt_color"):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with prof.section("tiling.compute_coords"):
            coords = _compute_tile_coords(W, H, self.tile_rows, self.tile_cols, self.tile_overlap)
            tile_count = len(coords)

        with prof.section("tiling.predict_tiles"):
            per_tile_payload = []
            
            if self.parallel_tiles and self.parallel_processor:
                per_tile_payload = self.parallel_processor.process_tiles_parallel(frame_rgb, coords)
            else:
                for pos, (x0, y0, x1, y1) in enumerate(coords):
                    tile_rgb = frame_rgb[y0:y1, x0:x1, :]
                    model_for_pos = self.model[pos] if (isinstance(self.model, dict) and pos in self.model) else self.model
                    seg, tile_prof = self._predict_tile(model_for_pos, tile_rgb)
                    prof.merge(tile_prof, prefix="tile.")
                    per_tile_payload.append({
                        "pos": pos,
                        "coords": (x0, y0, x1, y1),
                        "seg": seg
                    })

        with prof.section("stitch.assemble"):
            current_segmentation = _stitch_tiles(per_tile_payload, H, W)

        with prof.section("post.temporal_smooth"):
            self.temporal_buffer.append(current_segmentation)
            smoothed_map = np.mean(self.temporal_buffer, axis=0) if len(self.temporal_buffer) > 0 else current_segmentation
            anomaly_score = float(np.max(smoothed_map))

        with prof.section("post.threshold"):
            m = smoothed_map.copy()
            m[m < self.threshold] = 0
            m[m >= self.threshold] = 1
            if m.ndim == 3:
                m = m.mean(axis=0)

        with prof.section("post.mask_resize"):
            mask = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        # NEW: Extract anomaly areas and store them for skip frames
        with prof.section("post.extract_areas"):
            anomaly_areas = _extract_anomaly_areas(
                current_segmentation, 
                self.threshold,
                self.min_anomaly_area,
                self.min_aspect_ratio,
                self.max_aspect_ratio,
                self.min_intensity_std
            )
            
            # IMPORTANT: Store anomaly areas for skip frames
            self._sf_last_anomaly_areas = anomaly_areas.copy()  # Store for skip frames

        with prof.section("post.overlay"):
            frame_out = original_frame
            if np.any(mask):
                overlay = frame_out.copy()
                overlay[mask] = [0, 0, 255]
                frame_out = cv2.addWeighted(overlay, self.mask_alpha, frame_out, 1 - self.mask_alpha, 0)

        with prof.section("post.annotate"):
            processing_time_ms = (time.time() - frame_start_time) * 1000.0
            frame_out = self.add_annotations(frame_out, anomaly_score, processing_time_ms, anomaly_areas=anomaly_areas)

        prof.dump(prefix=f"[f{frame_idx:05d}] ")
        
        return AnomalyDetectionResult(
            processed_frame=frame_out,
            anomaly_score=anomaly_score,
            is_anomaly=anomaly_score >= self.threshold,
            anomaly_areas=anomaly_areas,
            segmentation_map=current_segmentation
        )



    def process_single_frame(self, frame, save_dir: Optional[str] = None,
                            filename: Optional[str] = None,
                            session_json_path: Optional[str] = None) -> AnomalyDetectionResult:
        """
        Simple interface for processing single frames
        Returns: AnomalyDetectionResult object with coordinates
        """
        frame_idx = getattr(self, "_sf_frame_idx", 0)
        last_seg = getattr(self, "_sf_last_segmentation", None)

        t0 = time.perf_counter()
        result = self.process_frame(frame, frame_idx, last_seg)
        processing_ms = (time.perf_counter() - t0) * 1000.0

        # update internal state for next call
        self._sf_frame_idx = frame_idx + 1
        self._sf_last_segmentation = result.segmentation_map
        
        # Clear stored anomaly areas if this is a new sequence (optional)
        if frame_idx == 0:
            self._sf_last_anomaly_areas = None

        # Optional artifacts
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            tag = "ANOMALY" if result.is_anomaly else "NORMAL"
            fname = filename or f"frame_{frame_idx:06d}_{tag}.jpg"
            cv2.imwrite(os.path.join(save_dir, fname), result.processed_frame)

        if session_json_path is not None:
            self._update_session_json(session_json_path, {
                "score": float(result.anomaly_score),
                "threshold": float(self.threshold),
                "is_anomaly": result.is_anomaly,
                "processing_ms": float(processing_ms),
                "anomaly_areas": result.anomaly_areas,
                "timestamp": time.time(),
            })

        return result


    def _update_session_json(self, json_path: str, frame_entry: dict):
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {
                "detection_session": {
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "frames": [],
                    "statistics": {"total": 0, "anomaly": 0, "normal": 0},
                }
            }
        sess = data["detection_session"]
        meta = {k: frame_entry[k] for k in ("score", "threshold", "is_anomaly", "processing_ms", "timestamp", "anomaly_areas")}
        sess["frames"].append(meta)
        sess["statistics"]["total"] += 1
        if meta["is_anomaly"]:
            sess["statistics"]["anomaly"] += 1
        else:
            sess["statistics"]["normal"] += 1
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'parallel_processor'):
            del self.parallel_processor

class MultiROIManager:
    """
    Runs multiple ROI detectors on the same full frame.
    - Inference runs on RECT crops.
    - Visualization is polygon-gated.
    - Score is computed INSIDE polygon.
    - Returns coordinates of all anomaly areas in full frame coordinates
    """
    def __init__(self, model_root: str, base_infer: AnomalyInference):
        self.model_root = model_root
        self.base = base_infer
        self.detectors = []
        from collections import deque
        self._roi_buffers = {}
        
        # Parallel processing settings
        self.parallel_rois = False
        self.roi_workers = None
        
        self.min_area = 50
        self.enable_ar = False
        self.min_ar, self.max_ar = 0.125, 8.0
        self.enable_intensity = False
        self.min_std = 0.05

        # Newly added
        self._poly_mask_cache = {}

        for name in sorted(os.listdir(model_root)):
            det_dir = os.path.join(model_root, name)
            if not os.path.isdir(det_dir): 
                continue
            if not os.path.isfile(os.path.join(det_dir, "roi_meta.json")):
                continue
                
            child = AnomalyInference(
                model_path=det_dir,
                threshold=None,
                imagesize=None,
                skip_frames=self.base.skip_frames,
                mask_alpha=self.base.mask_alpha,
                proximity_on_gpu=self.base.proximity_on_gpu,
                tile_rows=self.base.tile_rows, 
                tile_cols=self.base.tile_cols, 
                tile_overlap=self.base.tile_overlap,
                parallel_tiles=self.base.parallel_tiles,
                num_workers=self.base.num_workers,
                resource_sample_interval=self.base.resource_sample_interval,
                output_path=self.base.output_path,
            )
            
            # Load the model once
            child_model = child.load_model()
            self.detectors.append({"name": name, "dir": det_dir, "infer": child})
            self._roi_buffers[name] = deque(maxlen=5)

        logger.info(f"[Multi-ROI] Found {len(self.detectors)} detectors under: {model_root}")

    def _process_single_roi(self, det, full_frame_bgr):
        """Process a single ROI detector"""
        rect_meta, poly_meta, _ = _read_roi_meta(det["dir"])
        x = int(rect_meta.get("x", 0))
        y = int(rect_meta.get("y", 0))
        w = int(rect_meta.get("w", 0))
        h = int(rect_meta.get("h", 0))
        poly_scaled = [(int(px), int(py)) for (px, py) in poly_meta] if poly_meta else []
        
        if w <= 2 or h <= 2:
            return None
        
        crop_bgr = full_frame_bgr[y:y+h, x:x+w].copy()
        
        child: AnomalyInference = det["infer"]
        model = child.model if child.model is not None else child.load_model()
        
        t0 = time.perf_counter()
        result = child.process_frame(
            crop_bgr,
            frame_idx=getattr(child, "_sf_frame_idx", 0),
            last_segmentation=getattr(child, "_sf_last_segmentation", None)
        )
        setattr(child, "_sf_frame_idx", getattr(child, "_sf_frame_idx", 0) + 1)
        setattr(child, "_sf_last_segmentation", result.segmentation_map)
        processing_ms = (time.perf_counter() - t0) * 1000.0
        
        # Convert anomaly areas from ROI coordinates to full frame coordinates
        full_frame_areas = []
        for area in result.anomaly_areas:
            roi_x, roi_y, roi_w, roi_h = area['bbox']
            full_frame_areas.append({
                'bbox': [x + roi_x, y + roi_y, roi_w, roi_h],  # Convert to full frame coords
                'area': area['area'],
                'confidence': area['confidence'],
                'centroid': [x + area['centroid'][0], y + area['centroid'][1]],
                'roi_name': det["name"],
                'aspect_ratio': area.get('aspect_ratio', roi_w / max(roi_h, 1))
            })
        
        return {
            'det': det,
            'result': result,
            'processing_ms': processing_ms,
            'rect': (x, y, w, h),
            'poly_scaled': poly_scaled,
            'child': child,
            'anomaly_areas': full_frame_areas  # Now in full frame coordinates
        }
    
    def _process_rois_parallel(self, full_frame_bgr):
        """Process all ROIs in parallel using ThreadPoolExecutor"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = getattr(self, 'roi_workers', None) or len(self.detectors)
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._process_single_roi, det, full_frame_bgr): det 
                for det in self.detectors
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        return results
    
    def _process_rois_sequential(self, full_frame_bgr):
        """Process all ROIs sequentially"""
        results = []
        for det in self.detectors:
            result = self._process_single_roi(det, full_frame_bgr)
            if result is not None:
                results.append(result)
        return results


    def process_frame_all(self, full_frame_bgr) -> AnomalyDetectionResult:
        """
        Process frame with all ROIs
        Returns: AnomalyDetectionResult with all anomaly areas in full frame coordinates
        """
        # ROI color for polygon overlay (kept for ROI visualization)
        ROI_PINK_BGR = (193, 182, 255)
        roi_poly_alpha = float(getattr(self, "roi_alpha", 0.15))

        t_start_all = time.perf_counter()

        H, W = full_frame_bgr.shape[:2]
        canvas = full_frame_bgr.copy()
        overall_max = 0.0
        any_anom = False
        per_roi = []
        roi_lines = []
        
        # NEW: Collect all anomaly areas from all ROIs
        all_anomaly_areas = []

        # Process ROIs in parallel or sequential
        if getattr(self, 'parallel_rois', False):
            roi_results = self._process_rois_parallel(full_frame_bgr)
        else:
            roi_results = self._process_rois_sequential(full_frame_bgr)
        
        # Merge results from all ROIs
        for result in roi_results:
            det = result['det']
            roi_result = result['result']
            processing_ms = result['processing_ms']
            x, y, w, h = result['rect']
            poly_scaled = result['poly_scaled']
            child = result['child']
            roi_anomaly_areas = result['anomaly_areas']

            # Add ROI's anomaly areas to the complete list
            all_anomaly_areas.extend(roi_anomaly_areas)

            # Draw ROI polygon (optional - if you want to keep ROI visualization)
            if poly_scaled:
                pts_abs = np.array(poly_scaled, dtype=np.int32)
                poly_overlay = canvas.copy()
                cv2.fillPoly(poly_overlay, [pts_abs], ROI_PINK_BGR)
                canvas = cv2.addWeighted(poly_overlay, roi_poly_alpha, canvas, 1.0 - roi_poly_alpha, 0)
                cv2.polylines(canvas, [pts_abs], isClosed=True, color=ROI_PINK_BGR, thickness=2, lineType=cv2.LINE_AA)

            poly_mask = _polygon_mask_for_rect((x, y, w, h), poly_scaled, H, W) # new commented out

            # newly added
            # cache_key = det["name"]
            # if cache_key not in self._poly_mask_cache:
            #     self._poly_mask_cache[cache_key] = _polygon_mask_for_rect(
            #         (x, y, w, h), poly_scaled, H, W
            #     )
            
            # poly_mask = self._poly_mask_cache[cache_key]

            # temporal smoothing per ROI
            buf = self._roi_buffers[det["name"]]
            if roi_result.segmentation_map is not None:
                buf.append(roi_result.segmentation_map)
            smoothed_map = (np.mean(buf, axis=0) if len(buf) > 0 else roi_result.segmentation_map)

            # score inside polygon
            masked_score = _masked_max_inside_polygon(smoothed_map, poly_mask)

            th = float(child.threshold)
            overall_max = max(overall_max, masked_score)

            # threshold -> binary (rect local), gate by polygon
            seg_bin = (smoothed_map >= th).astype(np.uint8)
            if seg_bin.shape[:2] != (h, w):
                seg_bin = cv2.resize(seg_bin, (w, h), interpolation=cv2.INTER_NEAREST)
            viz_mask = (seg_bin > 0) & (poly_mask.astype(bool))

            survivors = []
            if np.any(viz_mask):
                mask_u8 = (viz_mask.astype(np.uint8) * 255)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

                for i in range(1, num_labels):
                    rx = int(stats[i, cv2.CC_STAT_LEFT])
                    ry = int(stats[i, cv2.CC_STAT_TOP])
                    rw = int(stats[i, cv2.CC_STAT_WIDTH])
                    rh = int(stats[i, cv2.CC_STAT_HEIGHT])
                    area = int(stats[i, cv2.CC_STAT_AREA])

                    if area < self.min_area:
                        continue

                    if self.enable_ar:
                        ar = rw / max(rh, 1)
                        if not (self.min_ar <= ar <= self.max_ar):
                            continue

                    if self.enable_intensity:
                        patch = smoothed_map[ry:ry+rh, rx:rx+rw]
                        if patch.size == 0 or float(np.std(patch)) < self.min_std:
                            continue

                    survivors.append((rx, ry, rw, rh))

                    # REMOVED: The pixel overlay code (blue color overlay)
                    # We only want bounding boxes

                # Draw bounding boxes for survivors (in full frame coordinates)
                for (rx, ry, rw, rh) in survivors:
                    # Convert ROI coordinates to full frame coordinates
                    fx, fy = x + rx, y + ry
                    cv2.rectangle(
                        canvas,
                        (fx, fy),
                        (fx + rw, fy + rh),
                        (0, 0, 255),  # Red color for anomaly bounding boxes
                        2  # Thickness
                    )
                    
                    # Optional: Add label with confidence
                    # Find matching anomaly area for this survivor to get confidence
                    for area in roi_anomaly_areas:
                        if (area['bbox'][0] == fx and area['bbox'][1] == fy and
                            area['bbox'][2] == rw and area['bbox'][3] == rh):
                            confidence = area['confidence']
                            # Draw confidence label
                            label = f"{confidence:.2f}"
                            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(canvas, 
                                        (fx, fy - label_size[1] - 5),
                                        (fx + label_size[0], fy),
                                        (0, 0, 255), -1)
                            cv2.putText(canvas, label, 
                                    (fx, fy - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            break

            roi_has_valid = (len(survivors) > 0)
            is_anom = bool(roi_has_valid and (masked_score >= th))
            any_anom = any_anom or is_anom

            name_raw = det["name"]; low = name_raw.lower(); base = name_raw
            for pref in ("roi_", "anomaly_", "anomalyarea_", "area_"):
                if low.startswith(pref):
                    base = name_raw[len(pref):]
                    break
            display_name = f"ROI_{base}"
            roi_lines.append(f"{display_name}  Score/Threshold  : {masked_score:.2f} / {th:.2f}")

            per_roi.append({
                "roi": det["name"],
                "rect": {"x": x, "y": y, "w": w, "h": h},
                "score": float(masked_score),
                "threshold": th,
                "is_anomaly": bool(is_anom),
                "processing_ms": float(processing_ms),
                "components": len(survivors),
                "anomaly_areas": roi_anomaly_areas  # Include areas for this ROI
            })

        total_ms = (time.perf_counter() - t_start_all) * 1000.0
        fps = (1000.0 / total_ms) if total_ms > 0 else 0.0

        def put_text(img, text, org, scale, color, thickness=2):
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

        margin = 14

        # Top-left: status (big)
        status = "ANOMALY" if any_anom else "NORMAL"
        status_color = (0, 0, 255) if any_anom else (0, 200, 0)
        put_text(canvas, status, (margin, 32), 1.2, status_color, 2)

        # Top-center: FPS
        fps_text = f"FPS: {fps:.2f}"
        size_fps, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        put_text(canvas, fps_text, ((W - size_fps[0]) // 2, 32), 1.0, (255, 255, 255), 2)

        # Top-right: processing ms
        proc_text = f"Processing: {total_ms:.1f}ms"
        size_proc, _ = cv2.getTextSize(proc_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        put_text(canvas, proc_text, (W - size_proc[0] - margin, 32), 1.0, (255, 255, 255), 2)

        # Bottom-left: compact ROI lines
        if roi_lines:
            y = H - margin - 2
            scale_small, thick_small = 0.38, 1
            line_h = int(cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale_small, thick_small)[0][1] * 1.6)
            for s in reversed(roi_lines):
                put_text(canvas, s, (margin, y), scale_small, (0, 0, 0), thick_small)
                y -= line_h

        # NEW: Return AnomalyDetectionResult with all anomaly areas
        return AnomalyDetectionResult(
            processed_frame=canvas,
            anomaly_score=overall_max,
            is_anomaly=any_anom,
            anomaly_areas=all_anomaly_areas,
            segmentation_map=None  # Not available for multi-ROI case
        )

    def cleanup(self):
        """Clean up all child detectors"""
        for det in self.detectors:
            if hasattr(det['infer'], 'cleanup'):
                det['infer'].cleanup()

# NEW: Unified interface function
def create_anomaly_detector(model_path: str, **kwargs) -> Any:
    """
    Unified function to create appropriate anomaly detector
    Returns: AnomalyInference or MultiROIManager based on model type
    """
    if _is_multi_model_root(model_path):
        base_infer = AnomalyInference(model_path, **kwargs)
        base_infer.load_model()  # Ensure model is loaded
        return MultiROIManager(model_path, base_infer)
    else:
        detector = AnomalyInference(model_path, **kwargs)
        detector.load_model()  # Ensure model is loaded
        return detector

# NEW: Simple usage example
def process_frame_with_coordinates(detector, frame_bgr) -> AnomalyDetectionResult:
    """
    Process a frame and get anomaly coordinates
    Works with both single ROI and multi-ROI detectors
    """
    if isinstance(detector, MultiROIManager):
        return detector.process_frame_all(frame_bgr)
    else:
        return detector.process_single_frame(frame_bgr)

