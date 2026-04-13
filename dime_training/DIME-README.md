# DIME – Dynamic Inspection Metrology & Evaluation

**Developed by:** Emagesoft  
**Version:** 1.0  

DIME (Dynamic Inspection Metrology & Evaluation) is an industrial anomaly‑detection system for real-time video inspection. It supports tiled high‑resolution processing, distributed (DDP) training, motion analysis, and real‑time inference.

This README focuses on **how to use** the codebase: setup, training, inference, and model updates.

---

## 1. Environment Setup

### 1.1 System Requirements

- **OS:** Windows 10/11 or Ubuntu 20.04+
- **Python:** 3.8–3.10
- **GPU:** NVIDIA GPU with CUDA (≥ 8 GB VRAM recommended)
- **RAM:** ≥ 16 GB
- **Disk:** ≥ 50 GB free (dataset + models)

### 1.2 Python Dependencies

Install dependencies:

```bash
pip install torch torchvision opencv-python numpy pillow
pip install faiss-gpu ultralytics timm scikit-learn tqdm
```

Ensure you have a **CUDA‑enabled** PyTorch build matching your driver/CUDA version.

---

## 2. Data Preparation

DIME is designed for **video‑based anomaly detection** with a "normal vs. defect" structure.

Suggested folder layout:

```text
data/
└── [subdatases]/
    ├── train/
    │   └── good/
    │       ├── video1.mp4
    │       ├── video2.mp4
    │       └── ...

```

- **Training** uses **only normal/good** videos to build the feature memory bank.
- **Test/defect** data is for evaluation and validation.

You can organize multiple subdatasets under `data/` and select them by name.

---

## 3. Training

### 3.1 Core Training Scripts

The main training logic:

- **`train_tuple.py`** – Core training pipeline
- **`ddp_4rank_stable.py`** – DDP launcher wrapper for 4-rank distributed training
- **`training_accelerator.py`** – GPU & DataLoader optimization utilities

### 3.2 Recommended Training Command

Use `torchrun` to launch 4‑rank DDP training:

```bash
torchrun --nproc_per_node=1 ddp_4rank_stable.py
```

**Why `--nproc_per_node=1`?**

- `ddp_4rank_stable.py` internally manages **4 ranks** using `mp.spawn`.
- From the outside, you launch **one process** which spawns 4 worker processes internally.
- The script:
  - Sets optimal CUDA/CPU environment variables
  - Initializes distributed training (4 ranks)
  - Configures 2×2 tiling (4 tiles total, 1 tile per rank)
  - Calls `train_tuple.py` with proper DDP configuration

### 3.3 Training Configuration Overview

`train_tuple.py` contains a default `config` dictionary with key parameters:

**Dataset Configuration:**
- `datapath`: `"data"`
- `subdatasets`: `"cam"`
- `batchsize`: `1` (adjusted automatically for tiling/DDP)
- `frameinterval`: `20` (frame sampling interval from video)

**Anomaly Detection:**
- `backbonenames`: `["wideresnet50"]`
- `layerstoextractfrom`: `["layer2", "layer3"]`
- `pretrainembeddimension`: `512`
- `targetembeddimension`: `512`
- `anomalyscorernumnn`: `2` (k for k-NN)


**Tiling Configuration:**
- `enable`: `True`
- `rows`: `2`
- `cols`: `2`
- `overlap`: `0.10` (10% tile overlap)

**DDP Configuration:**
- `enable`: `True`
- `backend`: `"gloo"` (Windows) or `"nccl"` (Linux)

You can override many parameters via command-line flags (see `build_arg_parser` in `train_tuple.py`).

### 3.4 Single‑GPU Training (Optional)

For debugging or simpler setups, run `train_tuple.py` directly:

```bash
python train_tuple.py \
  --datapath data \
  --subdatasets cam \
  --dataset-name anomaly \
  --batchsize 4 \
  --tiling-rows 2 \
  --tiling-cols 2 \
  --motion-enable \
  --speed-analysis \
  --motion-weight 0.3
```

For production, **use the DDP launcher** with `torchrun`.

### 3.5 Training Outputs

After training completes, you'll get:

```text
MODEL/
├── models/
│   └── cam/
│       ├── rank0/
│       │   ├── pos0/
│       │   │   ├── dime_params.pkl
│       │   │   ├── nnscorer_search_index.faiss
│       │   │   └── anomaly_rater_features.pkl
│       │   └── pos1/
│       └── rank1/
│           └── ...
├── training_config.json
├── training_metrics.json
└── models_index.json
```

- **`posX`** = tile position for that rank
- Each `posX` folder is a **tile model** (memory bank + FAISS index + parameters)

---

## 4. Inference

Inference is handled by **`infer_paralel.py`**, which supports:

- Per‑tile model loading (from 4-rank training)
- Tiled stitching for full‑frame anomaly maps
- Real‑time frame / video / stream processing
- Optional parallel tile processing

### 4.1 Basic Single‑Frame Inference

Test a trained tile model:

```python
import cv2
from infer_paralel import AnomalyInference

# Path to a trained tile model
model_path = "MODEL/models/cam_anomaly/models"

infer = AnomalyInference(
    model_path=model_path,
    threshold=None,      # Auto-read dynamic threshold
    imagesize=None,      # Auto-read from training params
    tile_rows=2,
    tile_cols=2,
    tile_overlap=0.10,
    parallel_tiles=True, # Multi-process tile inference
    num_workers=4,
)

infer.load_model()

frame = cv2.imread("test_frame.jpg")
result = infer.process_single_frame(frame)

print(f"Score: {result.anomaly_score:.3f}, Anomaly: {result.is_anomaly}")
for area in result.anomaly_areas:
    print(f"  Region: {area['bbox']}, Confidence: {area['confidence']:.3f}")

cv2.imwrite("annotated_frame.jpg", result.processed_frame)
```

### 4.2 Video Inference

Process an entire video:

```python
from infer_paralel import AnomalyInference

infer = AnomalyInference(
    model_path="MODEL/models",
    threshold=None,
    tile_rows=2,
    tile_cols=2,
    parallel_tiles=True,
    num_workers=4,
)
infer.load_model()

infer.process_video(
    input_video="input.mp4",
    output_video="output_annotated.mp4",
    save_frames=True,
    output_dir="anomaly_frames/",
    session_json_path="detection_log.json",
)
```

**Outputs:**
- `output_annotated.mp4`: Video with anomaly overlays and status text
- `anomaly_frames/`: Optional per‑frame images
- `detection_log.json`: Per‑frame scores and metadata

### 4.3 Real‑Time Stream (Webcam / RTSP)

```python
import cv2
from infer_paralel import AnomalyInference

infer = AnomalyInference(
    model_path="MODEL/models/cam/rank0/pos0",
    threshold=None,
    parallel_tiles=True,
    num_workers=4,
)
infer.load_model()

cap = cv2.VideoCapture(0)  # Webcam or RTSP URL

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = infer.process_single_frame(frame)
    cv2.imshow("DIME – Inspection", result.processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 5. Model Update (Incremental Learning)

`update_model_tiled.py` allows adding **new normal frames** to an existing model without full retraining.

### 5.1 Update Command

```bash
python update_model_tiled.py \
  --modelroot MODEL \
  --newframes new_data/frames \
  --datasetname cam
```

**Parameters:**
- `--modelroot`: Path containing `training_config.json` and `models/`
- `--newframes`: Folder with new frame images (PNG/JPG)
- `--datasetname`: Dataset folder name (e.g., `cam`)

### 5.2 Update Process

For each `rankX/posY`:

1. Check model health (required files, FAISS index)
2. Load existing tile model and FAISS index
3. Extract features from new frames (same tiling config)
4. Apply coreset selection on new features
5. Append new features to memory bank and FAISS index
6. Save updated model back to `posY` folder
7. Rebuild `models_index.json`

**Use this** to adapt DIME to gradually changing "normal" behavior without restarting from scratch.

---

## 6. Typical End‑to‑End Workflow

1. **Prepare data**
   - Organize videos into `data/cam/train/good` and `data/cam/test/good|defect`

2. **Start training**
   ```bash
   torchrun --nproc_per_node=1 ddp_4rank_stable.py
   ```
   - Wait for completion (models appear under `MODEL/`)

3. **Inspect outputs**
   - Check `MODEL/training_metrics.json`
   - Verify `MODEL/models/` structure

4. **Run inference**
   - Use `infer_paralel.py` API to process frames/videos/streams

5. **Integrate into production**
   - Wrap `AnomalyInference` into your application/service

6. **Update model periodically**
   - Save new "normal" frames
   - Run `update_model_tiled.py`

---

## 7. Configuration Parameters

### 7.1 Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backbone_names` | `wideresnet50` | Feature extraction network |
| `layers_to_extract_from` | `layer2, layer3` | Network layers for features |
| `target_embed_dimension` | `512` | Final feature dimension |
| `anomaly_scorer_num_nn` | `2` | k for k-NN distance |
| `feature_window` | `2` | Spatial neighborhood size |
| `selector_percentage` | `0.01` | Coreset sampling ratio (1%) |
| `tile_rows` | `2` | Tiling rows |
| `tile_cols` | `2` | Tiling columns |
| `tile_overlap` | `0.10` | Tile overlap fraction |
| 
### 7.2 Directory Structure After Training

```text
MODEL/
├── models/
│   └── cam/                    # Dataset name
│       ├── rank0/              # DDP rank 0
│       │   ├── pos0/           # Tile position 0
│       │   │   ├── dime_params.pkl
│       │   │   ├── nnscorer_search_index.faiss
│       │   │   ├── anomaly_rater_features.pkl
│       │   │   └── dynamic_threshold.pkl
│       │   └── pos1/           # Tile position 1
│       └── rank1/              # DDP rank 1
│           └── ...
├── training_config.json        # Complete training configuration
├── training_metrics.json       # Training statistics and timing
└── models_index.json           # Model routing information
```

---

## 8. Performance Tips

### 8.1 GPU Memory Optimization

If you encounter CUDA Out-Of-Memory errors:

- **Reduce batch size**: Set `--batchsize 1` or `2`
- **Increase tiling**: Use 4×4 tiling instead of 2×2
- **Disable features**: Temporarily disable motion/sequence models
- **Use gradient checkpointing**: Enabled by default in `training_accelerator.py`

### 8.2 Training Speed

For faster training:

- **Use DDP**: 4-rank training provides ~3.4× speedup
- **Enable mixed precision**: Automatically enabled for Ampere+ GPUs
- **Optimize DataLoader**: Increase `num_workers` (8-12 recommended)
- **Use TF32**: Enabled by default on RTX 30xx/40xx/50xx series

### 8.3 Inference Speed

For faster inference:

- **Enable parallel tiles**: Set `parallel_tiles=True`
- **Increase workers**: Set `num_workers=4` or higher
- **Use FP16**: Automatically enabled for CUDA inference
- **Skip frames**: Use `skip_frames` parameter for non-critical applications

---

## 9. Troubleshooting

### Common Issues

**Issue:** `RuntimeError: CUDA out of memory`
- **Solution:** Reduce batch size, increase tiling, or use smaller backbone

**Issue:** `FileNotFoundError: No FAISS index found`
- **Solution:** Ensure training completed successfully, check `MODEL/models/` structure

**Issue:** DDP initialization fails on Windows
- **Solution:** Script uses file-based init by default (Gloo backend)

**Issue:** Slow data loading
- **Solution:** Increase `num_workers`, enable `persistent_workers`

**Issue:** Low detection accuracy
- **Solution:** Check training data quality, increase training samples, adjust threshold

---

## 10. Advanced Features

### 10.1 Multi-ROI Processing

Process multiple regions of interest with separate models:

```python
from infer_paralel import MultiROIManager

manager = MultiROIManager(
    model_root="MODEL/models/",
    base_infer=base_inference_config
)

result = manager.process_frame_all(frame)
```

### 10.2 Motion Analysis Configuration

### 10.3 Custom Backbones

Supported backbone options:

- `wideresnet50` (default, best accuracy)
- `resnet50`, `resnet101` (faster)
- `efficientnet_b0` to `efficientnet_b7` (efficient)
- `mobilenetv3_large` (edge deployment)

Change via `--backbone-names` flag or config dictionary.

---

## 11. System Architecture

### 11.1 Data Flow

**Training:**
```
Videos → Frame Extraction → Tiling → Feature Extraction → 
Coreset Selection → Memory Bank → FAISS Index
```

**Inference:**
```
Frame → Tiling → Feature Extraction → k-NN Search → 
Score Computation → Stitching → Visualization
```

---

## 12. Best Practices

### 12.1 Data Collection

- Collect diverse normal samples (lighting, angles, positions)
- Ensure defect samples represent real production failures
- Use consistent video quality and frame rates
- Minimum 1000-5000 normal frames recommended

### 12.2 Training

- Always use DDP for production (faster, more stable)
- Keep tiling configuration consistent between train/inference
- Monitor training metrics in `training_metrics.json`
- Validate on held-out test set before deployment

### 12.3 Deployment

- Set conservative threshold initially (reduce false positives)
- Use temporal smoothing for video streams
- Log anomaly detections with timestamps and scores
- Periodically update model with new normal samples

---

## 13. License & Support

**Project:** DIME – Dynamic Inspection Metrology & Evaluation  
**Developed by:** Emagesoft  
**Version:** 1.0  

For technical support, custom deployment, or feature requests, contact Emagesoft technical team.

---

## 14. Quick Reference

### Training Commands

```bash
# 4-rank DDP training (recommended)
torchrun --nproc_per_node=1 ddp_4rank_stable.py

# Single GPU training
python train_tuple.py --datapath data --subdatasets cam
```

### Inference Commands

```python
# Single frame
from infer_paralel import AnomalyInference
infer = AnomalyInference(model_path="MODEL/models/")
infer.load_model()
result = infer.process_single_frame(frame)

# Video batch
infer.process_video(input_video="input.mp4", output_video="output.mp4")
```

### Update Command

```bash
python update_model_tiled.py \
  --modelroot MODEL \
  --newframes new_data/frames \
  --datasetname cam
```

---

**End of README**