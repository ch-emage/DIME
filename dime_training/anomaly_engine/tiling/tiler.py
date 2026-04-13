# anomaly_engine/tiling/tiler.py
# Utilities for tile generation and seam-aware stitching.

from typing import List, Tuple, Optional
import numpy as np
import cv2
import os

def compute_tiles(image: np.ndarray, rows: int, cols: int, overlap: float = 0.0):

    """
    Integer-base tiling (bit-stable, PatchCore-parity):
      base = H//rows, W//cols
      overlap_px = round(base * overlap)
      stride = base - overlap_px
      actual = base + overlap_px
    Returns:
      tiles:  List[HWC]
      coords: List[(x0,y0,x1,y1)] row-major
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0,1); got {overlap}")

    H, W = image.shape[:2]
    R, Cn = int(rows), int(cols)
    ov = float(overlap)

    base_h = max(1, H // R)
    base_w = max(1, W // Cn)

    overlap_h = int(round(base_h * ov))
    overlap_w = int(round(base_w * ov))

    stride_y = max(1, base_h - overlap_h)
    stride_x = max(1, base_w - overlap_w)

    actual_h = base_h + overlap_h
    actual_w = base_w + overlap_w

    tiles, coords = [], []
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
            tiles.append(image[y0:y1, x0:x1].copy())
            coords.append((int(x0), int(y0), int(x1), int(y1)))

    assert len(tiles) == R * Cn, f"Expected {R*Cn} tiles, got {len(tiles)}"
    return tiles, coords



def hann_weight(h: int, w: int) -> np.ndarray:
    """
    2D Hann window normalized to [0, 1] used to smoothly blend overlapping tiles.
    """
    wy = np.hanning(max(h, 2))[:, None]
    wx = np.hanning(max(w, 2))[None, :]
    w2d = wy * wx
    maxv = float(w2d.max()) if w2d.size else 1.0
    return (w2d / (maxv + 1e-8)).astype(np.float32)


def blend_into(canvas: np.ndarray, tile: np.ndarray, bbox: Tuple[int, int, int, int], weight: Optional[np.ndarray] = None):
    """
    Alpha-blend a tile into `canvas` at bbox=(x0,y0,x1,y1).
    Handles shape mismatches by resizing tile/weight to ROI.
    """
    x0, y0, x1, y1 = bbox
    roi = canvas[y0:y1, x0:x1]
    th, tw = tile.shape[:2]
    if roi.shape[:2] != (th, tw):
        tile = cv2.resize(tile, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
        if weight is not None:
            weight = cv2.resize(weight, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)

    if weight is None:
        canvas[y0:y1, x0:x1] = tile
        return

    if tile.ndim == 2 and roi.ndim == 3:
        tile = np.dstack([tile] * roi.shape[2])
    if tile.ndim == 3 and roi.ndim == 2:
        roi = np.dstack([roi] * tile.shape[2])
        canvas[y0:y1, x0:x1] = roi  # ensure same dims before blending

    w = weight if weight.ndim == 2 else weight[..., 0]
    if tile.ndim == 3:
        w = w[..., None]
    canvas[y0:y1, x0:x1] = (roi * (1.0 - w) + tile * w).astype(canvas.dtype)