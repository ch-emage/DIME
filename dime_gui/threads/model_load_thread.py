# =========================================================
# threads/model_load_thread.py — Background model loading
# =========================================================

import os
import time

import dime_v2

from PySide6.QtCore import QThread, Signal


class ModelLoadThread(QThread):
    """Loads a DIME model in a background thread so the UI stays responsive."""
    loaded = Signal(object, float)   # (detector, elapsed_ms)
    failed = Signal(str)             # error message

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            t = time.perf_counter()
            detector = dime_v2.create_detector(
                model_path=self.model_path,
                skip_frames=3, threshold=80, proximity_on_gpu=False,
                parallel_tiles=True, num_workers=1, tile_rows=1, tile_cols=1,
                save_comparison_results=True, enable_visualization=False,
            )
            elapsed_ms = (time.perf_counter() - t) * 1000

            # Best-effort: override thresholds for multi-ROI tiles if that structure exists.
            # A model that doesn't use multi_mgr (single-ROI, refactored API) should still load cleanly.
            try:
                inner = getattr(detector, "detector", None)
                engine = getattr(inner, "engine", None) if inner else None
                multi_mgr = getattr(engine, "multi_mgr", None) if engine else None
                detectors = getattr(multi_mgr, "detectors", None) if multi_mgr else None
                if detectors:
                    for det in detectors:
                        if isinstance(det, dict) and "infer" in det:
                            det["infer"].threshold = 170
            except Exception as thresh_err:
                print(f"⚠  Could not override tile thresholds: {thresh_err}")

            self.loaded.emit(detector, elapsed_ms)
        except Exception as e:
            self.failed.emit(str(e))
