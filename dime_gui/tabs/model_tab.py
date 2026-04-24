# =========================================================
# tabs/model_tab.py — Tab 1: Model Setup
# =========================================================

import os
import glob
import json
import pickle

import torch
import faiss

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QFileDialog
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer

from threads import ModelLoadThread
from widgets import (
    BTN_ACCENT, BTN_NEUTRAL,
    COLOR_SURFACE, COLOR_SURFACE_HI, COLOR_BORDER,
    COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_TEXT_DIM,
    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_WARN, COLOR_DANGER,
    FS_LABEL, FS_BODY, FS_SUBTITLE, FS_TITLE,
)


class ModelTab(QWidget):
    modelLoaded = Signal(object, object, object)  # (detector, roi_coords, thresholds)

    def __init__(self):
        super().__init__()
        self.detector     = None
        self.roi_coords   = None
        self._load_worker = None
        self._model_path  = None
        self._load_token  = 0   # bumped on each load; stale workers are ignored
        self._build_ui()

    # ── threshold helpers ────────────────────────────────

    @staticmethod
    def _read_thresholds(model_path: str) -> list[dict]:
        results = []

        def _try_load(pkl_path: str, label: str):
            try:
                with open(pkl_path, "rb") as f:
                    raw = float(pickle.load(f))
                results.append({"label": label, "raw": raw, "effective": raw * 1.25, "path": pkl_path})
            except Exception as e:
                results.append({"label": label, "raw": None, "effective": None, "error": str(e), "path": pkl_path})

        root_pkl = os.path.join(model_path, "dynamic_threshold.pkl")
        if os.path.isfile(root_pkl):
            _try_load(root_pkl, "Model (root)")
        try:
            for child in sorted(os.listdir(model_path)):
                child_dir = os.path.join(model_path, child)
                if not os.path.isdir(child_dir):
                    continue
                child_pkl = os.path.join(child_dir, "dynamic_threshold.pkl")
                if os.path.isfile(child_pkl):
                    _try_load(child_pkl, child)
        except Exception:
            pass
        if not results:
            for found in sorted(glob.glob(os.path.join(model_path, "**", "dynamic_threshold.pkl"), recursive=True)):
                _try_load(found, os.path.relpath(found, model_path))
        return results

    @staticmethod
    def get_roi_from_json(json_path: str) -> tuple:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        r = data.get("rectangle", {})
        return (r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"])

    # ── UI build ─────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Model Setup")
        title.setStyleSheet(
            f"color:{COLOR_TEXT}; font-size:{FS_TITLE}px; font-weight:600; "
            f"margin-bottom:6px; background:transparent;"
        )

        # Status card
        self.card = QFrame()
        self.card.setStyleSheet(
            f"QFrame {{ background:{COLOR_SURFACE}; "
            f"border:1px solid {COLOR_BORDER}; border-radius:8px; }}"
        )
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(14, 12, 14, 12)
        card_layout.setSpacing(6)
        self.lbl_model_path   = QLabel("No model loaded")
        self.lbl_model_path.setStyleSheet(
            f"color:{COLOR_TEXT_DIM}; font-size:{FS_BODY}px; background:transparent;"
        )
        self.lbl_model_status = QLabel("⏳ Waiting for model…")
        self.lbl_model_status.setStyleSheet(
            f"color:{COLOR_WARN}; font-size:{FS_SUBTITLE}px; "
            f"font-weight:600; background:transparent;"
        )
        card_layout.addWidget(self.lbl_model_path)
        card_layout.addWidget(self.lbl_model_status)
        self.card.setLayout(card_layout)

        # Threshold panel
        thresh_title = QLabel("Detection Threshold")
        thresh_title.setStyleSheet(
            f"color:{COLOR_TEXT_MUTED}; font-size:{FS_BODY}px; "
            f"font-weight:600; background:transparent;"
        )

        self.thresh_panel = QFrame()
        self.thresh_panel.setStyleSheet(
            f"QFrame {{ background:{COLOR_SURFACE}; "
            f"border:1px solid {COLOR_BORDER}; border-radius:8px; }}"
        )
        self.thresh_layout = QVBoxLayout(self.thresh_panel)
        self.thresh_layout.setSpacing(6)
        self.thresh_layout.setContentsMargins(10, 8, 10, 8)
        ph = QLabel("No threshold loaded yet")
        ph.setStyleSheet(
            f"color:{COLOR_TEXT_DIM}; font-size:{FS_LABEL}px; "
            f"font-style:italic; background:transparent;"
        )
        self.thresh_layout.addWidget(ph)

        # Buttons
        self.btn_load = QPushButton("📦  Select Model Directory")
        self.btn_load.setFixedHeight(42)
        self.btn_load.setStyleSheet(BTN_ACCENT)
        self.btn_load.clicked.connect(self._load_model)

        self.btn_reload = QPushButton("🔄  Change Model")
        self.btn_reload.setFixedHeight(42)
        self.btn_reload.setEnabled(False)
        self.btn_reload.setStyleSheet(BTN_NEUTRAL)
        self.btn_reload.clicked.connect(self._change_model)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background:{COLOR_BORDER}; max-height:1px; border:none;")

        # GPU / system info
        gpu_title = QLabel("System Info")
        gpu_title.setStyleSheet(
            f"color:{COLOR_TEXT_MUTED}; font-size:{FS_BODY}px; "
            f"font-weight:600; background:transparent;"
        )

        self.lbl_gpu = QLabel("Checking…")
        self.lbl_gpu.setStyleSheet(
            f"QLabel {{ background:#0b0d10; color:{COLOR_SUCCESS}; "
            f"font-family:'JetBrains Mono','Fira Code','Menlo','Consolas',monospace; "
            f"font-size:12px; padding:10px 12px; border:1px solid {COLOR_BORDER}; "
            f"border-radius:6px; }}"
        )
        self.lbl_gpu.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(self.card)
        layout.addSpacing(4)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_reload)
        layout.addSpacing(8)
        layout.addWidget(thresh_title)
        layout.addWidget(self.thresh_panel)
        layout.addSpacing(4)
        layout.addWidget(sep)
        layout.addWidget(gpu_title)
        layout.addWidget(self.lbl_gpu)
        layout.addStretch()
        self.setLayout(layout)
        self._populate_gpu_info()

    # ── threshold panel ──────────────────────────────────

    def _clear_thresh_panel(self):
        while self.thresh_layout.count():
            item = self.thresh_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _populate_thresh_panel(self, thresholds: list[dict]):
        self._clear_thresh_panel()
        if not thresholds:
            lbl = QLabel("⚠  dynamic_threshold.pkl not found in model folder")
            lbl.setStyleSheet(
                f"color:{COLOR_WARN}; font-size:{FS_LABEL}px; background:transparent;"
            )
            self.thresh_layout.addWidget(lbl)
            return
        mono = "'JetBrains Mono','Fira Code','Menlo','Consolas',monospace"
        for entry in thresholds:
            row = QFrame()
            row.setStyleSheet(
                f"QFrame {{ background:{COLOR_SURFACE_HI}; "
                f"border:1px solid {COLOR_BORDER}; border-radius:5px; }}"
            )
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(8, 4, 8, 4)
            row_layout.setSpacing(10)
            lbl_name = QLabel(entry["label"])
            lbl_name.setStyleSheet(
                f"color:{COLOR_PRIMARY}; font-size:{FS_LABEL}px; "
                f"font-weight:600; background:transparent;"
            )
            lbl_name.setMinimumWidth(130)
            if entry.get("error"):
                lbl_val = QLabel(f"❌  Read error: {entry['error']}")
                lbl_val.setStyleSheet(
                    f"color:{COLOR_DANGER}; font-size:{FS_LABEL}px; background:transparent;"
                )
                row_layout.addWidget(lbl_name)
                row_layout.addWidget(lbl_val)
            else:
                lbl_raw = QLabel(f"Raw: {entry['raw']:.6f}")
                lbl_raw.setStyleSheet(
                    f"QLabel {{ background:{COLOR_BORDER}; color:{COLOR_TEXT}; "
                    f"font-family:{mono}; font-size:{FS_LABEL}px; "
                    f"padding:2px 8px; border-radius:4px; }}"
                )
                lbl_eff = QLabel(f"Effective (×1.25): {entry['effective']:.6f}")
                lbl_eff.setStyleSheet(
                    f"QLabel {{ background:#1c3324; color:{COLOR_SUCCESS}; "
                    f"font-family:{mono}; font-size:{FS_LABEL}px; font-weight:600; "
                    f"padding:2px 8px; border-radius:4px; }}"
                )
                for w in (lbl_name, lbl_raw, lbl_eff):
                    w.setToolTip(entry["path"])
                row_layout.addWidget(lbl_name)
                row_layout.addWidget(lbl_raw)
                row_layout.addWidget(lbl_eff)
            row_layout.addStretch()
            self.thresh_layout.addWidget(row)

    def _populate_gpu_info(self):
        lines = [f"CUDA available : {torch.cuda.is_available()}"]
        if torch.cuda.is_available():
            lines.append(f"GPU            : {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            lines.append(f"GPU memory     : {mem:.2f} GB")
        lines.append(f"FAISS GPUs     : {faiss.get_num_gpus()}")
        self.lbl_gpu.setText("\n".join(lines))

    # ── model loading ────────────────────────────────────

    MODEL_LOAD_TIMEOUT_MS = 120_000  # 2 minutes

    def _load_model(self, checked=False, path=None):
        if not path:
            path = QFileDialog.getExistingDirectory(self, "Select DIME Model Directory")
        if not path:
            return
        self._model_path = path
        self._load_token += 1
        token = self._load_token
        self.lbl_model_status.setText("⏳ Loading…")
        self.lbl_model_status.setStyleSheet(
            f"color:{COLOR_WARN}; font-size:{FS_SUBTITLE}px; "
            f"font-weight:600; background:transparent;"
        )
        self.btn_load.setEnabled(False)
        self.btn_reload.setEnabled(False)
        self._clear_thresh_panel()

        worker = ModelLoadThread(path)
        self._load_worker = worker
        worker.loaded.connect(lambda d, ms, t=token: self._on_model_loaded(t, d, ms))
        worker.failed.connect(lambda err, t=token: self._on_model_failed(err, t))
        worker.start()
        QTimer.singleShot(self.MODEL_LOAD_TIMEOUT_MS, lambda t=token: self._check_load_timeout(t))

    def _check_load_timeout(self, token):
        # Only act if this token is still the active one and the worker hasn't completed.
        if token != self._load_token:
            return
        w = self._load_worker
        if w is not None and w.isRunning():
            print(f"⚠  Model load exceeded {self.MODEL_LOAD_TIMEOUT_MS // 1000}s — reporting timeout")
            # Invalidate the in-flight worker so its eventual loaded/failed signal is ignored.
            self._load_token += 1
            self._load_worker = None
            self._set_failed_ui(
                f"Model loading timed out after {self.MODEL_LOAD_TIMEOUT_MS // 1000} seconds"
            )

    def _on_model_loaded(self, token, detector, elapsed_ms):
        if token != self._load_token:
            # Stale result from a load we already gave up on.
            try:
                detector.cleanup()
            except Exception:
                pass
            return
        self._load_worker = None
        path = self._model_path
        try:
            json_path = None
            for root, dirs, files in os.walk(path):
                if "roi_meta.json" in files:
                    json_path = os.path.join(root, "roi_meta.json")
                    break
            if json_path is None:
                raise FileNotFoundError("roi_meta.json not found")
            self.detector   = detector
            self.roi_coords = self.get_roi_from_json(json_path)
            thresholds = self._read_thresholds(path)
            self._populate_thresh_panel(thresholds)
            for entry in thresholds:
                if entry.get("error"):
                    print(f"⚠  Threshold [{entry['label']}]: read error — {entry['error']}")
                else:
                    print(f"🎯  Threshold [{entry['label']}]:  raw={entry['raw']:.6f}  →  effective={entry['effective']:.6f}")
            self.lbl_model_path.setText(f"📁 {path}")
            self.lbl_model_status.setText(f"✅ Model ready  ({elapsed_ms:.0f} ms)")
            self.lbl_model_status.setStyleSheet(
                f"color:{COLOR_SUCCESS}; font-size:{FS_SUBTITLE}px; "
                f"font-weight:600; background:transparent;"
            )
            self.btn_reload.setEnabled(True)
            self.modelLoaded.emit(self.detector, self.roi_coords, thresholds)
            print(f"✅ Model loaded in {elapsed_ms:.2f} ms from: {path}")
        except Exception as e:
            self._set_failed_ui(str(e))

    def _on_model_failed(self, error: str, token=None):
        if token is not None and token != self._load_token:
            return
        self._load_worker = None
        self._set_failed_ui(error)

    def _set_failed_ui(self, error: str):
        self.lbl_model_status.setText(f"❌ Failed: {error}")
        self.lbl_model_status.setStyleSheet(
            f"color:{COLOR_DANGER}; font-size:{FS_SUBTITLE}px; "
            f"font-weight:600; background:transparent;"
        )
        self.btn_load.setEnabled(True)
        self.btn_reload.setEnabled(self.detector is not None)

    def _change_model(self):
        if self.detector:
            try:
                self.detector.cleanup()
            except Exception:
                pass
            self.detector = None
        self.lbl_model_status.setText("⏳ Waiting for model…")
        self.lbl_model_status.setStyleSheet(
            f"color:{COLOR_WARN}; font-size:{FS_SUBTITLE}px; "
            f"font-weight:600; background:transparent;"
        )
        self.lbl_model_path.setText("No model loaded")
        self._clear_thresh_panel()
        lbl = QLabel("No threshold loaded yet")
        lbl.setStyleSheet(
            f"color:{COLOR_TEXT_DIM}; font-size:{FS_LABEL}px; "
            f"font-style:italic; background:transparent;"
        )
        self.thresh_layout.addWidget(lbl)
        self._load_model()
