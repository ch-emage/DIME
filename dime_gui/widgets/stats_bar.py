# =========================================================
# widgets/stats_bar.py — Horizontal performance stats strip
# =========================================================

import time
from collections import deque

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel

from widgets.theme import (
    COLOR_SURFACE, COLOR_BORDER, COLOR_TEXT,
    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_WARN, COLOR_DANGER,
    FS_LABEL,
)


def _chip_css(color: str) -> str:
    return (
        f"QLabel {{ background:{COLOR_SURFACE}; color:{color}; "
        f"border:1px solid {COLOR_BORDER}; border-radius:4px; "
        f"padding:5px 10px; font-weight:600; font-size:{FS_LABEL}px; }}"
    )


class StatsBar(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.lbl_latency     = QLabel("Latency: -- ms")
        self.lbl_instant_fps = QLabel("Instant: -- FPS")
        self.lbl_avg_fps     = QLabel("Avg: -- FPS")
        self.lbl_frames      = QLabel("Frames: 0")
        self.lbl_status      = QLabel("⏹ Idle")
        self.lbl_score       = QLabel("Score: --")
        self.lbl_threshold   = QLabel("Threshold: --")

        self.lbl_latency.setStyleSheet(_chip_css(COLOR_WARN))
        self.lbl_instant_fps.setStyleSheet(_chip_css(COLOR_PRIMARY))
        self.lbl_avg_fps.setStyleSheet(_chip_css(COLOR_SUCCESS))
        self.lbl_frames.setStyleSheet(_chip_css(COLOR_TEXT))
        self.lbl_status.setStyleSheet(_chip_css(COLOR_DANGER))
        self.lbl_score.setStyleSheet(_chip_css(COLOR_TEXT))
        self.lbl_threshold.setStyleSheet(_chip_css(COLOR_WARN))

        for lbl in (self.lbl_latency, self.lbl_instant_fps, self.lbl_avg_fps,
                    self.lbl_frames, self.lbl_status, self.lbl_score, self.lbl_threshold):
            layout.addWidget(lbl)
        layout.addStretch()
        self.setLayout(layout)

        self._frame_times  = deque(maxlen=30)
        self._total_frames = 0
        self._threshold    = None

    # ── threshold ────────────────────────────────────────

    def set_threshold(self, thresholds: list[dict]):
        effective = None
        for e in thresholds:
            if e.get("label") == "Model (root)" and e.get("effective") is not None:
                effective = e["effective"]
                break
        if effective is None:
            for e in thresholds:
                if e.get("effective") is not None:
                    effective = e["effective"]
                    break
        self._threshold = effective
        self.lbl_threshold.setText(
            f"Threshold: {effective:.4f}" if effective else "Threshold: N/A"
        )

    def set_threshold_value(self, value: float):
        self._threshold = value
        self.lbl_threshold.setText(f"Threshold: {value:.4f}")

    # ── per-frame stats ──────────────────────────────────

    def update_stats(self, latency_ms, score: float = None):
        self._total_frames += 1
        self._frame_times.append(time.time())
        instant_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        avg_fps = 0.0
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            avg_fps = (len(self._frame_times) - 1) / span if span > 0 else 0.0
        self.lbl_latency.setText(f"Latency: {latency_ms:.1f} ms")
        self.lbl_instant_fps.setText(f"Instant: {instant_fps:.1f} FPS")
        self.lbl_avg_fps.setText(f"Avg: {avg_fps:.1f} FPS")
        self.lbl_frames.setText(f"Frames: {self._total_frames}")
        if score is not None:
            is_anom = (self._threshold is not None and score >= self._threshold)
            color = COLOR_DANGER if is_anom else COLOR_SUCCESS
            self.lbl_score.setText(f"Score: {score:.4f}")
            self.lbl_score.setStyleSheet(_chip_css(color))

    def reset(self):
        self._frame_times.clear()
        self._total_frames = 0
        self.lbl_latency.setText("Latency: -- ms")
        self.lbl_instant_fps.setText("Instant: -- FPS")
        self.lbl_avg_fps.setText("Avg: -- FPS")
        self.lbl_frames.setText("Frames: 0")
        self.lbl_score.setText("Score: --")
        self.lbl_score.setStyleSheet(_chip_css(COLOR_TEXT))

    # ── status ──────────────────────────────────────────

    def set_status(self, state: str):
        label, color = {
            "running": ("▶ Running", COLOR_SUCCESS),
            "paused":  ("⏸ Paused",  COLOR_WARN),
            "stopped": ("⏹ Stopped", COLOR_DANGER),
            "idle":    ("⏹ Idle",    COLOR_DANGER),
        }[state]
        self.lbl_status.setText(label)
        self.lbl_status.setStyleSheet(_chip_css(color))
