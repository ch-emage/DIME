# =========================================================
# tabs/live_tab.py — Tab 3: Live Camera (RTSP + OAK)
# =========================================================

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QInputDialog, QMessageBox, QDoubleSpinBox
)
from PySide6.QtCore import Slot, QLocale

from config import LIVE_CONFIRM_FRAMES, LIVE_HOLD_FRAMES, OAK_RESOLUTION
from inference import AnomalyDebouncer
from threads import VideoInferenceThread, OAKInferenceThread
from widgets import (
    FrameView, StatsBar, make_anomaly_alert_label, set_anomaly_label,
    BTN_ACCENT, BTN_NEUTRAL, BTN_PRIMARY,
    COLOR_TEXT_MUTED, COLOR_WARN, FS_LABEL, FS_BODY,
)


class LiveTab(QWidget):
    def __init__(self):
        super().__init__()
        self.detector    = self.roi_coords = None
        self.video_worker = self.oak_worker = None
        self._user_paused  = False
        self._mode         = None
        self._rtsp_source  = None
        self._rtsp_is_rtsp = True
        self._debouncer    = AnomalyDebouncer(LIVE_CONFIRM_FRAMES, LIVE_HOLD_FRAMES)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        self.btn_rtsp     = QPushButton("📡  Connect RTSP")
        self.btn_oak      = QPushButton("📷  OAK Camera")
        self.btn_pause    = QPushButton("⏸  Pause")
        self.btn_continue = QPushButton("▶  Continue")

        self.btn_rtsp.setStyleSheet(BTN_ACCENT)
        self.btn_oak.setStyleSheet(BTN_ACCENT)
        self.btn_pause.setStyleSheet(BTN_NEUTRAL)
        self.btn_continue.setStyleSheet(BTN_PRIMARY)

        for btn in (self.btn_rtsp, self.btn_oak, self.btn_pause, self.btn_continue):
            btn.setFixedHeight(38)
            ctrl.addWidget(btn)

        ctrl.addSpacing(12)
        lbl_thresh = QLabel("Threshold:")
        lbl_thresh.setStyleSheet(f"color:{COLOR_TEXT_MUTED}; font-size:{FS_LABEL}px;")
        ctrl.addWidget(lbl_thresh)
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setLocale(QLocale(QLocale.C))
        self.spin_threshold.setDecimals(2)
        self.spin_threshold.setRange(0.0, 10000.0)
        self.spin_threshold.setSingleStep(1.0)
        self.spin_threshold.setValue(80.0)
        self.spin_threshold.setFixedHeight(38)
        ctrl.addWidget(self.spin_threshold)

        self.btn_apply_thresh = QPushButton("Apply")
        self.btn_apply_thresh.setFixedHeight(38)
        self.btn_apply_thresh.setStyleSheet(BTN_NEUTRAL)
        self.btn_apply_thresh.clicked.connect(self._apply_threshold)
        ctrl.addWidget(self.btn_apply_thresh)

        ctrl.addStretch()

        self.btn_pause.setEnabled(False)
        self.btn_continue.setEnabled(False)

        self.btn_rtsp.clicked.connect(self._connect_rtsp)
        self.btn_oak.clicked.connect(self._connect_oak)
        self.btn_pause.clicked.connect(self._pause)
        self.btn_continue.clicked.connect(self._continue)

        self.lbl_source = QLabel("Source: None")
        self.lbl_source.setStyleSheet(
            f"color:{COLOR_WARN}; font-weight:600; font-size:{FS_BODY}px; background:transparent;"
        )

        self.alert_label = make_anomaly_alert_label()
        self.stats = StatsBar()
        self.view  = FrameView("Connect a camera to begin live detection")

        layout.addLayout(ctrl)
        layout.addWidget(self.lbl_source)
        layout.addWidget(self.alert_label)
        layout.addWidget(self.stats)
        layout.addWidget(self.view, 1)
        self.setLayout(layout)

    # ── model connection ─────────────────────────────────

    def set_detector(self, detector, roi_coords, thresholds=None):
        self.detector   = detector
        self.roi_coords = roi_coords
        if thresholds:
            self.stats.set_threshold(thresholds)

    def _iter_anomaly_infers(self):
        d = self.detector
        if d is None:
            return
        engine = getattr(getattr(d, "detector", None), "engine", None) or getattr(d, "engine", None)
        if engine is None:
            return
        mm = getattr(engine, "multi_mgr", None)
        if mm is not None:
            for det in mm.detectors:
                yield det["infer"]
        base = getattr(engine, "anomaly_infer", None)
        if base is not None:
            yield base

    def _apply_threshold(self):
        if not self._require_model():
            return
        value = float(self.spin_threshold.value())
        count = 0
        for infer in self._iter_anomaly_infers():
            infer.threshold = value
            count += 1
        self.stats.set_threshold_value(value)
        if count == 0:
            print(f"⚠  Threshold set to {value:.4f}, but no detector objects were found to apply it to — check model structure")
        else:
            print(f"🎚  Threshold set to {value:.4f} on {count} detector(s)")

    # ── RTSP / webcam ────────────────────────────────────

    def _connect_rtsp(self):
        if not self._require_model():
            return
        url, ok = QInputDialog.getText(self, "RTSP / Camera", "Enter RTSP URL or camera index (e.g. 0):")
        if not ok or not url.strip():
            return
        url    = url.strip()
        source = int(url) if url.isdigit() else url
        is_rtsp = not url.isdigit()
        self.lbl_source.setText(f"Source: {'Camera' if url.isdigit() else 'RTSP'}  —  {source}")
        self._mode = "rtsp"
        self._rtsp_source  = source
        self._rtsp_is_rtsp = is_rtsp
        self._stop_all()
        self._user_paused = False
        set_anomaly_label(self.alert_label, False)
        self._debouncer = AnomalyDebouncer(LIVE_CONFIRM_FRAMES, LIVE_HOLD_FRAMES)
        self._start_rtsp_worker(source, is_rtsp)

    def _start_rtsp_worker(self, source, is_rtsp):
        self.stats.reset()
        self.stats.set_status("running")
        self._set_controls(running=True)
        self.video_worker = VideoInferenceThread(
            self.detector, self.roi_coords, source, is_rtsp=is_rtsp,
            confirm_frames=LIVE_CONFIRM_FRAMES, hold_frames=LIVE_HOLD_FRAMES,
        )
        self.video_worker.frameProcessed.connect(self._on_frame)
        self.video_worker.finished.connect(self._on_stream_finished)
        self.video_worker.connectionLost.connect(self._on_connection_lost)
        self.video_worker.start()

    # ── OAK camera ───────────────────────────────────────

    def _connect_oak(self):
        if not self._require_model():
            return
        self._mode = "oak"
        self._stop_all()
        self._user_paused = False
        set_anomaly_label(self.alert_label, False)
        self._debouncer = AnomalyDebouncer(LIVE_CONFIRM_FRAMES, LIVE_HOLD_FRAMES)
        self.lbl_source.setText("Source: OAK Camera (DepthAI)")
        self._start_oak_worker()

    def _start_oak_worker(self):
        self.stats.reset()
        self.stats.set_status("running")
        self._set_controls(running=True)
        self.oak_worker = OAKInferenceThread(
            self.detector, self.roi_coords, resolution=OAK_RESOLUTION,
            confirm_frames=LIVE_CONFIRM_FRAMES, hold_frames=LIVE_HOLD_FRAMES,
        )
        self.oak_worker.frameProcessed.connect(self._on_frame)
        self.oak_worker.finished.connect(self._on_stream_finished)
        self.oak_worker.connectionLost.connect(self._on_connection_lost)
        self.oak_worker.start()

    # ── playback controls ────────────────────────────────

    def _pause(self):
        self._user_paused = True
        if self.video_worker:
            self.video_worker.stop()
        if self.oak_worker:
            self.oak_worker.stop()

    def _continue(self):
        if not self._require_model():
            return
        self._user_paused = False
        if self._mode == "oak":
            self._start_oak_worker()
        elif self._mode == "rtsp" and self._rtsp_source is not None:
            self._start_rtsp_worker(self._rtsp_source, self._rtsp_is_rtsp)

    def _stop_all(self):
        for attr in ("video_worker", "oak_worker"):
            w = getattr(self, attr)
            if not w:
                continue
            # Disconnect before stop so a queued finished signal can't fire later
            # against a freshly-started replacement worker.
            for sig_name in ("frameProcessed", "finished", "connectionLost"):
                sig = getattr(w, sig_name, None)
                if sig is None:
                    continue
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
            w.stop()
            w.wait()
            setattr(self, attr, None)

    def _set_controls(self, running=False, paused=False):
        self.btn_rtsp.setEnabled(not running and not paused)
        self.btn_oak.setEnabled(not running and not paused)
        self.btn_pause.setEnabled(running)
        self.btn_continue.setEnabled(paused)

    # ── slots ────────────────────────────────────────────

    @Slot(object, float, float, bool)
    def _on_frame(self, frame, latency, score, has_anomaly):
        display_anomaly = self._debouncer.update(has_anomaly)
        self.stats.update_stats(latency, score)
        self.view.show_frame(frame)
        set_anomaly_label(self.alert_label, display_anomaly)

    @Slot()
    def _on_stream_finished(self):
        sender = self.sender()
        # Ignore if this finished belongs to a worker we already replaced.
        if sender is not None and sender is not self.video_worker and sender is not self.oak_worker:
            return
        for attr in ("video_worker", "oak_worker"):
            if getattr(self, attr) is sender:
                setattr(self, attr, None)
        if self._user_paused:
            self.stats.set_status("paused")
            self._set_controls(running=False, paused=True)
        else:
            self.stats.set_status("stopped")
            set_anomaly_label(self.alert_label, False)
            self._debouncer.reset()
            self._set_controls(running=False, paused=False)

    @Slot(str)
    def _on_connection_lost(self, message):
        self.stats.set_status("stopped")
        set_anomaly_label(self.alert_label, False)
        self._debouncer.reset()
        self._set_controls(running=False, paused=False)
        QMessageBox.warning(self, "Connection Lost", message)

    def _require_model(self) -> bool:
        if not self.detector:
            QMessageBox.warning(self, "Error", "Load a model first (Tab 1 — Model Setup)")
            return False
        return True

    def cleanup(self):
        self._stop_all()
