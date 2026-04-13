# =========================================================
# tabs/media_tab.py — Tab 2: Image & Video inference
# =========================================================

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QMessageBox
)
from PySide6.QtCore import Slot

from config import VIDEO_CONFIRM_FRAMES, VIDEO_HOLD_FRAMES
from inference import AnomalyDebouncer
from threads import ImageInferenceThread, VideoInferenceThread
from widgets import FrameView, StatsBar, make_anomaly_alert_label, set_anomaly_label


class MediaTab(QWidget):
    def __init__(self):
        super().__init__()
        self.detector = self.roi_coords = self.image_path = None
        self.image_worker = self.video_worker = None
        self._paused_frame_pos   = 0
        self._paused_video_source = None
        self._user_paused        = False
        self._debouncer = AnomalyDebouncer(VIDEO_CONFIRM_FRAMES, VIDEO_HOLD_FRAMES)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        ctrl = QHBoxLayout()

        self.btn_load_img   = QPushButton("🖼  Load Image")
        self.btn_run_img    = QPushButton("▶  Run on Image")
        self.btn_load_video = QPushButton("🎬  Load Video")
        self.btn_pause      = QPushButton("⏸  Pause")
        self.btn_continue   = QPushButton("▶  Continue")

        for btn in (self.btn_load_img, self.btn_run_img, self.btn_load_video,
                    self.btn_pause, self.btn_continue):
            btn.setFixedHeight(38)
            ctrl.addWidget(btn)
        ctrl.addStretch()

        self.btn_run_img.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_continue.setEnabled(False)

        self.btn_load_img.clicked.connect(self._load_image)
        self.btn_run_img.clicked.connect(self._run_image)
        self.btn_load_video.clicked.connect(self._load_video)
        self.btn_pause.clicked.connect(self._pause)
        self.btn_continue.clicked.connect(self._continue)

        self.alert_label = make_anomaly_alert_label()
        self.stats = StatsBar()
        self.view  = FrameView("Load an image or video to begin")

        layout.addLayout(ctrl)
        layout.addWidget(self.alert_label)
        layout.addWidget(self.stats)
        layout.addWidget(self.view, 1)
        self.setLayout(layout)

    # ── model connection ─────────────────────────────────

    def set_detector(self, detector, roi_coords, thresholds=None):
        self.detector   = detector
        self.roi_coords = roi_coords
        self.btn_load_img.setEnabled(True)
        self.btn_load_video.setEnabled(True)
        if thresholds:
            self.stats.set_threshold(thresholds)

    # ── image ─────────────────────────────────────────────

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        self.image_path = path
        self._stop_video()
        import cv2
        self.view.show_frame(cv2.imread(path))
        self.stats.reset()
        self.stats.set_status("idle")
        set_anomaly_label(self.alert_label, False)
        self.btn_run_img.setEnabled(bool(self.detector))

    def _run_image(self):
        if not self.detector or not self.image_path:
            return
        self.btn_run_img.setEnabled(False)
        set_anomaly_label(self.alert_label, False)
        self.stats.reset()
        self.image_worker = ImageInferenceThread(self.detector, self.roi_coords, self.image_path)
        self.image_worker.finished.connect(self._on_image_done)
        self.image_worker.start()

    @Slot(object, float, float, bool)
    def _on_image_done(self, frame, latency, score, has_anomaly):
        self.stats.update_stats(latency, score)
        self.stats.set_status("idle")
        self.view.show_frame(frame)
        set_anomaly_label(self.alert_label, has_anomaly)
        self.btn_run_img.setEnabled(True)

    # ── video ─────────────────────────────────────────────

    def _load_video(self):
        if not self.detector:
            QMessageBox.warning(self, "Error", "Load a model first (Tab 1)")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.avi *.mov *.mkv *.MOV)"
        )
        if not path:
            return
        self._stop_video()
        self.stats.reset()
        set_anomaly_label(self.alert_label, False)
        self._debouncer = AnomalyDebouncer(VIDEO_CONFIRM_FRAMES, VIDEO_HOLD_FRAMES)
        self._user_paused = False
        self._paused_video_source = path
        self._start_video(path, 0)

    def _start_video(self, path, start_frame=0):
        self._stop_video()
        self.stats.set_status("running")
        self.btn_pause.setEnabled(True)
        self.btn_continue.setEnabled(False)
        self.video_worker = VideoInferenceThread(
            self.detector, self.roi_coords, path, is_rtsp=False, start_frame=start_frame
        )
        self.video_worker.frameProcessed.connect(self._on_frame)
        self.video_worker.finished.connect(self._on_video_finished)
        self.video_worker.start()

    def _pause(self):
        if self.video_worker:
            self._paused_frame_pos = self.video_worker.last_frame_pos
            self._user_paused = True
            self.video_worker.stop()

    def _continue(self):
        if not self._paused_video_source or not self.detector:
            return
        self._user_paused = False
        self._start_video(self._paused_video_source, self._paused_frame_pos)

    def _stop_video(self):
        if self.video_worker:
            self.video_worker.stop()
            self.video_worker.wait()
            self.video_worker = None

    @Slot(object, float, float, bool)
    def _on_frame(self, frame, latency, score, has_anomaly):
        display_anomaly = self._debouncer.update(has_anomaly)
        self.stats.update_stats(latency, score)
        self.view.show_frame(frame)
        set_anomaly_label(self.alert_label, display_anomaly)

    @Slot()
    def _on_video_finished(self):
        self.video_worker = None
        if self._user_paused and self._paused_video_source:
            self.stats.set_status("paused")
            self.btn_pause.setEnabled(False)
            self.btn_continue.setEnabled(True)
        else:
            self.stats.set_status("stopped")
            self.btn_pause.setEnabled(False)
            self.btn_continue.setEnabled(False)
            set_anomaly_label(self.alert_label, False)
            self._debouncer.reset()
            self._paused_video_source = None

    def cleanup(self):
        self._stop_video()
