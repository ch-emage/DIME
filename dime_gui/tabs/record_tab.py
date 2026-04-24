# =========================================================
# tabs/record_tab.py — Tab: Record Video (no model needed)
#   + SaveVideoDialog helper
# =========================================================

import os
import time
import shutil
import tempfile
from collections import deque

import cv2

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDialog, QDialogButtonBox, QFileDialog, QMessageBox,
    QRadioButton, QLineEdit
)
from PySide6.QtCore import Qt, Slot

from threads import OAKCaptureThread
from widgets import (
    FrameView, BTN_NEUTRAL, BTN_ACCENT, BTN_REC_IDLE, BTN_REC_ACTIVE,
    COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_TEXT_DIM,
    COLOR_SUCCESS, COLOR_WARN, COLOR_DANGER,
    FS_LABEL, FS_BODY,
)


# =========================================================
# Save-after-record dialog
# =========================================================
class SaveVideoDialog(QDialog):
    def __init__(self, rec_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Recording")
        self.setMinimumWidth(480)
        self.chosen_dir = None
        self._rec_path  = rec_path
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        info = QLabel(
            f"Recording finished.<br>"
            f"File: <b>{os.path.basename(self._rec_path)}</b><br><br>"
            f"Where would you like to save it?"
        )
        info.setTextFormat(Qt.RichText)
        info.setStyleSheet(f"color:{COLOR_TEXT}; font-size:{FS_BODY}px; background:transparent;")
        layout.addWidget(info)

        self._rb_existing = QRadioButton("Save into an existing folder")
        self._rb_new      = QRadioButton("Create a new folder and save there")
        self._rb_existing.setChecked(True)
        for rb in (self._rb_existing, self._rb_new):
            layout.addWidget(rb)

        # New-folder name row — hidden until "Create new folder" is selected
        self._name_row = QWidget()
        nr = QHBoxLayout(self._name_row)
        nr.setContentsMargins(20, 0, 0, 0)
        lbl = QLabel("New folder name:")
        lbl.setStyleSheet(
            f"color:{COLOR_TEXT_MUTED}; font-size:{FS_LABEL}px; "
            f"min-width:130px; background:transparent;"
        )
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g.  camera_1_good")
        nr.addWidget(lbl)
        nr.addWidget(self._name_edit)
        self._name_row.setVisible(False)
        layout.addWidget(self._name_row)

        self._rb_existing.toggled.connect(
            lambda checked: self._name_row.setVisible(not checked)
        )

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_accept(self):
        if self._rb_existing.isChecked():
            folder = QFileDialog.getExistingDirectory(
                self, "Select Destination Folder", os.path.dirname(self._rec_path)
            )
            if not folder:
                return
            self.chosen_dir = folder
        else:
            name = self._name_edit.text().strip()
            if not name:
                QMessageBox.warning(self, "Missing name", "Please enter a folder name.")
                return
            parent = QFileDialog.getExistingDirectory(
                self, "Select Parent Folder", os.path.dirname(self._rec_path)
            )
            if not parent:
                return
            target = os.path.join(parent, name)
            try:
                os.makedirs(target, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not create folder:\n{e}")
                return
            self.chosen_dir = target
        self.accept()


# =========================================================
# Record Video sub-tab
# =========================================================
class RecordSubTab(QWidget):
    def __init__(self):
        super().__init__()
        self._capture: OAKCaptureThread | None = None
        self._recording   = False
        self._writer: cv2.VideoWriter | None = None
        self._rec_path    = ""
        self._frame_count = 0
        self._start_time  = 0.0
        self._fps_times: deque[float] = deque(maxlen=30)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        banner = QLabel(
            "📋  Connect the OAK camera and press  ⏺ Start Recording  to capture training video.  "
            "No model is required.  After stopping, you can save the video to a new or existing folder."
        )
        banner.setWordWrap(True)
        banner.setStyleSheet(
            f"QLabel {{ background:#1a2a1a; color:#a3d158; "
            f"font-size:{FS_LABEL}px; padding:10px 14px; "
            f"border:1px solid #2a3b2a; border-radius:6px; }}"
        )
        layout.addWidget(banner)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)
        self.btn_connect    = QPushButton("📷  Connect OAK")
        self.btn_disconnect = QPushButton("⏹  Disconnect")
        self.btn_record     = QPushButton("⏺  Start Recording")
        for btn in (self.btn_connect, self.btn_disconnect, self.btn_record):
            btn.setFixedHeight(38)
        self.btn_connect.setStyleSheet(BTN_ACCENT)
        self.btn_disconnect.setStyleSheet(BTN_NEUTRAL)
        self.btn_record.setStyleSheet(BTN_REC_IDLE)
        self.btn_disconnect.setEnabled(False)
        self.btn_record.setEnabled(False)
        for btn in (self.btn_connect, self.btn_disconnect, self.btn_record):
            ctrl.addWidget(btn)
        ctrl.addStretch()
        self.btn_connect.clicked.connect(self._connect)
        self.btn_disconnect.clicked.connect(self._disconnect)
        self.btn_record.clicked.connect(self._toggle_record)

        status = QHBoxLayout()
        status.setSpacing(18)
        self.lbl_cam = QLabel("● Camera: Not connected")
        self.lbl_cam.setStyleSheet(
            f"color:{COLOR_DANGER}; font-weight:600; font-size:{FS_BODY}px; background:transparent;"
        )
        self.lbl_rec = QLabel("")
        self.lbl_rec.setStyleSheet(
            f"color:{COLOR_DANGER}; font-weight:600; font-size:{FS_BODY}px; background:transparent;"
        )
        self.lbl_fps = QLabel("")
        self.lbl_fps.setStyleSheet(
            f"color:{COLOR_TEXT_DIM}; font-size:{FS_LABEL}px; background:transparent;"
        )
        for w in (self.lbl_cam, self.lbl_rec, self.lbl_fps):
            status.addWidget(w)
        status.addStretch()

        self.view = FrameView("Connect the OAK camera to see live preview")

        layout.addLayout(ctrl)
        layout.addLayout(status)
        layout.addWidget(self.view, 1)

    # ── camera ──────────────────────────────────────────

    def _connect(self):
        if self._capture:
            return
        self.lbl_cam.setText("● Camera: Connecting…")
        self.lbl_cam.setStyleSheet(
            f"color:{COLOR_WARN}; font-weight:600; font-size:{FS_BODY}px; background:transparent;"
        )
        self.btn_connect.setEnabled(False)

        self._capture = OAKCaptureThread(resolution=(1920, 1080))
        self._capture.connectionReady.connect(self._on_connected)
        self._capture.frameReady.connect(self._on_frame)
        self._capture.finished.connect(self._on_finished)
        self._capture.connectionLost.connect(self._on_lost)
        self._capture.start()

    @Slot()
    def _on_connected(self):
        self.btn_disconnect.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.lbl_cam.setText("● Camera: Connected")
        self.lbl_cam.setStyleSheet(
            f"color:{COLOR_SUCCESS}; font-weight:600; font-size:{FS_BODY}px; background:transparent;"
        )

    def _disconnect(self):
        if self._recording:
            self._stop_recording()
        if self._capture:
            self._capture.stop()
            self._capture.wait()
            self._capture = None
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.lbl_cam.setText("● Camera: Not connected")
        self.lbl_cam.setStyleSheet(
            f"color:{COLOR_DANGER}; font-weight:600; font-size:{FS_BODY}px; background:transparent;"
        )
        self.lbl_rec.setText("")
        self.lbl_fps.setText("")

    # ── recording ────────────────────────────────────────

    def _toggle_record(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        timestamp      = time.strftime("%Y%m%d_%H%M%S")
        self._rec_path = os.path.join(tempfile.gettempdir(), f"recording_{timestamp}.mov")
        fourcc         = cv2.VideoWriter_fourcc(*"mp4v")
        writer         = cv2.VideoWriter(self._rec_path, fourcc, 30, (1920, 1080))
        if not writer.isOpened():
            try:
                writer.release()
            except Exception:
                pass
            self._writer = None
            QMessageBox.warning(self, "Recording Error", "Could not open VideoWriter. Check disk space.")
            return
        self._writer   = writer
        self._recording   = True
        self._frame_count = 0
        self._start_time  = time.time()
        self.btn_record.setText("⏹  Stop Recording")
        self.btn_record.setStyleSheet(BTN_REC_ACTIVE)
        self.btn_disconnect.setEnabled(False)
        self.lbl_rec.setText("🔴  Recording…")
        self.lbl_rec.setStyleSheet(
            f"color:{COLOR_DANGER}; font-weight:600; font-size:{FS_BODY}px; background:transparent;"
        )
        print(f"🔴 Recording started → {self._rec_path}")

    def _stop_recording(self):
        self._recording = False
        if self._writer:
            self._writer.release()
            self._writer = None
        elapsed = time.time() - self._start_time
        print(f"⏹  Recording stopped — {self._frame_count} frames, {elapsed:.1f}s")
        self.btn_record.setText("⏺  Start Recording")
        self.btn_record.setStyleSheet(BTN_REC_IDLE)
        self.btn_disconnect.setEnabled(True)
        self.lbl_rec.setText("")
        self._prompt_save()

    def _prompt_save(self):
        dlg = SaveVideoDialog(self._rec_path, parent=self)
        if dlg.exec() != QDialog.Accepted or not dlg.chosen_dir:
            QMessageBox.information(
                self, "Saved to Temp",
                f"File kept in temporary location:\n{self._rec_path}"
            )
            return
        dest = os.path.join(dlg.chosen_dir, os.path.basename(self._rec_path))
        try:
            shutil.move(self._rec_path, dest)
            print(f"✅ Recording saved to: {dest}")
            QMessageBox.information(self, "Saved", f"Recording saved to:\n{dest}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not move file:\n{e}")

    # ── frame slot ───────────────────────────────────────

    @Slot(object)
    def _on_frame(self, frame):
        now = time.time()
        self._fps_times.append(now)
        if len(self._fps_times) >= 2:
            span = self._fps_times[-1] - self._fps_times[0]
            self.lbl_fps.setText(f"FPS: {(len(self._fps_times)-1)/span:.1f}" if span > 0 else "")

        if self._recording and self._writer:
            self._writer.write(frame)
            self._frame_count += 1
            elapsed = now - self._start_time

            display = frame.copy()
            h, w    = display.shape[:2]
            cv2.circle(display, (w - 40, 30), 12, (0, 0, 255), -1)
            cv2.putText(display, f"REC  {int(elapsed)}s", (w - 140, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.view.show_frame(display)
            self.lbl_rec.setText(
                f"🔴  Recording…  {self._frame_count} frames  ({elapsed:.0f}s)"
            )
        else:
            self.view.show_frame(frame)

    # ── camera state callbacks ───────────────────────────

    @Slot()
    def _on_finished(self):
        if self._recording:
            self._stop_recording()
        self._capture = None
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.lbl_cam.setText("● Camera: Disconnected")
        self.lbl_cam.setStyleSheet(
            f"color:{COLOR_DANGER}; font-weight:600; font-size:{FS_BODY}px; background:transparent;"
        )

    @Slot(str)
    def _on_lost(self, message):
        if self._recording:
            self._stop_recording()
        self._capture = None
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.lbl_cam.setText("● Camera: Lost")
        self.lbl_cam.setStyleSheet(
            f"color:{COLOR_DANGER}; font-weight:600; font-size:{FS_BODY}px; background:transparent;"
        )
        QMessageBox.warning(self, "Connection Lost", message)

    def cleanup(self):
        if self._recording:
            self._recording = False
            if self._writer:
                self._writer.release()
                self._writer = None
        if self._capture:
            self._capture.stop()
            self._capture.wait()
            self._capture = None
