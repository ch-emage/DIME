# =========================================================
# widgets/frame_view.py — Scalable BGR frame display label
# =========================================================

import cv2

from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

from widgets.theme import COLOR_FRAME_BG, COLOR_BORDER, COLOR_TEXT_DIM


class FrameView(QLabel):
    def __init__(self, placeholder="Output"):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 270)
        self.setStyleSheet(
            f"QLabel {{ background:{COLOR_FRAME_BG}; color:{COLOR_TEXT_DIM}; "
            f"font-size:16px; border:1px solid {COLOR_BORDER}; border-radius:8px; }}"
        )
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def show_frame(self, frame):
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        self.setPixmap(
            QPixmap.fromImage(qimg).scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        )
