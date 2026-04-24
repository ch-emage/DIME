# =========================================================
# widgets/anomaly_label.py — Anomaly status banner helpers
# =========================================================

from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt

from widgets.theme import FS_SUBTITLE


_STYLE_NORMAL = (
    "QLabel { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
    "stop:0 #1e8449, stop:1 #23a55a); "
    f"color:white; font-size:{FS_SUBTITLE + 2}px; font-weight:700; "
    "border-radius:6px; padding:4px 16px; letter-spacing:1px; }"
)
_STYLE_ANOMALY = (
    "QLabel { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
    "stop:0 #a93226, stop:1 #cf3b2b); "
    f"color:white; font-size:{FS_SUBTITLE + 2}px; font-weight:700; "
    "border-radius:6px; padding:4px 16px; letter-spacing:1px; }"
)


def make_anomaly_alert_label() -> QLabel:
    lbl = QLabel("  ✔  Normal")
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setFixedHeight(38)
    lbl.setStyleSheet(_STYLE_NORMAL)
    return lbl


def set_anomaly_label(lbl: QLabel, has_anomaly: bool):
    if has_anomaly:
        lbl.setText("  ⚠  Anomaly Detected")
        lbl.setStyleSheet(_STYLE_ANOMALY)
    else:
        lbl.setText("  ✔  Normal")
        lbl.setStyleSheet(_STYLE_NORMAL)
