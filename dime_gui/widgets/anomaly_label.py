# =========================================================
# widgets/anomaly_label.py — Anomaly status banner helpers
# =========================================================

from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt

_STYLE_NORMAL  = (
    "background:#1e8449; color:white; font-size:16px; font-weight:bold; "
    "border-radius:5px; padding:4px 16px; letter-spacing:1px;"
)
_STYLE_ANOMALY = (
    "background:#c0392b; color:white; font-size:16px; font-weight:bold; "
    "border-radius:5px; padding:4px 16px; letter-spacing:1px;"
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
