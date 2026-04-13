# =========================================================
# widgets/styles.py — Shared Qt stylesheet constants
# =========================================================

FIELD_CSS = (
    "background:#111; color:#eee; border:1px solid #444; border-radius:4px; "
    "padding:4px 8px; font-size:13px;"
)
GROUP_CSS = (
    "QGroupBox { color:#aaa; font-weight:bold; border:1px solid #333; "
    "border-radius:6px; margin-top:8px; padding-top:6px; }"
    "QGroupBox::title { subcontrol-origin:margin; left:10px; }"
)
BTN_PRIMARY = (
    "QPushButton { background:#1a6b27; color:white; font-size:13px; font-weight:bold; "
    "border-radius:6px; padding:8px 18px; }"
    "QPushButton:hover { background:#27ae60; }"
    "QPushButton:disabled { background:#333; color:#666; }"
)
BTN_DANGER = (
    "QPushButton { background:#7b241c; color:white; font-size:13px; font-weight:bold; "
    "border-radius:6px; padding:8px 18px; }"
    "QPushButton:hover { background:#c0392b; }"
    "QPushButton:disabled { background:#333; color:#666; }"
)
BTN_NEUTRAL = (
    "QPushButton { background:#2c3e50; color:white; font-size:13px; font-weight:bold; "
    "border-radius:6px; padding:6px 14px; }"
    "QPushButton:hover { background:#3d566e; }"
    "QPushButton:disabled { background:#333; color:#666; }"
)
BTN_REC_IDLE = (
    "QPushButton { background:#7b241c; color:white; font-size:13px; font-weight:bold; "
    "border-radius:6px; padding:6px 14px; }"
    "QPushButton:hover { background:#c0392b; }"
    "QPushButton:disabled { background:#333; color:#666; }"
)
BTN_REC_ACTIVE = (
    "QPushButton { background:#c0392b; color:white; font-size:13px; font-weight:bold; "
    "border-radius:6px; padding:6px 14px; }"
    "QPushButton:hover { background:#e74c3c; }"
)
