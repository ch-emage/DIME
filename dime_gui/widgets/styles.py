# =========================================================
# widgets/styles.py — Button/field stylesheets + global APP_QSS
# =========================================================

from widgets.theme import (
    COLOR_BG, COLOR_SURFACE, COLOR_SURFACE_HI, COLOR_BORDER, COLOR_BORDER_HI,
    COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_TEXT_DIM,
    COLOR_PRIMARY, COLOR_PRIMARY_HI, COLOR_SUCCESS, COLOR_SUCCESS_HI,
    COLOR_DANGER, COLOR_DANGER_HI, COLOR_REC, COLOR_REC_HI,
    COLOR_LOG_BG,
    FS_BODY, FS_SUBTITLE,
)


def _btn(bg: str, bg_hover: str) -> str:
    return (
        f"QPushButton {{ background:{bg}; color:white; font-size:{FS_BODY}px; "
        f"font-weight:600; border:none; border-radius:6px; padding:8px 18px; }}"
        f"QPushButton:hover {{ background:{bg_hover}; }}"
        f"QPushButton:pressed {{ background:{bg}; padding-top:9px; padding-bottom:7px; }}"
        f"QPushButton:disabled {{ background:#2a2e38; color:{COLOR_TEXT_DIM}; }}"
    )


FIELD_CSS = (
    f"background:{COLOR_SURFACE}; color:{COLOR_TEXT}; "
    f"border:1px solid {COLOR_BORDER}; border-radius:5px; "
    f"padding:5px 10px; font-size:{FS_BODY}px; "
    f"selection-background-color:{COLOR_PRIMARY};"
)

GROUP_CSS = (
    f"QGroupBox {{ color:{COLOR_TEXT_MUTED}; font-weight:600; "
    f"font-size:{FS_SUBTITLE}px; border:1px solid {COLOR_BORDER}; "
    f"border-radius:8px; margin-top:14px; padding:14px 12px 10px 12px; }}"
    f"QGroupBox::title {{ subcontrol-origin:margin; left:12px; padding:0 6px; }}"
)

BTN_PRIMARY   = _btn(COLOR_SUCCESS, COLOR_SUCCESS_HI)   # start / affirmative
BTN_ACCENT    = _btn(COLOR_PRIMARY, COLOR_PRIMARY_HI)   # primary blue action
BTN_DANGER    = _btn(COLOR_DANGER, COLOR_DANGER_HI)
BTN_NEUTRAL   = (
    f"QPushButton {{ background:{COLOR_SURFACE_HI}; color:{COLOR_TEXT}; "
    f"font-size:{FS_BODY}px; font-weight:500; "
    f"border:1px solid {COLOR_BORDER}; border-radius:6px; padding:6px 14px; }}"
    f"QPushButton:hover {{ background:{COLOR_BORDER_HI}; border-color:{COLOR_BORDER_HI}; }}"
    f"QPushButton:disabled {{ background:#24272e; color:{COLOR_TEXT_DIM}; "
    f"border-color:{COLOR_BORDER}; }}"
)
BTN_REC_IDLE   = _btn(COLOR_REC, COLOR_REC_HI)
BTN_REC_ACTIVE = (
    f"QPushButton {{ background:{COLOR_REC_HI}; color:white; "
    f"font-size:{FS_BODY}px; font-weight:600; border:none; "
    f"border-radius:6px; padding:8px 18px; }}"
    f"QPushButton:hover {{ background:#ff5e4d; }}"
)


APP_QSS = f"""
* {{
  outline: 0;
}}
QMainWindow, QDialog, QWidget {{
  background: {COLOR_BG};
  color: {COLOR_TEXT};
  font-size: {FS_BODY}px;
}}
QLabel {{
  color: {COLOR_TEXT};
  font-size: {FS_BODY}px;
  background: transparent;
}}

/* ── tabs ─────────────────────────────────────────── */
QTabWidget::pane {{
  border: 1px solid {COLOR_BORDER};
  border-radius: 8px;
  background: {COLOR_BG};
  top: -1px;
}}
QTabBar::tab {{
  background: transparent;
  color: {COLOR_TEXT_MUTED};
  padding: 10px 22px;
  font-size: {FS_BODY}px;
  font-weight: 500;
  border: 1px solid transparent;
  border-top-left-radius: 8px;
  border-top-right-radius: 8px;
  margin-right: 2px;
  min-width: 110px;
}}
QTabBar::tab:hover {{
  color: {COLOR_TEXT};
  background: {COLOR_SURFACE};
}}
QTabBar::tab:selected {{
  color: {COLOR_TEXT};
  background: {COLOR_BG};
  border: 1px solid {COLOR_BORDER};
  border-bottom-color: {COLOR_BG};
  font-weight: 600;
}}

/* ── default button (overridden by BTN_* stylesheets) ─── */
QPushButton {{
  background: {COLOR_SURFACE_HI};
  color: {COLOR_TEXT};
  border: 1px solid {COLOR_BORDER};
  border-radius: 6px;
  padding: 6px 14px;
  font-size: {FS_BODY}px;
  font-weight: 500;
}}
QPushButton:hover {{
  background: {COLOR_BORDER_HI};
  border-color: {COLOR_BORDER_HI};
}}
QPushButton:pressed {{
  background: {COLOR_BORDER};
}}
QPushButton:disabled {{
  background: #24272e;
  color: {COLOR_TEXT_DIM};
  border-color: {COLOR_BORDER};
}}

/* ── line edit / spin box ─────────────────────────── */
QLineEdit, QDoubleSpinBox, QSpinBox {{
  background: {COLOR_SURFACE};
  color: {COLOR_TEXT};
  border: 1px solid {COLOR_BORDER};
  border-radius: 5px;
  padding: 5px 10px;
  font-size: {FS_BODY}px;
  selection-background-color: {COLOR_PRIMARY};
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
  border-color: {COLOR_PRIMARY};
}}

/* ── text edits / list widgets ────────────────────── */
QTextEdit {{
  background: {COLOR_LOG_BG};
  color: {COLOR_TEXT};
  border: 1px solid {COLOR_BORDER};
  border-radius: 6px;
  selection-background-color: {COLOR_PRIMARY};
}}
QListWidget {{
  background: {COLOR_SURFACE};
  color: {COLOR_TEXT};
  border: 1px solid {COLOR_BORDER};
  border-radius: 6px;
  padding: 2px;
}}
QListWidget::item {{ padding: 4px 6px; }}
QListWidget::item:selected {{
  background: {COLOR_PRIMARY};
  color: white;
  border-radius: 3px;
}}

/* ── splitter ─────────────────────────────────────── */
QSplitter::handle {{
  background: {COLOR_BORDER};
}}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical   {{ height: 2px; }}

/* ── scroll area / scroll bars ────────────────────── */
QScrollArea {{ border: none; background: {COLOR_BG}; }}
QScrollBar:vertical {{
  background: transparent; width: 10px; margin: 2px 0; border: none;
}}
QScrollBar::handle:vertical {{
  background: {COLOR_SURFACE_HI}; border-radius: 5px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {COLOR_BORDER_HI}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
  background: none; height: 0; width: 0;
}}
QScrollBar:horizontal {{
  background: transparent; height: 10px; margin: 0 2px; border: none;
}}
QScrollBar::handle:horizontal {{
  background: {COLOR_SURFACE_HI}; border-radius: 5px; min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{ background: {COLOR_BORDER_HI}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
  background: none; height: 0; width: 0;
}}

/* ── progress bar ─────────────────────────────────── */
QProgressBar {{
  border: none;
  border-radius: 3px;
  background: {COLOR_SURFACE};
  height: 6px;
  max-height: 6px;
  text-align: center;
  color: {COLOR_TEXT_MUTED};
}}
QProgressBar::chunk {{
  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
      stop:0 {COLOR_PRIMARY}, stop:1 {COLOR_PRIMARY_HI});
  border-radius: 3px;
}}

/* ── check / radio ───────────────────────────────── */
QRadioButton, QCheckBox {{
  color: {COLOR_TEXT};
  spacing: 6px;
  font-size: {FS_BODY}px;
  background: transparent;
}}

/* ── tooltip ─────────────────────────────────────── */
QToolTip {{
  background: {COLOR_SURFACE};
  color: {COLOR_TEXT};
  border: 1px solid {COLOR_BORDER};
  padding: 4px 8px;
  border-radius: 4px;
}}

/* ── dialog buttons ──────────────────────────────── */
QDialogButtonBox QPushButton {{ min-width: 80px; }}

/* ── menus / message boxes ───────────────────────── */
QMessageBox {{ background: {COLOR_BG}; }}
QMessageBox QLabel {{ color: {COLOR_TEXT}; }}
"""
