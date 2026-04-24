# ==========================================================
# main.py — Application entry point + main window
# ==========================================================

import sys
import os
import html
import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLabel, QPushButton, QSplitter, QTabWidget
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon, QKeySequence, QShortcut

from widgets import (
    EmittingStream, APP_QSS, BTN_NEUTRAL,
    COLOR_BG, COLOR_SURFACE, COLOR_SURFACE_HI, COLOR_BORDER,
    COLOR_TEXT, COLOR_TEXT_MUTED,
    COLOR_SUCCESS, COLOR_WARN, COLOR_DANGER, COLOR_REC, COLOR_PRIMARY,
    FS_BODY,
)
from tabs import ModelTab, MediaTab, LiveTab, RecordSubTab, TrainModelSubTab


# Map leading emoji → log color. Plain log lines use muted text.
_LOG_COLOR_BY_GLYPH = {
    "❌": COLOR_DANGER,
    "⚠":  COLOR_WARN,
    "✅": COLOR_SUCCESS,
    "🔴": COLOR_REC,
    "💾": COLOR_PRIMARY,
    "🎯": COLOR_PRIMARY,
    "📦": COLOR_PRIMARY,
    "🎚": COLOR_PRIMARY,
    "📁": COLOR_TEXT_MUTED,
    "📹": COLOR_TEXT_MUTED,
    "📋": COLOR_TEXT_MUTED,
    "🗓": COLOR_TEXT_MUTED,
    "⏹": COLOR_TEXT_MUTED,
    "⏳": COLOR_WARN,
    "ℹ":  COLOR_TEXT_MUTED,
}


def _log_color_for(line: str) -> str:
    """Return the text color for a log line based on its leading glyph."""
    stripped = line.lstrip()
    for glyph, color in _LOG_COLOR_BY_GLYPH.items():
        if stripped.startswith(glyph):
            return color
    return COLOR_TEXT


class DIMEMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIME Inference & Training GUI")
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Icon.png")
        self.setWindowIcon(QIcon(icon_path))
        self.resize(1400, 860)
        self._build_ui()
        self._redirect_stdout()

    def _build_ui(self):
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.tab_record = RecordSubTab()
        self.tab_model  = ModelTab()
        self.tab_media  = MediaTab()
        self.tab_live   = LiveTab()
        self.tab_train  = TrainModelSubTab()

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMinimumHeight(80)
        self.logs.setStyleSheet(
            f"QTextEdit {{ background:#0b0d10; color:{COLOR_TEXT}; "
            f"border:1px solid {COLOR_BORDER}; border-radius:6px; "
            f"font-family:'JetBrains Mono','Fira Code','Menlo','Consolas',monospace; "
            f"font-size:12px; padding:6px 8px; }}"
        )
        self.logs.document().setMaximumBlockCount(5000)

        self.tabs.addTab(self.tab_record, "📹   Record")
        self.tabs.addTab(self.tab_model,  "⚙    Model")
        self.tabs.addTab(self.tab_media,  "🖼    Media")
        self.tabs.addTab(self.tab_live,   "📡   Live")
        self.tabs.addTab(self.tab_train,  "🏋    Train")

        self.tab_model.modelLoaded.connect(self.tab_media.set_detector)
        self.tab_model.modelLoaded.connect(self.tab_live.set_detector)
        self.tab_model.modelLoaded.connect(self._on_model_loaded)

        log_header = QWidget()
        log_header_layout = QHBoxLayout(log_header)
        log_header_layout.setContentsMargins(0, 0, 0, 0)
        lbl_log = QLabel("📋  Log")
        lbl_log.setStyleSheet(
            f"color:{COLOR_TEXT_MUTED}; font-weight:600; font-size:{FS_BODY}px;"
        )
        log_header_layout.addWidget(lbl_log)
        log_header_layout.addStretch()
        self.btn_log_clear = QPushButton("Clear")
        self.btn_log_clear.setFixedHeight(22)
        self.btn_log_clear.setCursor(Qt.PointingHandCursor)
        self.btn_log_clear.setToolTip("Clear log buffer")
        self.btn_log_clear.setStyleSheet(
            f"QPushButton {{ background:{COLOR_SURFACE}; color:{COLOR_TEXT_MUTED}; "
            f"border:1px solid {COLOR_BORDER}; border-radius:4px; "
            f"padding:2px 10px; font-size:11px; font-weight:500; }}"
            f"QPushButton:hover {{ background:{COLOR_SURFACE_HI}; color:{COLOR_TEXT}; }}"
        )
        self.btn_log_clear.clicked.connect(self.logs.clear)
        log_header_layout.addWidget(self.btn_log_clear)

        self.btn_log_toggle = QPushButton("▼ Hide")
        self.btn_log_toggle.setFixedHeight(22)
        self.btn_log_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_log_toggle.setToolTip("Toggle log panel (Ctrl+L)")
        self.btn_log_toggle.setStyleSheet(
            f"QPushButton {{ background:{COLOR_SURFACE}; color:{COLOR_TEXT_MUTED}; "
            f"border:1px solid {COLOR_BORDER}; border-radius:4px; "
            f"padding:2px 10px; font-size:11px; font-weight:500; }}"
            f"QPushButton:hover {{ background:{COLOR_SURFACE_HI}; color:{COLOR_TEXT}; }}"
        )
        self.btn_log_toggle.clicked.connect(self._toggle_log)
        log_header_layout.addSpacing(4)
        log_header_layout.addWidget(self.btn_log_toggle)

        self._log_box = QWidget()
        log_layout = QVBoxLayout(self._log_box)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(4)
        log_layout.addWidget(log_header)
        log_layout.addWidget(self.logs)

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.tabs)
        self.splitter.addWidget(self._log_box)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([800, 140])
        self.splitter.setChildrenCollapsible(False)
        self._log_sizes = self.splitter.sizes()

        QShortcut(QKeySequence("Ctrl+L"), self, activated=self._toggle_log)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        layout.addWidget(self.splitter)
        self.setCentralWidget(root)

    def _toggle_log(self):
        if self.logs.isVisible():
            self._log_sizes = self.splitter.sizes()
            self.logs.hide()
            self.btn_log_toggle.setText("▲ Show")
        else:
            self.logs.show()
            self.btn_log_toggle.setText("▼ Hide")
            self.splitter.setSizes(self._log_sizes)

    def _redirect_stdout(self):
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._stdout = EmittingStream()
        self._stderr = EmittingStream()
        self._stdout.textWritten.connect(self._append_log)
        self._stderr.textWritten.connect(self._append_log)
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    @Slot(str)
    def _append_log(self, text):
        for raw_line in text.splitlines() or [text]:
            line = raw_line.rstrip()
            if not line:
                continue
            color = _log_color_for(line)
            safe = html.escape(line).replace("  ", "&nbsp;&nbsp;")
            self.logs.append(f'<span style="color:{color};">{safe}</span>')
        self.logs.verticalScrollBar().setValue(
            self.logs.verticalScrollBar().maximum()
        )

    @Slot(object, object, object)
    def _on_model_loaded(self, detector, roi_coords, thresholds):
        self.tabs.setCurrentWidget(self.tab_media)

    def closeEvent(self, event):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        for tab, name in (
            (self.tab_record, "Record"),
            (self.tab_media,  "Media"),
            (self.tab_live,   "Live"),
            (self.tab_train,  "Train"),
        ):
            try:
                tab.cleanup()
            except Exception as e:
                print(f"⚠  Error cleaning up {name} tab: {e}")
        if self.tab_model.detector:
            try:
                self.tab_model.detector.cleanup()
            except Exception as e:
                print(f"⚠  Error cleaning up detector: {e}")
            finally:
                self.tab_model.detector = None
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_QSS)
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Icon.png")
    app.setWindowIcon(QIcon(icon_path))
    win = DIMEMainWindow()
    win.showMaximized()
    sys.exit(app.exec())
