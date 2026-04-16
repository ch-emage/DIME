# ==========================================================
# main.py — Application entry point + main window
# ==========================================================

import sys
import os
import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QTextEdit, QLabel, QSplitter, QTabWidget
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon

from widgets import EmittingStream
from tabs import ModelTab, MediaTab, LiveTab, RecordSubTab, TrainModelSubTab


class DIMEMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIME Inference & Training GUI")
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Icon.png")
        self.setWindowIcon(QIcon(icon_path))
        self.resize(1400, 860)
        self.setStyleSheet("background:#1e1e1e; color:#eee;")
        self._build_ui()
        self._redirect_stdout()

    def _build_ui(self):
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border:1px solid #333; background:#1e1e1e; }
            QTabBar::tab {
                background:#2a2a2a; color:#aaa;
                padding:10px 24px; font-size:13px; font-weight:bold;
                border-top-left-radius:6px; border-top-right-radius:6px; margin-right:2px;
            }
            QTabBar::tab:selected { background:#1e1e1e; color:#fff; border-bottom:2px solid #2471a3; }
            QTabBar::tab:hover    { background:#333; color:#fff; }
        """)

        self.tab_record = RecordSubTab()
        self.tab_model  = ModelTab()
        self.tab_media  = MediaTab()
        self.tab_live   = LiveTab()
        self.tab_train  = TrainModelSubTab()

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMinimumHeight(80)
        self.logs.setStyleSheet("background:#111; color:#0f0; font-family:monospace; font-size:12px;")
        self.logs.document().setMaximumBlockCount(5000)

        self.tabs.addTab(self.tab_record, "📹  Record Video")
        self.tabs.addTab(self.tab_model,  "⚙  Model Setup")
        self.tabs.addTab(self.tab_media,  "🖼  Image & Video")
        self.tabs.addTab(self.tab_live,   "📡  Live Camera")
        self.tabs.addTab(self.tab_train,  "🏋  Train Model")

        # Wire model-loaded signal to the tabs that consume it
        self.tab_model.modelLoaded.connect(self.tab_media.set_detector)
        self.tab_model.modelLoaded.connect(self.tab_live.set_detector)
        self.tab_model.modelLoaded.connect(self._on_model_loaded)

        log_box = QWidget()
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(2)
        log_layout.addWidget(QLabel("📋 Log"))
        log_layout.addWidget(self.logs)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.tabs)
        splitter.addWidget(log_box)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([800, 140])
        splitter.setChildrenCollapsible(False)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(splitter)
        self.setCentralWidget(root)

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
        self.logs.append(text.rstrip())
        self.logs.verticalScrollBar().setValue(self.logs.verticalScrollBar().maximum())

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
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Icon.png")
    app.setWindowIcon(QIcon(icon_path))
    win = DIMEMainWindow()
    win.showMaximized()
    sys.exit(app.exec())
