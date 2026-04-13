# =========================================================
# tabs/train_tab.py — Tab: Train Model
# =========================================================

import os
import sys
import time
import shutil
import subprocess

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QFileDialog, QMessageBox, QSplitter, QScrollArea,
    QGroupBox, QGridLayout, QListWidget, QListWidgetItem, QProgressBar,
    QLineEdit
)
from PySide6.QtCore import Qt, Slot

from threads import TrainingThread
from widgets import FIELD_CSS, GROUP_CSS, BTN_NEUTRAL


class TrainModelSubTab(QWidget):
    def __init__(self):
        super().__init__()
        self._worker: TrainingThread | None = None
        self._build_ui()

    # ── layout ───────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        splitter = QSplitter(Qt.Horizontal)

        # Left: config
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border:none; background:#1e1e1e; }")
        cfg_widget = QWidget()
        cfg_layout = QVBoxLayout(cfg_widget)
        cfg_layout.setContentsMargins(4, 4, 4, 4)
        cfg_layout.setSpacing(8)
        cfg_layout.addWidget(self._build_paths_group())
        cfg_layout.addWidget(self._build_videos_group())
        cfg_layout.addStretch()
        scroll.setWidget(cfg_widget)
        splitter.addWidget(scroll)

        # Right: log
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        log_title = QLabel("Training Log")
        log_title.setStyleSheet("color:#aaa; font-weight:bold; font-size:13px;")
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(
            "background:#0d0d0d; color:#b5e7a0; font-family:monospace; font-size:12px; border-radius:6px;"
        )
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border:none; border-radius:4px; background:#222; }"
            "QProgressBar::chunk { background:#27ae60; border-radius:4px; }"
        )
        self.lbl_progress = QLabel("")
        self.lbl_progress.setStyleSheet("color:#27ae60; font-size:12px; font-weight:bold;")
        right_layout.addWidget(log_title)
        right_layout.addWidget(self.log_view, 1)
        right_layout.addWidget(self.progress_bar)
        right_layout.addWidget(self.lbl_progress)
        splitter.addWidget(right)
        splitter.setSizes([420, 760])
        root.addWidget(splitter, 1)

        # Bottom buttons
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start Training")
        self.btn_start.setFixedHeight(38)
        self.btn_start.setStyleSheet(
            "QPushButton { background:#1a6b27; color:white; font-weight:bold; border-radius:4px; }"
            "QPushButton:hover { background:#27ae60; }"
            "QPushButton:disabled { background:#333; color:#666; }"
        )
        self.btn_start.clicked.connect(self._start_training)

        self.btn_stop = QPushButton("⏹  Stop")
        self.btn_stop.setFixedHeight(38)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            "QPushButton { background:#7b241c; color:white; font-weight:bold; border-radius:4px; }"
            "QPushButton:hover { background:#c0392b; }"
            "QPushButton:disabled { background:#333; color:#666; }"
        )
        self.btn_stop.clicked.connect(self._stop_training)

        self.btn_clear_log = QPushButton("🗑  Clear Log")
        self.btn_clear_log.setFixedHeight(38)
        self.btn_clear_log.setStyleSheet(
            "QPushButton { background:#2c3e50; color:white; font-weight:bold; border-radius:4px; }"
            "QPushButton:hover { background:#3d566e; }"
        )
        self.btn_clear_log.clicked.connect(self.log_view.clear)

        self.btn_open_output = QPushButton("📂  Open Output Folder")
        self.btn_open_output.setFixedHeight(38)
        self.btn_open_output.setStyleSheet(
            "QPushButton { background:#2c3e50; color:white; font-weight:bold; border-radius:4px; }"
            "QPushButton:hover { background:#3d566e; }"
        )
        self.btn_open_output.clicked.connect(self._open_output_folder)

        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_clear_log)
        btn_row.addWidget(self.btn_open_output)
        root.addLayout(btn_row)
        self.setLayout(root)

    # ── config groups ────────────────────────────────────

    def _row(self, label_text, widget, layout):
        row = layout.rowCount()
        lbl = QLabel(label_text)
        lbl.setStyleSheet("color:#ccc; font-size:12px;")
        layout.addWidget(lbl, row, 0)
        layout.addWidget(widget, row, 1)

    def _build_paths_group(self) -> QGroupBox:
        grp = QGroupBox("Paths & Script")
        grp.setStyleSheet(GROUP_CSS)
        g = QGridLayout(grp)
        g.setColumnStretch(1, 1)

        self.edit_script = QLineEdit()
        self.edit_script.setPlaceholderText("Path to train.py …")
        self.edit_script.setStyleSheet(FIELD_CSS)
        btn_script = QPushButton("Browse")
        btn_script.setStyleSheet(BTN_NEUTRAL)
        btn_script.clicked.connect(self._browse_script)
        rw = QHBoxLayout()
        rw.addWidget(self.edit_script)
        rw.addWidget(btn_script)
        w = QWidget()
        w.setLayout(rw)
        self._row("Training script", w, g)

        self.edit_workspace = QLineEdit()
        self.edit_workspace.setPlaceholderText("Temp data staging folder …")
        self.edit_workspace.setStyleSheet(FIELD_CSS)
        btn_ws = QPushButton("Browse")
        btn_ws.setStyleSheet(BTN_NEUTRAL)
        btn_ws.clicked.connect(self._browse_workspace)
        rw2 = QHBoxLayout()
        rw2.addWidget(self.edit_workspace)
        rw2.addWidget(btn_ws)
        w2 = QWidget()
        w2.setLayout(rw2)
        self._row("Workspace (data_path)", w2, g)

        self.edit_results = QLineEdit()
        self.edit_results.setText("MODEL/")
        self.edit_results.setStyleSheet(FIELD_CSS)
        btn_res = QPushButton("Browse")
        btn_res.setStyleSheet(BTN_NEUTRAL)
        btn_res.clicked.connect(self._browse_results)
        rw3 = QHBoxLayout()
        rw3.addWidget(self.edit_results)
        rw3.addWidget(btn_res)
        w3 = QWidget()
        w3.setLayout(rw3)
        self._row("Output / results path", w3, g)

        self.edit_project = QLineEdit("CAMERA_1")
        self.edit_project.setStyleSheet(FIELD_CSS)
        self._row("Log project", self.edit_project, g)

        self.edit_group = QLineEdit("SLC")
        self.edit_group.setStyleSheet(FIELD_CSS)
        self._row("Log group", self.edit_group, g)

        return grp

    def _build_videos_group(self) -> QGroupBox:
        grp = QGroupBox("Training Videos  (normal / good samples)")
        grp.setStyleSheet(GROUP_CSS)
        layout = QVBoxLayout(grp)

        sub_row = QHBoxLayout()
        sub_lbl = QLabel("Subdataset name:")
        sub_lbl.setStyleSheet("color:#ccc; font-size:12px;")
        self.edit_subdataset = QLineEdit("eyeball")
        self.edit_subdataset.setStyleSheet(FIELD_CSS)
        self.edit_subdataset.setMaximumWidth(200)
        sub_row.addWidget(sub_lbl)
        sub_row.addWidget(self.edit_subdataset)
        sub_row.addStretch()
        layout.addLayout(sub_row)

        self.video_list = QListWidget()
        self.video_list.setStyleSheet(
            "background:#111; color:#eee; border:1px solid #333; border-radius:4px; font-size:12px;"
        )
        self.video_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.video_list.setFixedHeight(120)
        layout.addWidget(self.video_list)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("➕  Add Videos")
        btn_add.setStyleSheet(BTN_NEUTRAL)
        btn_add.clicked.connect(self._add_videos)
        btn_rem = QPushButton("➖  Remove Selected")
        btn_rem.setStyleSheet(BTN_NEUTRAL)
        btn_rem.clicked.connect(self._remove_selected_videos)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_rem)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        note = QLabel("ℹ Videos will be copied into  <workspace>/<subdataset>/train/good/  before training.")
        note.setStyleSheet("color:#666; font-size:11px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        return grp

    # ── browse helpers ───────────────────────────────────

    def _browse_script(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select train.py", "", "Python files (*.py)")
        if path:
            self.edit_script.setText(path)

    def _browse_workspace(self):
        path = QFileDialog.getExistingDirectory(self, "Select Workspace / Data Staging Folder")
        if path:
            self.edit_workspace.setText(path)

    def _browse_results(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output / Results Folder")
        if path:
            self.edit_results.setText(path)

    def _add_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Training Videos", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.MOV *.MP4)"
        )
        existing = {self.video_list.item(i).data(Qt.UserRole) for i in range(self.video_list.count())}
        for p in paths:
            if p not in existing:
                item = QListWidgetItem(os.path.basename(p))
                item.setData(Qt.UserRole, p)
                item.setToolTip(p)
                self.video_list.addItem(item)

    def _remove_selected_videos(self):
        for item in self.video_list.selectedItems():
            self.video_list.takeItem(self.video_list.row(item))

    def _open_output_folder(self):
        path = self.edit_results.text().strip()
        if path and os.path.isdir(path):
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        else:
            QMessageBox.information(self, "Folder not found", f"Path does not exist yet:\n{path}")

    # ── validation & staging ─────────────────────────────

    def _validate(self) -> tuple[bool, str]:
        if not self.edit_script.text().strip():
            return False, "Please select the training script (train.py)."
        if not os.path.isfile(self.edit_script.text().strip()):
            return False, f"Training script not found:\n{self.edit_script.text()}"
        if not self.edit_workspace.text().strip():
            return False, "Please set a workspace folder."
        if not self.edit_results.text().strip():
            return False, "Please set an output / results path."
        if not self.edit_subdataset.text().strip():
            return False, "Please enter a subdataset name."
        if self.video_list.count() == 0:
            return False, "Please add at least one training video."
        return True, ""

    def _stage_videos(self, workspace: str, subdataset: str) -> bool:
        dest = os.path.join(workspace, subdataset, "train", "good")
        os.makedirs(dest, exist_ok=True)
        self._log(f"📁  Staging videos → {dest}")
        for i in range(self.video_list.count()):
            src = self.video_list.item(i).data(Qt.UserRole)
            dst = os.path.join(dest, os.path.basename(src))
            if os.path.abspath(src) == os.path.abspath(dst):
                self._log(f"   skip (same path): {os.path.basename(src)}")
                continue
            try:
                shutil.copy2(src, dst)
                self._log(f"   ✅  {os.path.basename(src)}")
            except Exception as e:
                self._log(f"   ❌  Failed to copy {os.path.basename(src)}: {e}")
                return False
        return True

    def _build_args(self, workspace, subdataset) -> list:
        return [
            "--results_path", self.edit_results.text().strip(),
            "--data_path",    workspace,
            "--subdatasets",  subdataset,
            "--log_project",  self.edit_project.text().strip() or "CAMERA",
            "--log_group",    self.edit_group.text().strip() or "SLC",
            "--video-to-frames", "--save-model",
        ]

    # ── training control ─────────────────────────────────

    def _log(self, text: str):
        self.log_view.append(text)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _start_training(self):
        ok, msg = self._validate()
        if not ok:
            QMessageBox.warning(self, "Configuration error", msg)
            return
        workspace  = self.edit_workspace.text().strip()
        subdataset = self.edit_subdataset.text().strip()
        self._log("=" * 60)
        self._log(f"🗓  Starting training session  —  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("=" * 60)
        if not self._stage_videos(workspace, subdataset):
            self._log("❌  Staging failed — aborting.")
            return
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.lbl_progress.setText("⏳  Training in progress…")
        self._worker = TrainingThread(self.edit_script.text().strip(), self._build_args(workspace, subdataset))
        self._worker.logLine.connect(self._log)
        self._worker.progressHint.connect(self.lbl_progress.setText)
        self._worker.finished.connect(self._on_training_finished)
        self._worker.start()

    def _stop_training(self):
        if self._worker:
            self._worker.abort()
            self.btn_stop.setEnabled(False)
            self.lbl_progress.setText("⏹  Stopping…")

    @Slot(bool, str)
    def _on_training_finished(self, success: bool, message: str):
        self._worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        color = "#27ae60" if success else "#e74c3c"
        self.lbl_progress.setStyleSheet(f"color:{color}; font-size:13px; font-weight:bold;")
        self.lbl_progress.setText(message)
        self._log(f"\n{'✅' if success else '❌'}  {message}\n")

    def cleanup(self):
        if self._worker:
            self._worker.abort()
            self._worker.wait()
