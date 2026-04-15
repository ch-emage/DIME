# =========================================================
# threads/training_thread.py — Subprocess wrapper for train.py
# =========================================================

import sys
import subprocess

from PySide6.QtCore import QThread, Signal


class TrainingThread(QThread):
    logLine      = Signal(str)
    finished     = Signal(bool, str)
    progressHint = Signal(str)

    def __init__(self, train_script: str, args: list):
        super().__init__()
        self.train_script = train_script
        self.args  = args
        self._proc = None
        self._abort = False

    def run(self):
        cmd = [sys.executable, self.train_script] + self.args
        self.logLine.emit(f"🚀  Launching: {' '.join(cmd)}\n")
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            try:
                for line in self._proc.stdout:
                    if self._abort:
                        break
                    self.logLine.emit(line.rstrip())
                    low = line.lower()
                    if "training model"  in low: self.progressHint.emit("Training backbone…")
                    elif "validation"    in low: self.progressHint.emit("Running validation…")
                    elif "saving" in low or "save" in low: self.progressHint.emit("Saving model…")
                    elif "dynamic threshold" in low: self.progressHint.emit("Computing threshold…")
            finally:
                if self._proc.stdout:
                    try:
                        self._proc.stdout.close()
                    except Exception:
                        pass

            self._proc.wait()
            if self._abort:
                self.finished.emit(False, "Training cancelled by user.")
            elif self._proc.returncode == 0:
                self.finished.emit(True, "Training completed successfully ✅")
            else:
                self.finished.emit(False, f"Training exited with code {self._proc.returncode} ❌")
        except Exception as e:
            self.finished.emit(False, f"Error: {e}")
        finally:
            self._proc = None

    def abort(self):
        self._abort = True
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
