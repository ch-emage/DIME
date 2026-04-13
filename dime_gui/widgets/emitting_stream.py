# =========================================================
# widgets/emitting_stream.py — Redirects stdout/stderr to Qt signal
# =========================================================

from PySide6.QtCore import QObject, Signal


class EmittingStream(QObject):
    textWritten = Signal(str)

    def write(self, text):
        if text.strip():
            self.textWritten.emit(text)

    def flush(self):
        pass
