# =========================================================
# threads/capture_thread.py — Raw OAK capture, NO inference
# Used only by the Record Video tab.
# =========================================================

import depthai as dai

from PySide6.QtCore import QThread, Signal


class OAKCaptureThread(QThread):
    frameReady      = Signal(object)   # raw BGR numpy frame
    connectionReady = Signal()         # emitted once pipeline is running
    finished        = Signal()
    connectionLost  = Signal(str)

    def __init__(self, resolution=(1920, 1080)):
        super().__init__()
        self.resolution = resolution
        self._running   = True

    def run(self):
        pipeline = None
        queue = None
        try:
            try:
                pipeline = dai.Pipeline()
                cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
                cam.initialControl.setAutoWhiteBalanceMode(
                    dai.CameraControl.AutoWhiteBalanceMode.OFF
                )
                video_out = cam.requestOutput(self.resolution, dai.ImgFrame.Type.BGR888i)
                queue = video_out.createOutputQueue(maxSize=1)
                pipeline.start()
                print(f"✅ OAK capture ready: {self.resolution[0]}x{self.resolution[1]}")
            except Exception as e:
                print(f"❌ Failed to open OAK camera: {e}")
                self.connectionLost.emit(str(e))
                return

            self.connectionReady.emit()

            while self._running and pipeline.isRunning():
                self.frameReady.emit(queue.get().getCvFrame())
        except Exception as e:
            print(f"⚠  OAK capture error: {e}")
            self.connectionLost.emit(f"OAK capture error: {e}")
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            self.finished.emit()
            print("🔴 OAK capture stopped.")

    def stop(self):
        self._running = False
