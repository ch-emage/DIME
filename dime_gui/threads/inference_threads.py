# =========================================================
# threads/inference_threads.py — QThreads that run the DIME model
# =========================================================

import os
import cv2
import depthai as dai

from PySide6.QtCore import QThread, Signal

from inference import run_inference


class ImageInferenceThread(QThread):
    finished = Signal(object, float, float, bool)

    def __init__(self, detector, roi_coords, image_path):
        super().__init__()
        self.detector   = detector
        self.roi_coords = roi_coords
        self.image_path = image_path

    def run(self):
        frame = cv2.imread(self.image_path)
        if frame is None:
            return
        frame, latency_ms, score, has_anomaly = run_inference(
            self.detector, frame, self.roi_coords
        )
        self.finished.emit(frame, latency_ms, score, has_anomaly)


class VideoInferenceThread(QThread):
    frameProcessed = Signal(object, float, float, bool)
    finished       = Signal()
    connectionLost = Signal(str)

    def __init__(self, detector, roi_coords, source, is_rtsp=False, start_frame=0):
        super().__init__()
        self.detector       = detector
        self.roi_coords     = roi_coords
        self.source         = source
        self.is_rtsp        = is_rtsp
        self.start_frame    = start_frame
        self._running       = True
        self.last_frame_pos = 0

    def run(self):
        if self.is_rtsp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            if self.is_rtsp:
                self.connectionLost.emit(f"Could not connect to: {self.source}")
            self.finished.emit()
            return

        if self.is_rtsp:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"📹 RTSP stream: {w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.1f} FPS")
        elif self.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        failures = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                failures += 1
                if self.is_rtsp and failures < 30:
                    QThread.msleep(50)
                    continue
                if self.is_rtsp:
                    self.connectionLost.emit("RTSP stream connection lost")
                break
            failures = 0
            if not self.is_rtsp:
                self.last_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame, latency_ms, score, has_anomaly = run_inference(
                self.detector, frame, self.roi_coords
            )
            self.frameProcessed.emit(frame, latency_ms, score, has_anomaly)
            if not self.is_rtsp:
                QThread.msleep(30)

        cap.release()
        self.finished.emit()

    def stop(self):
        self._running = False


class OAKInferenceThread(QThread):
    """OAK thread WITH anomaly inference — used in Live Camera tab."""
    frameProcessed = Signal(object, float, float, bool)
    finished       = Signal()
    connectionLost = Signal(str)

    def __init__(self, detector, roi_coords, resolution=(1920, 1080)):
        super().__init__()
        self.detector   = detector
        self.roi_coords = roi_coords
        self.resolution = resolution
        self._running   = True

    def run(self):
        try:
            pipeline = dai.Pipeline()
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            cam.initialControl.setAutoWhiteBalanceMode(
                dai.CameraControl.AutoWhiteBalanceMode.OFF
            )
            video_out = cam.requestOutput(self.resolution, dai.ImgFrame.Type.BGR888i)
            queue = video_out.createOutputQueue(maxSize=1)
            pipeline.start()
            print(f"✅ OAK camera: {self.resolution[0]}x{self.resolution[1]}")
        except Exception as e:
            print(f"❌ Failed to open OAK camera: {e}")
            self.connectionLost.emit(str(e))
            self.finished.emit()
            return

        while self._running and pipeline.isRunning():
            frame = queue.get().getCvFrame()
            frame, latency_ms, score, has_anomaly = run_inference(
                self.detector, frame, self.roi_coords
            )
            self.frameProcessed.emit(frame, latency_ms, score, has_anomaly)

        self.finished.emit()
        print("🔴 OAK camera stopped.")

    def stop(self):
        self._running = False
