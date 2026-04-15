# =========================================================
# threads/inference_threads.py — QThreads that run the DIME model
# =========================================================

import os
import time
import cv2
import depthai as dai

from PySide6.QtCore import QThread, Signal

from config import CLIP_ENABLED, FULL_VIDEO_ENABLED
from inference import run_inference, AnomalyClipRecorder, FullVideoRecorder


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
    clipSaved      = Signal(str, str)          # (raw_path, annotated_path)
    finished       = Signal()
    connectionLost = Signal(str)

    def __init__(self, detector, roi_coords, source, is_rtsp=False,
                 start_frame=0, confirm_frames=5, hold_frames=5):
        super().__init__()
        self.detector        = detector
        self.roi_coords      = roi_coords
        self.source          = source
        self.is_rtsp         = is_rtsp
        self.start_frame     = start_frame
        self.confirm_frames  = confirm_frames
        self.hold_frames     = hold_frames
        self._running        = True
        self.last_frame_pos  = 0

    def run(self):
        cap = None
        recorder = None
        full_recorder = None
        try:
            if self.is_rtsp:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                cap = cv2.VideoCapture(self.source)

            if not cap.isOpened():
                if self.is_rtsp:
                    self.connectionLost.emit(f"Could not connect to: {self.source}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if self.is_rtsp:
                print(f"📹 RTSP stream: {w}x{h} @ {fps:.1f} FPS")
            elif self.start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            if CLIP_ENABLED:
                recorder = AnomalyClipRecorder(
                    fps, w, h, self.confirm_frames, self.hold_frames
                )

            if FULL_VIDEO_ENABLED:
                full_recorder = FullVideoRecorder(fps, w, h)

            frame_interval = 1.0 / fps   # target seconds per frame

            failures = 0
            while self._running:
                frame_start = time.perf_counter()

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

                raw_frame = frame.copy()
                frame, latency_ms, score, has_anomaly = run_inference(
                    self.detector, frame, self.roi_coords
                )
                self.frameProcessed.emit(frame, latency_ms, score, has_anomaly)

                if recorder:
                    result = recorder.feed(raw_frame, frame, has_anomaly)
                    if result:
                        self.clipSaved.emit(result[0], result[1])

                if full_recorder:
                    full_recorder.feed(raw_frame, frame)

                if not self.is_rtsp:
                    elapsed = time.perf_counter() - frame_start
                    remaining_ms = int((frame_interval - elapsed) * 1000)
                    if remaining_ms > 0:
                        QThread.msleep(remaining_ms)
        except Exception as e:
            print(f"⚠  VideoInferenceThread error: {e}")
            if self.is_rtsp:
                self.connectionLost.emit(f"Stream error: {e}")
        finally:
            if recorder:
                try:
                    result = recorder.flush()
                    if result:
                        self.clipSaved.emit(result[0], result[1])
                except Exception as e:
                    print(f"⚠  Clip flush error: {e}")
            if full_recorder:
                try:
                    full_recorder.flush()
                except Exception as e:
                    print(f"⚠  Full video flush error: {e}")
            if cap is not None:
                cap.release()
            self.finished.emit()

    def stop(self):
        self._running = False


class OAKInferenceThread(QThread):
    """OAK thread WITH anomaly inference — used in Live Camera tab."""
    frameProcessed = Signal(object, float, float, bool)
    clipSaved      = Signal(str, str)          # (raw_path, annotated_path)
    finished       = Signal()
    connectionLost = Signal(str)

    def __init__(self, detector, roi_coords, resolution=(1920, 1080),
                 confirm_frames=3, hold_frames=3):
        super().__init__()
        self.detector       = detector
        self.roi_coords     = roi_coords
        self.resolution     = resolution
        self.confirm_frames = confirm_frames
        self.hold_frames    = hold_frames
        self._running       = True

    def run(self):
        pipeline = None
        queue = None
        recorder = None
        full_recorder = None
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
                print(f"✅ OAK camera: {self.resolution[0]}x{self.resolution[1]}")
            except Exception as e:
                print(f"❌ Failed to open OAK camera: {e}")
                self.connectionLost.emit(str(e))
                return

            if CLIP_ENABLED:
                recorder = AnomalyClipRecorder(
                    30, self.resolution[0], self.resolution[1],
                    self.confirm_frames, self.hold_frames,
                )

            if FULL_VIDEO_ENABLED:
                full_recorder = FullVideoRecorder(
                    30, self.resolution[0], self.resolution[1]
                )

            while self._running and pipeline.isRunning():
                frame = queue.get().getCvFrame()
                raw_frame = frame.copy()
                frame, latency_ms, score, has_anomaly = run_inference(
                    self.detector, frame, self.roi_coords
                )
                self.frameProcessed.emit(frame, latency_ms, score, has_anomaly)

                if recorder:
                    result = recorder.feed(raw_frame, frame, has_anomaly)
                    if result:
                        self.clipSaved.emit(result[0], result[1])

                if full_recorder:
                    full_recorder.feed(raw_frame, frame)
        except Exception as e:
            print(f"⚠  OAKInferenceThread error: {e}")
            self.connectionLost.emit(f"OAK stream error: {e}")
        finally:
            if recorder:
                try:
                    result = recorder.flush()
                    if result:
                        self.clipSaved.emit(result[0], result[1])
                except Exception as e:
                    print(f"⚠  Clip flush error: {e}")
            if full_recorder:
                try:
                    full_recorder.flush()
                except Exception as e:
                    print(f"⚠  Full video flush error: {e}")
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            self.finished.emit()
            print("🔴 OAK camera stopped.")

    def stop(self):
        self._running = False
