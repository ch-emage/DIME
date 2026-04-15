# =========================================================
# inference.py — Shared inference logic and anomaly debouncing
# =========================================================

import cv2
import os
import time
import datetime
import subprocess
from collections import deque

from config import (
    ROI_MARGIN, SHOW_MARGIN, MIN_ANOMALY_AREA,
    CLIP_ENABLED, CLIP_SECONDS_BEFORE, CLIP_SECONDS_AFTER, CLIP_OUTPUT_DIR,
    FULL_VIDEO_ENABLED, FULL_VIDEO_SAVE_RAW, FULL_VIDEO_SAVE_PROCESSED,
    FULL_VIDEO_OUTPUT_DIR,
)


def run_inference(detector, frame, roi_coords, roi_margin=ROI_MARGIN, show_margin=SHOW_MARGIN):
    """Run the detector on a frame and draw bounding boxes. Returns (frame, latency_ms, score, has_anomaly)."""
    start = time.perf_counter()
    xmin, ymin, xmax, ymax = roi_coords
    ixmin = xmin + roi_margin
    iymin = ymin + roi_margin
    ixmax = xmax - roi_margin
    iymax = ymax - roi_margin

    try:
        result = detector.process_frame(frame.copy())
    except Exception as e:
        print(f"⚠  Inference error: {e}")
        return frame, 0.0, 0.0, False
    latency_ms = (time.perf_counter() - start) * 1000

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    if roi_margin > 0 and show_margin:
        cv2.rectangle(frame, (ixmin, iymin), (ixmax, iymax), (0, 180, 80), 1)

    has_anomaly = False
    for a in result["anomaly_areas"]:
        x, y, w, h = map(int, a["bbox"])
        if roi_margin > 0:
            cx, cy = x + w // 2, y + h // 2
            if cx < ixmin or cy < iymin or cx > ixmax or cy > iymax:
                continue
        if w * h < MIN_ANOMALY_AREA:  # filter out tiny detections that are likely noise
            continue
        has_anomaly = True
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label, font = "anomaly", cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 1)
        cv2.rectangle(frame, (x, y - th - 6), (x + tw, y), (0, 0, 255), -1)
        cv2.putText(frame, label, (x, y - 3), font, 0.6, (255, 255, 255), 1)

    return frame, latency_ms, result["anomaly_score"], has_anomaly


class AnomalyDebouncer:
    """Prevents flicker by requiring N consecutive anomaly frames before triggering,
    and holding the alert for M frames after the anomaly clears."""

    def __init__(self, confirm_frames: int = 5, hold_frames: int = 15):
        self.confirm_frames = confirm_frames
        self.hold_frames    = hold_frames
        self._consec        = 0
        self._hold_counter  = 0
        self._confirmed     = False

    def update(self, has_anomaly: bool) -> bool:
        if has_anomaly:
            self._consec += 1
            self._hold_counter = self.hold_frames
            if self._consec >= self.confirm_frames:
                self._confirmed = True
        else:
            self._consec = 0
            if self._hold_counter > 0:
                self._hold_counter -= 1
            else:
                self._confirmed = False
        return self._confirmed

    def reset(self):
        self._consec = self._hold_counter = 0
        self._confirmed = False


class AnomalyClipRecorder:
    """Records anomaly clips (raw + annotated) using a rolling frame buffer
    and ffmpeg subprocesses. Keeps CLIP_SECONDS_BEFORE of footage before the
    anomaly is confirmed and CLIP_SECONDS_AFTER after it clears."""

    def __init__(self, fps, width, height, confirm_frames, hold_frames):
        self.fps    = max(1, fps)
        self.width  = width
        self.height = height

        buf_size = max(1, int(self.fps * CLIP_SECONDS_BEFORE))
        self._raw_buf = deque(maxlen=buf_size)
        self._ann_buf = deque(maxlen=buf_size)

        self._debouncer   = AnomalyDebouncer(confirm_frames, hold_frames)
        self._recording   = False
        self._post_frames = 0
        self._post_limit  = max(1, int(self.fps * CLIP_SECONDS_AFTER))

        self._raw_proc = None
        self._ann_proc = None
        self._clip_path_raw = None
        self._clip_path_ann = None

        os.makedirs(CLIP_OUTPUT_DIR, exist_ok=True)

    # ── public API ───────────────────────────────────────

    def feed(self, raw_frame, annotated_frame, has_anomaly):
        """Feed a frame pair. Returns (raw_path, ann_path) when a clip
        finishes saving, otherwise None."""
        if not CLIP_ENABLED:
            return None

        confirmed = self._debouncer.update(has_anomaly)

        # Always buffer (overwrites oldest when full)
        self._raw_buf.append(raw_frame)
        self._ann_buf.append(annotated_frame)

        # Transition: not recording → confirmed → start clip
        if confirmed and not self._recording:
            self._start_recording()

        # While recording, write every frame
        if self._recording:
            self._write_frame(raw_frame, annotated_frame)
            if not confirmed:
                self._post_frames += 1
                if self._post_frames >= self._post_limit:
                    return self._stop_recording()
            else:
                self._post_frames = 0

        return None

    def flush(self):
        """Call when the stream ends to close any in-progress clip."""
        if self._recording:
            return self._stop_recording()
        return None

    # ── internals ────────────────────────────────────────

    def _start_recording(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_dir = os.path.join(CLIP_OUTPUT_DIR, f"anomaly_clips_{ts}")
        raw_dir  = os.path.join(clip_dir, "raw")
        proc_dir = os.path.join(clip_dir, "processed")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(proc_dir, exist_ok=True)

        self._clip_dir      = clip_dir
        self._clip_path_raw = os.path.join(raw_dir, f"anomaly_{ts}.mp4")
        self._clip_path_ann = os.path.join(proc_dir, f"anomaly_{ts}.mp4")

        self._raw_proc = self._open_ffmpeg(self._clip_path_raw)
        self._ann_proc = self._open_ffmpeg(self._clip_path_ann)

        # Dump the "before" buffer into the pipes
        try:
            for raw, ann in zip(self._raw_buf, self._ann_buf):
                self._raw_proc.stdin.write(raw.tobytes())
                self._ann_proc.stdin.write(ann.tobytes())
        except (BrokenPipeError, OSError) as e:
            print(f"⚠  Could not write buffered frames to ffmpeg: {e}")
            self._raw_buf.clear()
            self._ann_buf.clear()
            self._stop_recording()
            return

        self._raw_buf.clear()
        self._ann_buf.clear()

        self._recording   = True
        self._post_frames = 0

    def _open_ffmpeg(self, path):
        return subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(int(self.fps)),
                "-i", "pipe:0",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _write_frame(self, raw_frame, annotated_frame):
        try:
            self._raw_proc.stdin.write(raw_frame.tobytes())
            self._ann_proc.stdin.write(annotated_frame.tobytes())
        except (BrokenPipeError, OSError):
            self._stop_recording()

    def _stop_recording(self):
        paths = (self._clip_path_raw, self._clip_path_ann)
        for proc in (self._raw_proc, self._ann_proc):
            if proc is None:
                continue
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass
        self._raw_proc = self._ann_proc = None
        self._recording = False
        self._post_frames = 0
        print(f"💾 Anomaly clip saved → {self._clip_dir}\n   Raw:       {paths[0]}\n   Processed: {paths[1]}")
        return paths


class FullVideoRecorder:
    """Records the entire inference session to disk. Depending on config,
    saves the raw feed, the annotated feed, or both — each in its own
    subdirectory (raw/ and processed/) under a timestamped session folder."""

    def __init__(self, fps, width, height):
        self.fps    = max(1, int(fps))
        self.width  = width
        self.height = height

        self._raw_proc = None
        self._ann_proc = None
        self._raw_path = None
        self._ann_path = None
        self._session_dir = None
        self._started = False

        self._save_raw = FULL_VIDEO_SAVE_RAW
        self._save_ann = FULL_VIDEO_SAVE_PROCESSED

    @property
    def enabled(self) -> bool:
        return FULL_VIDEO_ENABLED and (self._save_raw or self._save_ann)

    def feed(self, raw_frame, annotated_frame):
        if not self.enabled:
            return
        if not self._started:
            self._start()
        if self._raw_proc is not None:
            try:
                self._raw_proc.stdin.write(raw_frame.tobytes())
            except (BrokenPipeError, OSError):
                self._raw_proc = None
        if self._ann_proc is not None:
            try:
                self._ann_proc.stdin.write(annotated_frame.tobytes())
            except (BrokenPipeError, OSError):
                self._ann_proc = None

    def flush(self):
        if not self._started:
            return None
        paths = (self._raw_path if self._save_raw else None,
                 self._ann_path if self._save_ann else None)
        for proc in (self._raw_proc, self._ann_proc):
            if proc is None:
                continue
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass
        self._raw_proc = self._ann_proc = None
        self._started = False
        msg = [f"💾 Full video saved → {self._session_dir}"]
        if paths[0]:
            msg.append(f"   Raw:       {paths[0]}")
        if paths[1]:
            msg.append(f"   Processed: {paths[1]}")
        print("\n".join(msg))
        return paths

    def _start(self):
        os.makedirs(FULL_VIDEO_OUTPUT_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = os.path.join(FULL_VIDEO_OUTPUT_DIR, f"session_{ts}")
        os.makedirs(self._session_dir, exist_ok=True)

        if self._save_raw:
            raw_dir = os.path.join(self._session_dir, "raw")
            os.makedirs(raw_dir, exist_ok=True)
            self._raw_path = os.path.join(raw_dir, f"full_{ts}.mp4")
            self._raw_proc = self._open_ffmpeg(self._raw_path)
        if self._save_ann:
            proc_dir = os.path.join(self._session_dir, "processed")
            os.makedirs(proc_dir, exist_ok=True)
            self._ann_path = os.path.join(proc_dir, f"full_{ts}.mp4")
            self._ann_proc = self._open_ffmpeg(self._ann_path)
        self._started = True

    def _open_ffmpeg(self, path):
        return subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(int(self.fps)),
                "-i", "pipe:0",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
