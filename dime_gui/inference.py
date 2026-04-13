# =========================================================
# inference.py — Shared inference logic and anomaly debouncing
# =========================================================

import cv2
import time

from config import ROI_MARGIN, SHOW_MARGIN


def run_inference(detector, frame, roi_coords, roi_margin=ROI_MARGIN, show_margin=SHOW_MARGIN):
    """Run the detector on a frame and draw bounding boxes. Returns (frame, latency_ms, score, has_anomaly)."""
    start = time.perf_counter()
    xmin, ymin, xmax, ymax = roi_coords
    ixmin = xmin + roi_margin
    iymin = ymin + roi_margin
    ixmax = xmax - roi_margin
    iymax = ymax - roi_margin

    result     = detector.process_frame(frame.copy())
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
