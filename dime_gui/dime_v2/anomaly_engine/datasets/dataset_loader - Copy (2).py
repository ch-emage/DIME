import os
import json
from enum import Enum
import cv2
import PIL
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from ultralytics import YOLO
import logging
import shutil

# -----------------------------
# Logger
# -----------------------------
logger = logging.getLogger(__name__)

# -----------------------------
# Constants
# -----------------------------
_CLASSNAMES = ["dime"]
IMAGENET_MEAN = [0.601, 0.601, 0.601]
IMAGENET_STD = [0.340, 0.340, 0.340]

# -----------------------------
# Enums
# -----------------------------
class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

# -----------------------------
# YOLO-based Object Detector (kept as-is; unchanged logic)
# -----------------------------
class ObjectDetector:
    """Simple object detector for ROI selection"""
    def __init__(self):
        try:
            model_paths = [
                'yolov8n.pt',
                os.path.expanduser('~/.ultralytics/yolov8n.pt'),
                'models/yolov8n.pt'
            ]
            for model_path in model_paths:
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                    logger.info(f"Loaded YOLOv8 model from {model_path}")
                    break
            else:
                self.model = YOLO('yolov8n.pt')
                logger.info("Downloaded and loaded YOLOv8 Nano model")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8: {e}")
            self.model = None
    
    def detect_objects(self, frame, confidence_threshold=0.25):
        """Detect objects in frame with lower confidence threshold"""
        if self.model is None:
            return []
        try:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            results = self.model(frame_rgb, verbose=False, conf=confidence_threshold)
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf.item()
                        cls_id = int(box.cls.item())
                        cls_name = self.model.names[cls_id]
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': cls_id,
                            'class_name': "object"
                        })
            logger.info(f"Detected {len(detections)} objects")
            return detections
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []

# -----------------------------
# Original single-ROI/object selection (kept; unchanged)
# -----------------------------
def select_roi_and_object(video_path, enable_object_detection=False):
    """(Unchanged) Single ROI / object-assisted selection"""
    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()
    if not success:
        vidcap.release()
        raise ValueError(f"Cannot read video: {video_path}")
    
    original_frame = frame.copy()
    roi = None
    object_roi = None
    target_object_class = None
    
    print(f"\nProcessing video: {os.path.basename(video_path)}")
    print("Options:")
    print("1. Press 'r' to select ROI region")
    print("2. Press 'o' to detect and select object (if object detection enabled)")
    print("3. Press 'Enter' to use full frame")
    print("4. Press 'q' to quit")
    
    window_name = f"ROI/Object Selection - {os.path.basename(video_path)}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    detector = None
    if enable_object_detection:
        detector = ObjectDetector()
        if detector.model is None:
            print("Object detection disabled due to model loading failure")
            enable_object_detection = False
    
    while True:
        display_frame = original_frame.copy()
        if roi:
            cv2.rectangle(display_frame, (roi['x'], roi['y']),
                          (roi['x'] + roi['w'], roi['y'] + roi['h']), (0, 255, 255), 2)
            cv2.putText(display_frame, "ROI Selected", (roi['x'], roi['y'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if object_roi:
            cv2.rectangle(display_frame, (object_roi['x'], object_roi['y']),
                          (object_roi['x'] + object_roi['w'], object_roi['y'] + object_roi['h']), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Object: {target_object_class}",
                        (object_roi['x'], object_roi['y'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            cv2.destroyWindow(window_name)
            roi_coords = cv2.selectROI(f"Select ROI - {os.path.basename(video_path)}",
                                       original_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(f"Select ROI - {os.path.basename(video_path)}")
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            if roi_coords != (0, 0, 0, 0):
                roi = {"x": int(roi_coords[0]), "y": int(roi_coords[1]),
                       "w": int(roi_coords[2]), "h": int(roi_coords[3])}
                print(f"ROI selected: {roi}")
        
        elif key == ord('o') and enable_object_detection and detector:
            detections = detector.detect_objects(original_frame, confidence_threshold=0.25)
            if not detections:
                print("No objects detected in frame")
                continue
            detection_frame = original_frame.copy()
            print("\nDetected objects:")
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(detection_frame, f"{i}: {det['class_name']} ({det['confidence']:.2f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                print(f"{i}: {det['class_name']} (confidence: {det['confidence']:.2f})")
            cv2.imshow("Detected Objects", detection_frame)
            cv2.waitKey(3000)
            cv2.destroyWindow("Detected Objects")
            try:
                choice = int(input(f"Select object (0-{len(detections)-1}), or -1 to cancel: "))
                if 0 <= choice < len(detections):
                    selected_det = detections[choice]
                    x1, y1, x2, y2 = selected_det['bbox']
                    expansion_factor = 0.2
                    w, h = x2 - x1, y2 - y1
                    x1 = max(0, int(x1 - w * expansion_factor))
                    y1 = max(0, int(y1 - h * expansion_factor))
                    x2 = min(original_frame.shape[1], int(x2 + w * expansion_factor))
                    y2 = min(original_frame.shape[0], int(y2 + h * expansion_factor))
                    object_roi = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
                    target_object_class = selected_det['class_name']
                    print(f"Object ROI selected: {object_roi}, Class: {target_object_class}")
            except (ValueError, IndexError):
                print("Invalid selection")
        
        elif key == 13 or key == 10:  # Enter
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            vidcap.release()
            return None, None, None
    
    cv2.destroyAllWindows()
    vidcap.release()
    return roi, object_roi, target_object_class

# -----------------------------
# NEW: Multiple ROI selection in one session (added)
# -----------------------------
def select_multiple_rois_at_once(video_path, num_rois=3):
    """Prompt user to select multiple ROIs on the same frame with scaling & colored overlays."""
    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()
    if not success:
        vidcap.release()
        raise ValueError(f"Cannot read video: {video_path}")
    
    # Screen dims (best effort)
    try:
        import ctypes
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)
    except:
        screen_width, screen_height = 1920, 1080
    
    H, W = frame.shape[:2]
    scale_factor = 1.0
    if W > screen_width * 0.8 or H > screen_height * 0.8:
        scale_w = (screen_width * 0.8) / W
        scale_h = (screen_height * 0.8) / H
        scale_factor = min(scale_w, scale_h)
        disp_w, disp_h = int(W * scale_factor), int(H * scale_factor)
        frame_display = cv2.resize(frame, (disp_w, disp_h))
    else:
        frame_display = frame.copy()
    
    colors = [
        (0, 0, 255), (255, 0, 255), (0, 255, 0),
        (255, 255, 0), (255, 0, 0), (0, 165, 255)
    ]
    
    class State:
        def __init__(self):
            self.base = frame_display.copy()
            self.current = None
            self.rois = []
            self.drawing = False
            self.ix, self.iy = -1, -1
            self.needs_redraw = True
    
    state = State()
    window = f"Select {num_rois} ROIs - {os.path.basename(video_path)}"
    print(f"\n=== Selecting {num_rois} ROIs for {os.path.basename(video_path)} ===")
    print("ENTER: Confirm | 'c': Cancel current | 'q': Quit early")
    
    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.drawing = True
            state.ix, state.iy = x, y
            state.current = [x, y, 0, 0]
            state.needs_redraw = True
        elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
            state.current = [state.ix, state.iy, x - state.ix, y - state.iy]
            state.needs_redraw = True
        elif event == cv2.EVENT_LBUTTONUP:
            state.drawing = False
            if state.current and abs(state.current[2]) > 10 and abs(state.current[3]) > 10:
                x1, y1, x2, y2 = state.ix, state.iy, x, y
                state.current = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]
            else:
                state.current = None
            state.needs_redraw = True
    
    def draw():
        disp = state.base.copy()
        # existing
        for i, (x, y, w, h) in enumerate(state.rois):
            c = colors[i % len(colors)]
            cv2.rectangle(disp, (x, y), (x + w, y + h), c, 4)
            cv2.putText(disp, f"ROI {i+1}", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        # current
        if state.current is not None:
            x, y, w, h = state.current
            c = colors[len(state.rois) % len(colors)]
            cv2.rectangle(disp, (x, y), (x + w, y + h), c, 5 if state.drawing else 4)
        return disp
    
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, frame_display.shape[1], frame_display.shape[0])
    cv2.setMouseCallback(window, mouse_cb)
    
    for i in range(num_rois):
        state.current = None
        state.drawing = False
        state.needs_redraw = True
        while True:
            if state.needs_redraw:
                cv2.imshow(window, draw())
                state.needs_redraw = False
            key = cv2.waitKey(10) & 0xFF
            if key in (13, 10):  # Enter
                if state.current and state.current[2] > 10 and state.current[3] > 10:
                    state.rois.append(state.current)
                    print(f"ROI {i+1} = {state.current}")
                    state.current = None
                    state.needs_redraw = True
                    break
                else:
                    print("Draw a valid ROI (>=10x10)")
            elif key == ord('c'):
                state.current = None
                state.needs_redraw = True
            elif key == ord('q'):
                cv2.destroyWindow(window)
                vidcap.release()
                # scale back
                out = []
                for (x, y, w, h) in state.rois:
                    if scale_factor != 1.0:
                        out.append({"x": int(x / scale_factor), "y": int(y / scale_factor),
                                    "w": int(w / scale_factor), "h": int(h / scale_factor)})
                    else:
                        out.append({"x": x, "y": y, "w": w, "h": h})
                return out
        cv2.waitKey(250)
    
    cv2.destroyWindow(window)
    vidcap.release()
    # scale back
    out = []
    for (x, y, w, h) in state.rois:
        if scale_factor != 1.0:
            out.append({"x": int(x / scale_factor), "y": int(y / scale_factor),
                        "w": int(w / scale_factor), "h": int(h / scale_factor)})
        else:
            out.append({"x": x, "y": y, "w": w, "h": h})
    return out

# -----------------------------
# ORIGINAL: Frame extraction with tracking (kept, unchanged)
# -----------------------------
def extract_frames_with_object_tracking(video_path, output_dir, frame_interval=20, roi=None,
                                       object_roi=None, target_object_class=None, roi_save_path=None):
    """(Unchanged) Original tracking-based extractor retained for compatibility."""
    os.makedirs(output_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    if roi_save_path:
        os.makedirs(roi_save_path, exist_ok=True)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        success, image = vidcap.read()
        total_frames = 0
        while success:
            total_frames += 1
            success, image = vidcap.read()
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pbar = tqdm(total=total_frames, desc=f"Extracting {os.path.basename(video_path)}", unit='frame')

    tracker = None
    if object_roi and target_object_class:
        try:
            tracker = cv2.TrackerCSRT_create()
        except:
            try:
                tracker = cv2.legacy.TrackerCSRT_create()
            except:
                logger.warning("Could not create tracker")
                tracker = None

    count = 0
    frame_count = 0
    success, image = vidcap.read()
    tracker_initialized = False
    tracked_object_rois = []
    while success:
        if count % frame_interval == 0:
            current_roi = roi
            current_object_roi = object_roi
            if tracker and object_roi:
                if not tracker_initialized:
                    bbox = (object_roi['x'], object_roi['y'], object_roi['w'], object_roi['h'])
                    success_init = tracker.init(image, bbox)
                    if success_init:
                        tracker_initialized = True
                        logger.info("Object tracker initialized")
                else:
                    success_track, bbox = tracker.update(image)
                    if success_track:
                        x, y, w, h = [int(v) for v in bbox]
                        current_object_roi = {"x": x, "y": y, "w": w, "h": h}
                        tracked_object_rois.append(current_object_roi)
            if current_roi:
                x, y, w, h = current_roi["x"], current_roi["y"], current_roi["w"], current_roi["h"]
                cropped_image = image[y:y+h, x:x+w]
                if cropped_image.size == 0:
                    print(f"Warning: ROI cropping resulted in empty image for frame {frame_count}")
                    success, image = vidcap.read()
                    count += 1
                    pbar.update(1)
                    continue
                save_image = cropped_image
            else:
                save_image = image
            frame_file = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_file, save_image)
            if current_object_roi and roi_save_path:
                roi_info_file = os.path.join(roi_save_path, f"frame_{frame_count:04d}_object_roi.json")
                roi_info = {
                    "object_roi": current_object_roi,
                    "target_class": target_object_class,
                    "frame_number": frame_count
                }
                with open(roi_info_file, 'w') as f:
                    json.dump(roi_info, f)
            frame_count += 1
        success, image = vidcap.read()
        count += 1
        pbar.update(1)
    vidcap.release()
    pbar.close()
    if roi_save_path:
        os.makedirs(roi_save_path, exist_ok=True)
        if roi:
            roi_file = os.path.join(roi_save_path, f"{os.path.splitext(os.path.basename(video_path))[0]}_roi.json")
            with open(roi_file, 'w') as f:
                json.dump(roi, f)
        if object_roi and target_object_class:
            object_info = {
                "initial_object_roi": object_roi,
                "target_object_class": target_object_class,
                "tracked_rois": tracked_object_rois,
                "total_frames": frame_count
            }
            object_file = os.path.join(roi_save_path, f"{os.path.splitext(os.path.basename(video_path))[0]}_object_tracking.json")
            with open(object_file, 'w') as f:
                json.dump(object_info, f, indent=2)
    return frame_count

# -----------------------------
# NEW: Simple ROI extractor (per ROI index) with per-ROI folder naming
# -----------------------------
def extract_frames(video_path, output_dir, frame_interval=1, roi=None, roi_index=0):
    """Extract frames from video; optional static ROI crop; saves every N frames; per-ROI folder name."""
    os.makedirs(output_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        success, image = vidcap.read()
        total_frames = 0
        while success:
            total_frames += 1
            success, image = vidcap.read()
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pbar = tqdm(total=total_frames, desc=f"Extracting ROI {roi_index+1} - {os.path.basename(video_path)}", unit='frame')

    count = 0
    frame_count = 0
    success, image = vidcap.read()
    while success:
        if count % frame_interval == 0:
            if roi and all(k in roi for k in ("x", "y", "w", "h")):
                x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
                x = max(0, min(x, image.shape[1] - 1))
                y = max(0, min(y, image.shape[0] - 1))
                w = max(1, min(w, image.shape[1] - x))
                h = max(1, min(h, image.shape[0] - y))
                crop = image[y:y+h, x:x+w]
                if crop.size == 0:
                    print(f"Warning: empty crop for frame {frame_count}; saving full frame fallback")
                    save_img = image
                else:
                    save_img = crop
            else:
                save_img = image
            frame_file = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_file, save_img)
            frame_count += 1
        success, image = vidcap.read()
        count += 1
        pbar.update(1)
    vidcap.release()
    pbar.close()
    print(f"Extracted {frame_count} frames for ROI {roi_index} to {output_dir}")
    return frame_count

# -----------------------------
# NEW: Infer resize from saved ROI frames
# -----------------------------
def get_roi_dimensions_from_frames(frames_dir):
    """Return [H, W] from first saved frame in frames_dir, else None."""
    if not os.path.exists(frames_dir):
        return None
    frame_files = [f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.png')]
    if not frame_files:
        return None
    frame_files.sort()
    try:
        first_frame_path = os.path.join(frames_dir, frame_files[0])
        img = cv2.imread(first_frame_path)
        if img is not None:
            h, w = img.shape[:2]
            return [h, w]
    except Exception as e:
        print(f"Error reading frame dims: {e}")
    return None

# -----------------------------
# Dataset
# -----------------------------
class VideoAnomalyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        seed=0,
        augment=False,
        video_to_frames=True,
        frame_interval=20,   # original default preserved
        model_result_dir=None,
        sequence_length=16,
        motion_config=None,
        sequence_config=None,
        object_detection_config=None,
        roi_selection_mode=False,
        enable_object_detection=False,
        # NEW:
        current_roi_index=0,
        multiple_rois=None,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname else _CLASSNAMES
        self.train_val_split = train_val_split
        self.seed = seed
        self.frame_interval = frame_interval
        self.model_result_dir = model_result_dir
        self.sequence_length = sequence_length
        self.motion_config = motion_config or {}
        self.sequence_config = sequence_config or {}
        self.object_detection_config = object_detection_config or {}
        self.roi_selection_mode = roi_selection_mode
        self.enable_object_detection = enable_object_detection

        # NEW: multi-ROI parameters
        self.current_roi_index = int(current_roi_index)
        self.multiple_rois = multiple_rois or []

        # ROI/object-tracking data kept as-is (unused by multi-ROI flow, but preserved)
        self.roi_data = {}
        self.object_tracking_data = {}

        if video_to_frames:
            self._process_videos_to_frames()

        self.imgpaths_per_class, self.data_to_iterate = self._get_image_data()
        if len(self.data_to_iterate) == 0:
            raise ValueError(f"No images found in {self.source} for class {classname} and split {split}")

        # --- NEW: dynamic resize from ROI-specific frames, else fallback to original behavior ---
        dynamic_resize = self._get_dynamic_resize()
        if dynamic_resize:
            resize = dynamic_resize
            imagesize = dynamic_resize
            print(f"Using dynamic resize dimensions: {dynamic_resize} for ROI {self.current_roi_index}")
        else:
            print(f"Using default resize dimensions: {resize} for ROI {self.current_roi_index}")

        # If resize is an int (old style), keep old semantics; if it's [H,W], enforce that
        if isinstance(resize, (list, tuple)) and len(resize) == 2:
            self.transform_img = transforms.Compose([
                transforms.Resize((resize[0], resize[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            self.transform_mask = transforms.Compose([
                transforms.Resize((resize[0], resize[1])),
                transforms.ToTensor(),
            ])
            self.imagesize = (3, resize[0], resize[1])
        else:
            # Original aspect-preserving path
            self.transform_img = transforms.Compose([
                transforms.Resize((resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            self.transform_mask = transforms.Compose([
                transforms.Resize((resize)),
                transforms.ToTensor(),
            ])
            self.imagesize = (3, imagesize, imagesize)

    # --- NEW: look up dynamic [H,W] from ROI-specific frame folders for current ROI index ---
    def _get_dynamic_resize(self):
        for classname in self.classnames_to_use:
            class_dir = os.path.join(self.source, classname)
            for split_dir_name in ['train', 'test']:
                split_dir = os.path.join(class_dir, split_dir_name)
                if not os.path.exists(split_dir):
                    continue
                for anomaly_type in os.listdir(split_dir):
                    anomaly_dir = os.path.join(split_dir, anomaly_type)
                    if not os.path.isdir(anomaly_dir):
                        continue
                    frame_dirs = [
                        d for d in os.listdir(anomaly_dir)
                        if d.endswith(f'_frames_roi_{self.current_roi_index}')
                        and os.path.isdir(os.path.join(anomaly_dir, d))
                    ]
                    for frame_dir in frame_dirs:
                        dims = get_roi_dimensions_from_frames(os.path.join(anomaly_dir, frame_dir))
                        if dims:
                            return dims
        return None

    # --- MODIFIED: Process videos to frames per CURRENT ROI index with <video>_frames_roi_{index} ---
    def _process_videos_to_frames(self):
        print(f"Processing videos to frames for ROI index {self.current_roi_index}")
        for classname in self.classnames_to_use:
            class_dir = os.path.join(self.source, classname)
            for split in ['train', 'test']:
                split_dir = os.path.join(class_dir, split)
                if not os.path.exists(split_dir):
                    continue
                for anomaly_type in os.listdir(split_dir):
                    anomaly_dir = os.path.join(split_dir, anomaly_type)
                    if not os.path.isdir(anomaly_dir):
                        continue
                    video_files = [f for f in os.listdir(anomaly_dir)
                                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.MOV'))]
                    for video_file in tqdm(video_files, desc=f"Processing videos in {anomaly_type}"):
                        video_path = os.path.join(anomaly_dir, video_file)
                        frames_dir = os.path.join(
                            anomaly_dir,
                            f"{os.path.splitext(video_file)[0]}_frames_roi_{self.current_roi_index}"
                        )
                        if os.path.exists(frames_dir):
                            print(f"Frames exist for {video_file} ROI {self.current_roi_index}")
                            continue

                        # Decide ROI for this index:
                        roi = None
                        if self.multiple_rois and self.current_roi_index < len(self.multiple_rois):
                            roi = self.multiple_rois[self.current_roi_index]
                            print(f"Using provided ROI[{self.current_roi_index}]: {roi}")
                        elif self.roi_selection_mode:
                            # If interactive multi-ROI selection requested, collect them once per video
                            # and pick the current index if available; otherwise, fallback to full frame
                            try:
                                selected = select_multiple_rois_at_once(
                                    video_path, num_rois=max(1, len(self.multiple_rois) or 1)
                                )
                                if selected and self.current_roi_index < len(selected):
                                    roi = selected[self.current_roi_index]
                                    print(f"Selected ROI[{self.current_roi_index}] from UI: {roi}")
                                else:
                                    print(f"No ROI[{self.current_roi_index}] selected; using full frame for {video_file}")
                            except Exception as e:
                                print(f"Multi-ROI selection failed: {e}. Using full frame.")

                        # Extract frames for this ROI index (no tracking for multi-ROI path)
                        extract_frames(
                            video_path=video_path,
                            output_dir=frames_dir,
                            frame_interval=self.frame_interval,
                            roi=roi,
                            roi_index=self.current_roi_index
                        )

    # --- MODIFIED: Scan ROI-specific folders for current ROI index (train/test) ---
    def _get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}
        for classname in self.classnames_to_use:
            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            if self.split in [DatasetSplit.TRAIN, DatasetSplit.VAL]:
                train_dir = os.path.join(self.source, classname, "train")
                good_dir = os.path.join(train_dir, "good")
                if not os.path.exists(good_dir):
                    raise FileNotFoundError(f"Training directory not found: {good_dir}")

                image_paths = []

                # Prefer ROI-specific frames for current ROI index
                frame_dirs = [
                    d for d in os.listdir(good_dir)
                    if d.endswith(f'_frames_roi_{self.current_roi_index}')
                    and os.path.isdir(os.path.join(good_dir, d))
                ]
                for frame_dir in frame_dirs:
                    frame_path = os.path.join(good_dir, frame_dir)
                    frames = [
                        os.path.join(frame_path, f)
                        for f in os.listdir(frame_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                    ]
                    frames.sort()
                    image_paths.extend(frames)
                    print(f"Found {len(frames)} ROI {self.current_roi_index} images in {frame_dir}")

                # Fallback to original images if no ROI frames found
                if not image_paths:
                    originals = [
                        os.path.join(good_dir, f)
                        for f in os.listdir(good_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                    ]
                    originals.sort()
                    image_paths.extend(originals)
                    print(f"Found {len(originals)} original images in good directory")

                if not image_paths:
                    raise ValueError(f"No training images found in {good_dir} for ROI {self.current_roi_index}")

                if self.train_val_split < 1.0:
                    train_paths, val_paths = train_test_split(
                        image_paths, test_size=1.0 - self.train_val_split, random_state=self.seed
                    )
                    imgpaths_per_class[classname]["good"] = (
                        train_paths if self.split == DatasetSplit.TRAIN else val_paths
                    )
                else:
                    imgpaths_per_class[classname]["good"] = image_paths

                maskpaths_per_class[classname]["good"] = [None] * len(imgpaths_per_class[classname]["good"])

            elif self.split == DatasetSplit.TEST:
                test_dir = os.path.join(self.source, classname, "test")
                if not os.path.exists(test_dir):
                    raise FileNotFoundError(f"Test directory not found: {test_dir}")

                for anomaly_type in os.listdir(test_dir):
                    anomaly_dir = os.path.join(test_dir, anomaly_type)
                    if not os.path.isdir(anomaly_dir):
                        continue
                    image_paths = []

                    # ROI-specific frames first
                    frame_dirs = [
                        d for d in os.listdir(anomaly_dir)
                        if d.endswith(f'_frames_roi_{self.current_roi_index}')
                        and os.path.isdir(os.path.join(anomaly_dir, d))
                    ]
                    for frame_dir in frame_dirs:
                        frame_path = os.path.join(anomaly_dir, frame_dir)
                        frames = [
                            os.path.join(frame_path, f)
                            for f in os.listdir(frame_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                        ]
                        frames.sort()
                        image_paths.extend(frames)

                    # Include any direct images as well
                    originals = [
                        os.path.join(anomaly_dir, f)
                        for f in os.listdir(anomaly_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                    ]
                    originals.sort()
                    image_paths.extend(originals)

                    imgpaths_per_class[classname][anomaly_type] = image_paths

                    if anomaly_type != "good":
                        mask_dir = os.path.join(test_dir, anomaly_type, "ground_truth")
                        if os.path.exists(mask_dir):
                            mask_paths = [
                                os.path.join(mask_dir, f)
                                for f in os.listdir(mask_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                            ]
                            mask_paths.sort()
                            maskpaths_per_class[classname][anomaly_type] = mask_paths
                        else:
                            maskpaths_per_class[classname][anomaly_type] = [None] * len(image_paths)
                    else:
                        maskpaths_per_class[classname][anomaly_type] = [None] * len(image_paths)

        data_to_iterate = []
        for classname in imgpaths_per_class:
            for anomaly_type in imgpaths_per_class[classname]:
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly_type]):
                    mask_path = (
                        maskpaths_per_class[classname][anomaly_type][i]
                        if maskpaths_per_class[classname][anomaly_type] else None
                    )
                    data_to_iterate.append((classname, anomaly_type, image_path, mask_path))
        print(f"Total images found for ROI {self.current_roi_index}: {len(data_to_iterate)}")
        return imgpaths_per_class, data_to_iterate

    # --- (Kept) Object ROI lookup (still available; harmless if unused in multi-ROI flow) ---
    def get_object_roi_for_image(self, image_path):
        """Get object ROI information for a specific image (original JSON-based lookup)."""
        path_parts = image_path.split(os.sep)
        try:
            frame_dir_idx = None
            for i, part in enumerate(path_parts):
                if part.endswith('_frames'):
                    frame_dir_idx = i
                    break
            if frame_dir_idx is not None:
                video_name = path_parts[frame_dir_idx].replace('_frames', '')
                anomaly_type = path_parts[frame_dir_idx - 1]
                classname = path_parts[frame_dir_idx - 3]
                video_key = f"{classname}_{anomaly_type}_{video_name}"
                if video_key in self.object_tracking_data:
                    return self.object_tracking_data[video_key]
                if self.model_result_dir:
                    frame_name = os.path.splitext(os.path.basename(image_path))[0]
                    roi_file = os.path.join(self.model_result_dir, f"{frame_name}_object_roi.json")
                    if os.path.exists(roi_file):
                        with open(roi_file, 'r') as f:
                            return json.load(f)
        except:
            pass
        return None

    # --- __getitem__ / __len__ (unchanged except transforms shape handled above) ---
    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        start_idx = max(0, idx - self.sequence_length + 1)
        seq_frames = []
        for i in range(start_idx, idx + 1):
            _, _, seq_image_path, _ = self.data_to_iterate[i]
            seq_image = Image.open(seq_image_path).convert("RGB")
            seq_frames.append(self.transform_img(seq_image))
        if len(seq_frames) < self.sequence_length:
            padding = [torch.zeros_like(seq_frames[0])
                       for _ in range(self.sequence_length - len(seq_frames))]
            seq_frames = padding + seq_frames
        sequence = torch.stack(seq_frames)

        # (Kept) Optional object ROI info
        object_info = self.get_object_roi_for_image(image_path)

        result = {
            "image": image,
            "sequence": sequence,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
        }
        if object_info:
            result["object_roi"] = object_info.get("object_roi", object_info.get("roi"))
            result["target_object_class"] = object_info.get("target_class", object_info.get("class"))
        return result

    def __len__(self):
        return len(self.data_to_iterate)

# -----------------------------
# NEW: cleanup utility for ROI-specific frame folders
# -----------------------------
def cleanup_roi_frames(data_path, roi_index):
    """Clean up extracted frames for a specific ROI index (<video>_frames_roi_{roi_index})."""
    print(f"Cleaning up frames for ROI {roi_index}")
    for classname in os.listdir(data_path):
        class_dir = os.path.join(data_path, classname)
        for split in ['train', 'test']:
            split_dir = os.path.join(class_dir, split)
            if not os.path.exists(split_dir):
                continue
            for anomaly_type in os.listdir(split_dir):
                anomaly_dir = os.path.join(split_dir, anomaly_type)
                if not os.path.isdir(anomaly_dir):
                    continue
                frame_dirs = [
                    d for d in os.listdir(anomaly_dir)
                    if d.endswith(f'_frames_roi_{roi_index}')
                ]
                for frame_dir in frame_dirs:
                    frame_path = os.path.join(anomaly_dir, frame_dir)
                    try:
                        shutil.rmtree(frame_path)
                        print(f"Cleaned up: {frame_path}")
                    except Exception as e:
                        print(f"Error cleaning up {frame_path}: {e}")
