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

# Set up logger
logger = logging.getLogger(__name__)

_CLASSNAMES = ["dime"]

IMAGENET_MEAN = [0.601, 0.601, 0.601]
IMAGENET_STD = [0.340, 0.340, 0.340]

class ObjectDetector:
    """Simple object detector for ROI selection"""
    def __init__(self):
        try:
            # Try different model paths
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
                # Download model if not found
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
            # Convert BGR to RGB (YOLO expects RGB)
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
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': cls_id,
                            'class_name': "object"
                        }
                        detections.append(detection)
            
            logger.info(f"Detected {len(detections)} objects")
            return detections
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []

def select_roi_and_object(video_path, enable_object_detection=False):
    """Enhanced ROI selection with object detection option"""
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
    
    # Initialize object detector if enabled
    detector = None
    if enable_object_detection:
        detector = ObjectDetector()
        if detector.model is None:
            print("Object detection disabled due to model loading failure")
            enable_object_detection = False
    
    while True:
        display_frame = original_frame.copy()
        
        # Show current selections
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
            # Select ROI region
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
            # Detect and select object
            detections = detector.detect_objects(original_frame, confidence_threshold=0.25)
            if not detections:
                print("No objects detected in frame")
                continue
            
            # Show detections
            detection_frame = original_frame.copy()
            print("\nDetected objects:")
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(detection_frame, f"{i}: {det['class_name']} ({det['confidence']:.2f})", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                print(f"{i}: {det['class_name']} (confidence: {det['confidence']:.2f})")
            
            cv2.imshow("Detected Objects", detection_frame)
            cv2.waitKey(3000)  # Show for 3 seconds
            cv2.destroyWindow("Detected Objects")
            
            # Let user select object
            try:
                choice = int(input(f"Select object (0-{len(detections)-1}), or -1 to cancel: "))
                if 0 <= choice < len(detections):
                    selected_det = detections[choice]
                    x1, y1, x2, y2 = selected_det['bbox']
                    # Expand bbox slightly
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
        
        elif key == 13 or key == 10:  # Enter key
            break
        
        elif key == ord('q'):
            cv2.destroyAllWindows()
            vidcap.release()
            return None, None, None
    
    cv2.destroyAllWindows()
    vidcap.release()
    
    return roi, object_roi, target_object_class

def extract_frames_with_object_tracking(video_path, output_dir, frame_interval=20, roi=None, 
                                       object_roi=None, target_object_class=None, roi_save_path=None):
    """Enhanced frame extraction with object tracking"""
    os.makedirs(output_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    
    # Create ROI save directory if it doesn't exist
    if roi_save_path:
        os.makedirs(roi_save_path, exist_ok=True)
    
    
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Count frames manually if total_frames is not available
        success, image = vidcap.read()
        total_frames = 0
        while success:
            total_frames += 1
            success, image = vidcap.read()
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    pbar = tqdm(total=total_frames, desc=f"Extracting {os.path.basename(video_path)}", unit='frame')
    
    # Initialize object tracker if object ROI is provided
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
    
    # Track object ROI across frames
    tracked_object_rois = []
    
    while success:
        if count % frame_interval == 0:
            current_roi = roi
            current_object_roi = object_roi
            
            # Update object tracker
            if tracker and object_roi:
                if not tracker_initialized:
                    # Initialize tracker with object ROI
                    bbox = (object_roi['x'], object_roi['y'], object_roi['w'], object_roi['h'])
                    success_init = tracker.init(image, bbox)
                    if success_init:
                        tracker_initialized = True
                        logger.info("Object tracker initialized")
                else:
                    # Update tracker
                    success_track, bbox = tracker.update(image)
                    if success_track:
                        x, y, w, h = [int(v) for v in bbox]
                        current_object_roi = {"x": x, "y": y, "w": w, "h": h}
                        tracked_object_rois.append(current_object_roi)
            
            # Apply ROI cropping if specified
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
            
            # Save frame
            frame_file = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_file, save_image)
            
            # Save object ROI info if available
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
    
    # Save ROI and object tracking information
    if roi_save_path:
        os.makedirs(roi_save_path, exist_ok=True)
        
        # Save main ROI
        if roi:
            roi_file = os.path.join(roi_save_path, f"{os.path.splitext(os.path.basename(video_path))[0]}_roi.json")
            with open(roi_file, 'w') as f:
                json.dump(roi, f)
        
        # Save object tracking info
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

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

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
        frame_interval=20,  # This will now be properly used
        model_result_dir=None,
        sequence_length=16,
        motion_config=None,
        sequence_config=None,
        object_detection_config=None,
        roi_selection_mode=False,
        enable_object_detection=False,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname else _CLASSNAMES
        self.train_val_split = train_val_split
        self.seed = seed
        self.frame_interval = frame_interval  # Store frame interval
        self.model_result_dir = model_result_dir
        self.sequence_length = sequence_length
        self.motion_config = motion_config or {}
        self.sequence_config = sequence_config or {}
        self.object_detection_config = object_detection_config or {}
        self.roi_selection_mode = roi_selection_mode
        self.enable_object_detection = enable_object_detection
        
        # ROI and object detection storage
        self.roi_data = {}
        self.object_tracking_data = {}
        
        if video_to_frames:
            self._process_videos_to_frames()

        self.imgpaths_per_class, self.data_to_iterate = self._get_image_data()

        if len(self.data_to_iterate) == 0:
            raise ValueError(f"No images found in {self.source} for class {classname} and split {split}")
        
        # _dynamic = (resize == "auto" or resize in (None, -1))
        # if _dynamic:
        #     from PIL import Image
        #     with Image.open(self.image_paths[0]) as _im0:
        #         w0, h0 = _im0.size
        #     effective_resize = (h0, w0)   # (H, W)
        # else:
        #     effective_resize = (resize, resize) if isinstance(resize, int) else tuple(resize)

        _dynamic = (resize == "auto" or resize in (None, -1))
        if _dynamic:
            # Pick the first real image path from data_to_iterate: (classname, anomaly, image_path, mask_path)
            first_img = None
            for _, _, img_path, _ in self.data_to_iterate:
                if os.path.exists(img_path):
                    first_img = img_path
                    break
            if not first_img:
                raise RuntimeError(f"No readable images found to infer (H,W) for classname={classname}, split={split}")

            with Image.open(first_img) as _im0:
                w0, h0 = _im0.size  # PIL returns (W,H)
            effective_resize = (h0, w0)   # (H, W)
        else:
            effective_resize = (resize, resize) if isinstance(resize, int) else tuple(resize)


        self.transform_img = transforms.Compose([
            transforms.Resize((effective_resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize((effective_resize)),
            transforms.ToTensor(),
        ])

        self.imagesize = (3, imagesize, imagesize)

    def _process_videos_to_frames(self):
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
                        frames_dir = os.path.join(anomaly_dir, f"{os.path.splitext(video_file)[0]}_frames")
                        
                        if not os.path.exists(frames_dir):
                            roi = None
                            object_roi = None
                            target_object_class = None
                            
                            # Enhanced ROI selection with object detection
                            if self.roi_selection_mode:
                                roi, object_roi, target_object_class = select_roi_and_object(
                                    video_path, 
                                    enable_object_detection=self.enable_object_detection
                                )
                                
                                # Store ROI and object data for later use
                                video_key = f"{classname}_{anomaly_type}_{os.path.splitext(video_file)[0]}"
                                if roi:
                                    self.roi_data[video_key] = roi
                                if object_roi and target_object_class:
                                    self.object_tracking_data[video_key] = {
                                        "roi": object_roi,
                                        "class": target_object_class
                                    }
                            
                            # Extract frames with proper frame interval
                            extract_frames_with_object_tracking(
                                video_path,
                                frames_dir,
                                frame_interval=self.frame_interval,  # Use stored frame interval
                                roi=roi,
                                object_roi=object_roi,
                                target_object_class=target_object_class,
                                roi_save_path=self.model_result_dir
                            )

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
                # Get direct images
                image_paths.extend([
                    os.path.join(good_dir, f) 
                    for f in os.listdir(good_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                ])
                
                # Get frame directories
                frame_dirs = [
                    d for d in os.listdir(good_dir) 
                    if d.endswith('_frames') and os.path.isdir(os.path.join(good_dir, d))
                ]
                
                for frame_dir in frame_dirs:
                    frame_path = os.path.join(good_dir, frame_dir)
                    frame_images = [
                        os.path.join(frame_path, f) 
                        for f in os.listdir(frame_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                    ]
                    # Sort frame images by name to maintain order
                    frame_images.sort()
                    image_paths.extend(frame_images)
                
                if not image_paths:
                    raise ValueError(f"No training images found in {good_dir}")
                
                # Train/val split
                if self.train_val_split < 1.0:
                    train_paths, val_paths = train_test_split(
                        image_paths,
                        test_size=1.0 - self.train_val_split,
                        random_state=self.seed
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
                    # Get direct images
                    image_paths.extend([
                        os.path.join(anomaly_dir, f) 
                        for f in os.listdir(anomaly_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                    ])
                    
                    # Get frame directories
                    frame_dirs = [
                        d for d in os.listdir(anomaly_dir) 
                        if d.endswith('_frames') and os.path.isdir(os.path.join(anomaly_dir, d))
                    ]
                    
                    for frame_dir in frame_dirs:
                        frame_path = os.path.join(anomaly_dir, frame_dir)
                        frame_images = [
                            os.path.join(frame_path, f) 
                            for f in os.listdir(frame_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                        ]
                        # Sort frame images by name to maintain order
                        frame_images.sort()
                        image_paths.extend(frame_images)
                    
                    imgpaths_per_class[classname][anomaly_type] = image_paths
                    
                    # Handle masks
                    if anomaly_type != "good":
                        mask_dir = os.path.join(test_dir, anomaly_type, "ground_truth")
                        if os.path.exists(mask_dir):
                            mask_paths = [
                                os.path.join(mask_dir, f) 
                                for f in os.listdir(mask_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                            ]
                            mask_paths.sort()  # Sort masks too
                            maskpaths_per_class[classname][anomaly_type] = mask_paths
                        else:
                            maskpaths_per_class[classname][anomaly_type] = [None] * len(image_paths)
                    else:
                        maskpaths_per_class[classname][anomaly_type] = [None] * len(image_paths)

        # Build data iteration list
        data_to_iterate = []
        for classname in imgpaths_per_class:
            for anomaly_type in imgpaths_per_class[classname]:
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly_type]):
                    mask_path = (
                        maskpaths_per_class[classname][anomaly_type][i] 
                        if maskpaths_per_class[classname][anomaly_type] 
                        else None
                    )
                    data_to_iterate.append((classname, anomaly_type, image_path, mask_path))

        return imgpaths_per_class, data_to_iterate

    def get_object_roi_for_image(self, image_path):
        """Get object ROI information for a specific image"""
        # Extract video info from path
        path_parts = image_path.split(os.sep)
        try:
            # Find frame directory
            frame_dir_idx = None
            for i, part in enumerate(path_parts):
                if part.endswith('_frames'):
                    frame_dir_idx = i
                    break
            
            if frame_dir_idx is not None:
                video_name = path_parts[frame_dir_idx].replace('_frames', '')
                anomaly_type = path_parts[frame_dir_idx - 1]
                classname = path_parts[frame_dir_idx - 3]  # Assuming structure: class/train|test/anomaly_type/video_frames
                
                video_key = f"{classname}_{anomaly_type}_{video_name}"
                
                # Check if we have object tracking data
                if video_key in self.object_tracking_data:
                    return self.object_tracking_data[video_key]
                
                # Try to load from saved JSON files
                if self.model_result_dir:
                    frame_name = os.path.splitext(os.path.basename(image_path))[0]
                    roi_file = os.path.join(self.model_result_dir, f"{frame_name}_object_roi.json")
                    if os.path.exists(roi_file):
                        with open(roi_file, 'r') as f:
                            return json.load(f)
        except:
            pass
        
        return None

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        # Get sequence of frames
        start_idx = max(0, idx - self.sequence_length + 1)
        seq_frames = []
        for i in range(start_idx, idx + 1):
            _, _, seq_image_path, _ = self.data_to_iterate[i]
            seq_image = Image.open(seq_image_path).convert("RGB")
            seq_frames.append(self.transform_img(seq_image))
        
        # Pad at beginning if needed
        if len(seq_frames) < self.sequence_length:
            padding = [torch.zeros_like(seq_frames[0]) 
                    for _ in range(self.sequence_length - len(seq_frames))]
            seq_frames = padding + seq_frames
        
        sequence = torch.stack(seq_frames)

        # Get object ROI information
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
        
        # Add object detection information if available
        if object_info:
            result["object_roi"] = object_info.get("object_roi", object_info.get("roi"))
            result["target_object_class"] = object_info.get("target_class", object_info.get("class"))
        
        return result

    def __len__(self):
        return len(self.data_to_iterate)

