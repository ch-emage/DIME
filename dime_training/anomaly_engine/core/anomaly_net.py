"""AnomalyNet and defect detection methods with enhanced object detection."""
import logging
import os
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import pickle
import faiss
from pathlib import Path
from torch.cuda.amp import autocast
from ultralytics import YOLO

import anomaly_engine.models
import anomaly_engine.core.core_utils as anomaly_engine_core_utils
import anomaly_engine.selectors.feature_selector
import anomaly_engine.preprocessing.feature_processor
from .motion_utils import MotionProcessor  # New import
from anomaly_engine.core.feature_utils import FeatureSlicer as FeatureSlicer_obj
from .motion_utils import MotionSpeedAnalyzer
LOGGER = logging.getLogger(__name__)

class ObjectDetectionModule:
    """Enhanced object detection module with ROI tracking"""
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.tracker = None
        self.target_classes = set()
        
        if self.config.get("enable", False):
            self._load_model()
    
    def _load_model(self):
        """Load object detection model"""
        try:
            model_name = self.config.get("model", "yolov8n")
            self.model = YOLO(f'{model_name}.pt')
            LOGGER.info(f"Loaded {model_name} for object detection")
        except Exception as e:
            LOGGER.error(f"Failed to load object detection model: {e}")
            self.model = None
    
    def set_target_classes(self, classes):
        """Set target object classes to detect"""
        if isinstance(classes, str):
            classes = [classes]
        self.target_classes = set(classes)
        LOGGER.info(f"Target classes set to: {self.target_classes}")
    
    def detect_objects(self, frame, confidence_threshold=None, roi=None):
        """Detect objects in frame with optional ROI filtering"""
        if self.model is None:
            return []
        
        if confidence_threshold is None:
            confidence_threshold = self.config.get("confidence_threshold", 0.3)
        
        try:
            results = self.model(frame, verbose=False, conf=confidence_threshold)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf.item()
                        cls_id = int(box.cls.item())
                        cls_name = self.model.names[cls_id]
                        
                        # Filter by target classes if specified
                        if self.target_classes and cls_name not in self.target_classes:
                            continue
                        
                        # Filter by ROI if specified
                        if roi and not self._is_bbox_in_roi([x1, y1, x2, y2], roi):
                            continue
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': cls_id,
                            'class_name': cls_name
                        }
                        detections.append(detection)
            
            return detections
        except Exception as e:
            LOGGER.error(f"Error in object detection: {e}")
            return []
    
    def _is_bbox_in_roi(self, bbox, roi):
        """Check if bounding box intersects with ROI"""
        x1, y1, x2, y2 = bbox
        roi_x1, roi_y1 = roi["x"], roi["y"]
        roi_x2, roi_y2 = roi_x1 + roi["w"], roi_y1 + roi["h"]
        
        # Check if center of bbox is inside ROI
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return (roi_x1 <= center_x <= roi_x2) and (roi_y1 <= center_y <= roi_y2)

class AnomalyNet(torch.nn.Module):
    def __init__(self, device):
        """AnomalyNet defect detection class with enhanced object detection."""
        super(AnomalyNet, self).__init__()
        self.device = device
        self.feature_buffer = []  # For sequence processing
        self.sequence_length = 16  # Default sequence length
        self.object_detector = None  # Object detection module
        self.speed_analyzer = None
        self.speed_threshold = 0.3  # Configurable threshold
        self.motion_anomaly_weight = 0.3  # Weight for motion in final score

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        feature_window=3,
        window_step=1,
        anomaly_score_num_nn=1,
        featuresampler=anomaly_engine.selectors.feature_selector.IdentitySelector(),
        nn_method=anomaly_engine_core_utils.ProximitySearcher(False, 4),
        motion_config=None,
        sequence_config=None,
        object_detection_config=None,
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device
        self.feature_slicer = FeatureSlicer_obj(feature_window, stride=window_step)
        self.forward_modules = torch.nn.ModuleDict({})
        
        feature_extractor = anomaly_engine_core_utils.FeatureExtractor(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_extractor.feature_dimensions(input_shape)
        self.forward_modules["feature_extractor"] = feature_extractor
        
        preprocessing = anomaly_engine.preprocessing.feature_processor.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing
        self.forward_modules["enhancer"] = anomaly_engine.preprocessing.feature_processor.FeatureEnhancer()
        
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = anomaly_engine.preprocessing.feature_processor.Aggregator(
            target_dim=target_embed_dimension
        )
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator
        
        self.anomaly_rater = anomaly_engine_core_utils.AnomalyRater(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )
        self.anomaly_segmentor = anomaly_engine_core_utils.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )
        self.featuresampler = featuresampler
        
                # Initialize motion speed analyzer
        if motion_config and motion_config.get("enable", False):
            self.speed_analyzer = MotionSpeedAnalyzer(
                method=motion_config.get("method", "farneback")
            )
    
        def _predict_with_motion(self, images, motion_data_list=None):
            """
            Enhanced prediction with motion analysis
            """
            # Get appearance-based predictions
            appearance_scores, appearance_masks = self._predict(images)
            
            # Initialize motion-based scores
            motion_anomaly_scores = np.zeros(len(images))
            motion_anomaly_flags = np.zeros(len(images))
            
            # Process motion data if available
            if motion_data_list and self.speed_analyzer:
                for i, motion_data in enumerate(motion_data_list):
                    if motion_data:
                        speed = motion_data.get('speed', 0)
                        is_slow = motion_data.get('is_slow', False)
                        slowdown_ratio = motion_data.get('slowdown_ratio', 0)
                        
                        # Convert to anomaly score (0-1, higher = more anomalous)
                        motion_anomaly_scores[i] = min(slowdown_ratio, 1.0)
                        motion_anomaly_flags[i] = 1.0 if is_slow else 0.0
            
            # Combine scores
            combined_scores = []
            for i in range(len(images)):
                # Weighted combination of appearance and motion scores
                combined_score = (
                    (1 - self.motion_anomaly_weight) * appearance_scores[i] +
                    self.motion_anomaly_weight * motion_anomaly_scores[i]
                )
                combined_scores.append(combined_score)
            
            return {
                'scores': combined_scores,
                'masks': appearance_masks,
                'appearance_scores': appearance_scores,
                'motion_scores': motion_anomaly_scores,
                'motion_flags': motion_anomaly_flags,
                'combined_scores': combined_scores
            }
        
        def predict_with_motion(self, dataloader):
            """
            Predict with motion analysis for dataloader
            """
            all_results = []
            
            for batch in tqdm.tqdm(dataloader, desc="Detecting anomalies with motion..."):
                images = batch["image"] if isinstance(batch, dict) else batch
                motion_data = batch.get("motion_data", None)
                
                results = self._predict_with_motion(images, motion_data)
                all_results.append(results)
            
            return self._aggregate_results(all_results)
        
        
        # Motion processing
        self.motion_config = motion_config if motion_config else {}
        if self.motion_config.get("enable", False):
            self.motion_processor = MotionProcessor(
                input_channels=1,
                output_dim=self.motion_config.get("motion_embed_dim", 128)
            ).to(self.device)
        
        # Sequence modeling
        self.sequence_config = sequence_config if sequence_config else {}
        if self.sequence_config.get("enable", False):
            self.sequence_length = self.sequence_config.get("sequence_length", 16)
            if self.sequence_config["model_type"] == "lstm":
                self.sequence_model = torch.nn.LSTM(
                    input_size=target_embed_dimension,
                    hidden_size=self.sequence_config["hidden_size"],
                    num_layers=self.sequence_config["num_layers"],
                    batch_first=True
                )
            elif self.sequence_config["model_type"] == "gru":
                self.sequence_model = torch.nn.GRU(
                    input_size=target_embed_dimension,
                    hidden_size=self.sequence_config["hidden_size"],
                    num_layers=self.sequence_config["num_layers"],
                    batch_first=True
                )
            elif self.sequence_config["model_type"] == "transformer":
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=target_embed_dimension,
                    nhead=8
                )
                self.sequence_model = torch.nn.TransformerEncoder(
                    encoder_layer, 
                    num_layers=self.sequence_config["num_layers"]
                )
            
            self.sequence_decoder = torch.nn.Sequential(
                torch.nn.Linear(self.sequence_config["hidden_size"], target_embed_dimension),
                torch.nn.ReLU()
            )
            self.reconstruction_loss = torch.nn.MSELoss()
            self.reconstruction_loss_weight = self.sequence_config.get("reconstruction_loss_weight", 0.5)
            self.sequence_model.to(self.device)
            self.sequence_decoder.to(self.device)
        
        # Object detection
        self.object_detection_config = object_detection_config if object_detection_config else {}
        if self.object_detection_config.get("enable", False):
            self.object_detector = ObjectDetectionModule(self.object_detection_config)

    def set_target_object_classes(self, classes):
        """Set target object classes for detection"""
        if self.object_detector:
            self.object_detector.set_target_classes(classes)

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Extracts feature embeddings for images."""
        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features
        
        _ = self.forward_modules["feature_extractor"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_extractor"](images)
        features = [features[layer] for layer in self.layers_to_extract_from]
        features = [self.feature_slicer.slice_features(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]
        
        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]
            _features = _features.reshape(_features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:])
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(_features.unsqueeze(1), size=(ref_num_patches[0], ref_num_patches[1]), mode="bilinear", align_corners=False)
            _features = _features.squeeze(1)
            _features = _features.reshape(*perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1])
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["enhancer"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        
        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """Trains AnomalyNet by building a feature bank and sequence model."""
        self._fill_feature_bank(training_data)
        
        # Train sequence model if enabled
        if hasattr(self, "sequence_model"):
            self._train_sequence_model(training_data)
        
        # Train object detection targets if enabled
        if self.object_detector:
            self._train_object_detection(training_data)

    def _train_object_detection(self, dataloader):
        """Train object detection by collecting target classes from training data"""
        target_classes = set()
        
        for batch in dataloader:
            if "target_object_class" in batch:
                batch_classes = batch["target_object_class"]
                for cls in batch_classes:
                    if cls and cls != "none":
                        target_classes.add(cls)
        
        if target_classes:
            self.set_target_object_classes(list(target_classes))
            LOGGER.info(f"Trained object detection with target classes: {target_classes}")

    def _train_sequence_model(self, dataloader):
        optimizer = torch.optim.Adam(
            list(self.sequence_model.parameters()) + list(self.sequence_decoder.parameters()), 
            lr=1e-4
        )
        self.sequence_model.train()
        self.sequence_decoder.train()
        
        LOGGER.info("Training sequence model...")
        for epoch in range(5):  # Train for 5 epochs
            total_loss = 0
            
            for batch in dataloader:
                sequences = batch["sequence"].to(self.device)
                batch_size, seq_len, C, H, W = sequences.shape
                
                frame_features = []
                for i in range(seq_len):
                    with torch.no_grad():
                        features = self._embed(sequences[:, i], detach=False)
                        # Handle multi-layer features
                        if isinstance(features, list):
                            features = torch.cat([f.reshape(f.shape[0], -1) for f in features], dim=1)
                        frame_features.append(features)
                        
                # Stack features into a tensor of shape [batch, seq_len, embed_dim]
                frame_features = torch.stack(frame_features, dim=1)
                
                # Forward pass through sequence model
                output, _ = self.sequence_model(frame_features)
                reconstructed = self.sequence_decoder(output)
                
                # Calculate reconstruction loss
                loss = self.reconstruction_loss(reconstructed, frame_features)
                total_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            LOGGER.info(f"Sequence Epoch {epoch+1}/5, Loss: {total_loss/len(dataloader):.4f}")

    def _fill_feature_bank(self, input_data):
        _ = self.forward_modules.eval()
        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)
        features = []
        with tqdm.tqdm(input_data, desc="Building feature database...", position=1, leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))
        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)
        self.anomaly_rater.fit(detection_features=[features])

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """Provides anomaly scores and maps for dataloaders."""
        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        object_detections = []
        
        with tqdm.tqdm(dataloader, desc="Detecting anomalies...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    # Extract object detection info if available
                    object_rois = image.get("object_roi", [None] * len(image["image"]))
                    target_classes = image.get("target_object_class", [None] * len(image["image"]))
                    batch_object_info = []
                    for roi, cls in zip(object_rois, target_classes):
                        batch_object_info.append({"roi": roi, "class": cls})
                    object_detections.extend(batch_object_info)
                    image_data = image["image"]
                else:
                    image_data = image
                    object_detections.extend([None] * len(image_data))
                
                _scores, _masks = self._predict(image_data)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        
        return scores, masks, labels_gt, masks_gt, object_detections

    def _predict(self, images, motion_features=None):
        """Infers anomaly scores and maps for a batch of images with object detection."""
        images = images.to(torch.float).to(self.device)
        if motion_features is not None:
            motion_features = motion_features.to(self.device)
        else:
            # Create zero motion features if not provided
            batch_size, _, h, w = images.shape
            motion_features = torch.zeros(batch_size, 1, h, w).to(self.device)
        
        _ = self.forward_modules.eval()
        batch_size = images.shape[0]
        
        # Convert images to numpy for object detection
        numpy_images = []
        for i in range(batch_size):
            img_tensor = images[i]
            # Denormalize if needed
            img_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            numpy_images.append(img_np)
        
        with torch.no_grad():
            with autocast():
                # Extract appearance features
                features, patch_shapes = self._embed(images, provide_patch_shapes=True)
                features = np.asarray(features)
                
                # Process motion features if available
                if motion_features is not None and hasattr(self, "motion_processor"):
                    motion_embeds = self.motion_processor(motion_features)
                    
                    # Project if dimension mismatch
                    if motion_embeds.shape[1] != self.target_embed_dimension:
                        if not hasattr(self, 'motion_projection'):
                            self.motion_projection = torch.nn.Linear(
                                motion_embeds.shape[1], self.target_embed_dimension
                            ).to(self.device)
                        motion_embeds = self.motion_projection(motion_embeds)
                    
                    # Expand to match patch count
                    patches_per_image = features.shape[0] // images.shape[0]
                    motion_embeds = motion_embeds.unsqueeze(1).repeat(1, patches_per_image, 1)
                    motion_embeds = motion_embeds.view(-1, self.target_embed_dimension)
                    
                    # Fuse features
                    features = features + motion_embeds.detach().cpu().numpy() * 0.5

                region_scores = image_scores = self.anomaly_rater.predict([features])[0]
                image_scores = self.feature_slicer.unslice_scores(image_scores, batchsize=batch_size)
                image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
                image_scores = self.feature_slicer.score(image_scores)
                region_scores = self.feature_slicer.unslice_scores(region_scores, batchsize=batch_size)
                scales = patch_shapes[0]
                region_scores = region_scores.reshape(batch_size, scales[0], scales[1])
                masks = self.anomaly_segmentor.convert_to_segmentation(region_scores)
                
                # Sequence-based anomaly detection
                if hasattr(self, "sequence_model"):
                    seq_scores = self._detect_sequence_anomalies(features)
                    # Combine scores
                    image_scores = [img_score * (1 - self.reconstruction_loss_weight) + 
                                  seq_score * self.reconstruction_loss_weight
                                  for img_score, seq_score in zip(image_scores, seq_scores)]
        
        return [score for score in image_scores], [mask for mask in masks]

    def predict_with_objects(self, images, roi=None, motion_features=None):
        """Enhanced prediction with object detection"""
        # Get anomaly predictions
        anomaly_scores, anomaly_masks = self._predict(images, motion_features)
        
        # Get object detections
        object_detections = []
        if self.object_detector:
            # Convert tensor images to numpy for object detection
            numpy_images = []
            for i in range(images.shape[0]):
                img_tensor = images[i]
                # Denormalize image
                img_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
                # Convert from [-1,1] to [0,255] range approximately
                img_np = ((img_np + 1) * 127.5).astype(np.uint8)
                numpy_images.append(img_np)
            
            for img_np in numpy_images:
                detections = self.object_detector.detect_objects(
                    img_np, 
                    roi=roi
                )
                object_detections.append(detections)
        
        return anomaly_scores, anomaly_masks, object_detections

    def _detect_sequence_anomalies(self, features):
        """Detects anomalies based on sequence inconsistencies."""
        self.sequence_model.eval()
        anomaly_scores = []
        
        # Process each sample in the batch
        for i in range(len(features)):
            # Add to feature buffer
            self.feature_buffer.append(features[i])
            if len(self.feature_buffer) > self.sequence_length:
                self.feature_buffer.pop(0)
            
            if len(self.feature_buffer) == self.sequence_length:
                # Form a sequence
                seq = torch.tensor(np.array(self.feature_buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output, _ = self.sequence_model(seq)
                    reconstructed = self.sequence_decoder(output)
                    
                    # Calculate reconstruction error
                    error = torch.mean((seq - reconstructed)**2, dim=-1)
                    # Use the last frame's error as the anomaly score
                    seq_score = error[0, -1].item()
            else:
                seq_score = 0.0  # Not enough frames
            
            anomaly_scores.append(seq_score)
        
        return anomaly_scores

    def update_with_new_frames(self, new_data_loader, update_percentage=1.0, til_cfg=None, is_ddp=False, selector_config=None):
        """
        Update model with new frames - PRESERVE ALL EXISTING FEATURES
        """
        print(f"🔄 Starting model update with {len(new_data_loader)} new frames...")
        
        # Get current feature count before update - CRITICAL FOR VERIFICATION
        current_count = self.anomaly_rater.get_feature_count()
        print(f"📊 CURRENT feature bank: {current_count} features")
        
        if current_count == 0:
            print("⚠️ CRITICAL: Current feature bank is EMPTY! This will break the model.")
            return False
        
        # Extract features from new frames
        new_features = []
        frame_count = 0
        
        with torch.no_grad():
            for batch in tqdm.tqdm(new_data_loader, desc="Extracting features from new frames"):
                if isinstance(batch, dict):
                    images = batch["image"]
                else:
                    images = batch
                    
                if images is None or images.nelement() == 0:
                    print("⚠️ Skipping empty batch")
                    continue
                    
                images = images.to(torch.float).to(self.device)
                frame_count += images.shape[0]
                
                try:
                    # Handle tiling if enabled
                    if til_cfg and til_cfg.get("enable", False):
                        from train_tuple import TiledTrainIterator
                        tiled_iterator = TiledTrainIterator(
                            loader=[batch],
                            rows=til_cfg["rows"],
                            cols=til_cfg["cols"],
                            overlap=til_cfg["overlap"],
                            ddp_enabled=is_ddp,
                            timers=None
                        )
                        
                        # Extract features from each tile
                        for tile_batch in tiled_iterator:
                            tile_images = tile_batch["image"] if isinstance(tile_batch, dict) else tile_batch
                            tile_images = tile_images.to(torch.float).to(self.device)
                            features = self._embed(tile_images)
                            if features is not None and len(features) > 0:
                                new_features.append(features)
                    else:
                        # No tiling - process full images
                        features = self._embed(images)
                        if features is not None and len(features) > 0:
                            new_features.append(features)
                            
                except Exception as e:
                    print(f"❌ Error extracting features from batch: {e}")
                    continue
        
        print(f"📸 Processed {frame_count} frames")
        
        if new_features:
            try:
                new_features_combined = np.concatenate(new_features, axis=0)
                print(f"🎯 Extracted {len(new_features_combined)} NEW features from {frame_count} frames")
                
                # CRITICAL: Verify we're not losing existing features
                if current_count > 0 and len(new_features_combined) > 0:
                    print(f"🔒 PRESERVING {current_count} existing features + adding {len(new_features_combined)} new features")
                
                # Apply the SAME feature selection as training
                if selector_config and selector_config.get("name") == "approx_greedy_coreset":
                    from anomaly_engine.selectors.feature_selector import ApproximateGreedyCoresetSelector
                    
                    selector_percentage = selector_config.get("percentage", 0.01)
                    print(f"🎯 Applying approx_greedy_coreset selection with {selector_percentage*100}%")
                    
                    update_selector = ApproximateGreedyCoresetSelector(
                        percentage=selector_percentage,
                        device=self.device,
                        number_of_starting_points=10,
                        dimension_to_project_features_to=128
                    )
                    
                    sampled_features = update_selector.run(new_features_combined)
                    print(f"🎯 Selected {len(sampled_features)} features from {len(new_features_combined)} using coreset")
                else:
                    # Use model's existing featuresampler
                    sampled_features = self.featuresampler.run(new_features_combined)
                    print(f"🎯 Selected {len(sampled_features)} features using model's featuresampler")
                
                # CRITICAL: Update the anomaly rater - this should PRESERVE old features
                print("🔄 Updating anomaly rater with new features...")
                success = self.anomaly_rater.update([sampled_features])
                
                if success:
                    new_count = self.anomaly_rater.get_feature_count()
                    added_count = new_count - current_count
                    
                    # VERIFY: Feature count should INCREASE or stay the same
                    if new_count < current_count:
                        print(f"❌ CRITICAL ERROR: Features LOST! {current_count} → {new_count}")
                        return False
                    elif new_count == current_count:
                        print(f"⚠️  No features added. Check selection percentage.")
                    else:
                        print(f"✅ Model update successful!")
                        print(f"📈 Features: {current_count} → {new_count} (+{added_count})")
                    
                    # Verify feature distribution
                    if hasattr(self.anomaly_rater, 'detection_features') and self.anomaly_rater.detection_features is not None:
                        feature_stats = self.anomaly_rater.detection_features
                        print(f"📊 Final feature stats - Min: {np.min(feature_stats):.4f}, Max: {np.max(feature_stats):.4f}, Mean: {np.mean(feature_stats):.4f}")
                    
                    return True
                else:
                    print("❌ Failed to update anomaly rater")
                    return False
                    
            except Exception as e:
                print(f"❌ Error during feature processing: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("❌ No features extracted from new frames")
            return False

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "dime_params.pkl")

    # In anomaly_net.py - Fix the save_to_path method

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving model parameters.")
        # CRITICAL: Always save features separately for updates
        self.anomaly_rater.save(save_path, save_features_separately=True, prepend=prepend)
        
        # Include all configs in parameters
        anomaly_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules["preprocessing"].output_dim,
            "target_embed_dimension": self.forward_modules["preadapt_aggregator"].target_dim,
            "feature_window": self.feature_slicer.feature_window,
            "window_step": self.feature_slicer.stride,
            "anomaly_scorer_num_nn": self.anomaly_rater.n_nearest_neighbours,
            "motion_config": getattr(self, "motion_config", {}),
            "sequence_config": getattr(self, "sequence_config", {}),
            "object_detection_config": getattr(self, "object_detection_config", {})
        }
        
        # Save sequence model if exists
        if hasattr(self, "sequence_model"):
            sequence_state = {
                "sequence_model": self.sequence_model.state_dict(),
                "sequence_decoder": self.sequence_decoder.state_dict()
            }
            torch.save(sequence_state, os.path.join(save_path, prepend + "sequence_model.pth"))
        
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(anomaly_params, save_file)
        
        # Also save feature count for verification
        feature_count = self.anomaly_rater.get_feature_count()
        print(f"💾 Saved model with {feature_count} features to {save_path}")

    def load_from_path(self, load_path: str, device: torch.device, nn_method: anomaly_engine_core_utils.ProximitySearcher(False, 4), prepend: str = "") -> None:
        LOGGER.info("Loading and initializing model.")
        
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            anomaly_params = pickle.load(load_file)
        
        # Extract configs BEFORE unpacking
        motion_config = anomaly_params.pop("motion_config", {})
        sequence_config = anomaly_params.pop("sequence_config", {})
        object_detection_config = anomaly_params.pop("object_detection_config", {})
            
        anomaly_params["backbone"] = anomaly_engine.models.network_models.load(anomaly_params["backbone.name"])
        anomaly_params["backbone"].name = anomaly_params["backbone.name"]
        del anomaly_params["backbone.name"]
        
        # Now pass configs explicitly
        self.load(**anomaly_params, device=device, nn_method=nn_method, 
                motion_config=motion_config, sequence_config=sequence_config,
                object_detection_config=object_detection_config)
        self.anomaly_rater.load(load_path, prepend)
        
        # Load sequence model if exists
        if sequence_config.get("enable", False) and os.path.exists(os.path.join(load_path, prepend + "sequence_model.pth")):
            sequence_state = torch.load(os.path.join(load_path, prepend + "sequence_model.pth"))
            self.sequence_model.load_state_dict(sequence_state["sequence_model"])
            self.sequence_decoder.load_state_dict(sequence_state["sequence_decoder"])