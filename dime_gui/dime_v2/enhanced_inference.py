#!/usr/bin/env python3
"""
Enhanced Anomaly Detection with Golden Frame Comparison and Multi-ROI Support
FIXED VERSION: Proper coordinate handling for multi-ROI mode
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import time
import torch
from collections import deque

# Import your existing inference module
from .inference import AnomalyInference, _is_multi_model_root, MultiROIManager, AnomalyDetectionResult



class CameraMovementCompensator:
    """
    Compensate for camera movement by registering frames and adjusting ROI coordinates
    Uses feature matching to detect camera shake and adjust ROI positions
    """
    
    def __init__(self, 
                 reference_frame: Optional[np.ndarray] = None,
                 enable_compensation: bool = True,
                 movement_threshold: float = 5.0,  # pixels
                 max_movement: float = 50.0,  # pixels - maximum allowed movement
                 use_orb_features: bool = True,
                 num_features: int = 500,
                 reference_frame_idx: int = 3):  # Use 3rd frame as reference
                 
        self.enable_compensation = enable_compensation
        self.movement_threshold = movement_threshold
        self.max_movement = max_movement
        self.use_orb_features = use_orb_features
        self.num_features = num_features
        self.reference_frame_idx = reference_frame_idx
        
        # Reference frame and features
        self.reference_frame = None
        self.ref_gray = None
        self.ref_keypoints = None
        self.ref_descriptors = None
        
        # Movement tracking
        self.movement_history = deque(maxlen=30)  # Store recent movements
        self.current_movement = (0.0, 0.0)  # (dx, dy)
        self.movement_smoothed = (0.0, 0.0)
        self.frame_idx = 0
        
        # Feature matchers
        if self.use_orb_features:
            self.orb = cv2.ORB_create(nfeatures=num_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self.sift = cv2.SIFT_create(nfeatures=num_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        # Initialize with reference frame if provided
        if reference_frame is not None:
            self.set_reference_frame(reference_frame)
    
    def set_reference_frame(self, frame: np.ndarray):
        """Set reference frame for movement compensation"""
        self.reference_frame = frame.copy()
        self.ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract features from reference frame
        if self.use_orb_features:
            self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(self.ref_gray, None)
        else:
            self.ref_keypoints, self.ref_descriptors = self.sift.detectAndCompute(self.ref_gray, None)
        
        print(f"📊 [Movement Compensation] Reference frame set - Features: {len(self.ref_keypoints)}")
        
        # Reset movement tracking
        self.movement_history.clear()
        self.current_movement = (0.0, 0.0)
        self.movement_smoothed = (0.0, 0.0)
        self.frame_idx = 0
    
    def set_reference_from_idx(self, video_path: str, frame_idx: int = 3) -> bool:
        """Set reference frame from specific frame index in video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            
            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_idx >= total_frames:
                frame_idx = total_frames - 1
            
            # Set frame position and read
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                self.set_reference_frame(frame)
                self.reference_frame_idx = frame_idx
                return True
        
        except Exception as e:
            print(f"❌ [Movement Compensation] Error setting reference from video: {e}")
        
        return False
    
    def _calculate_movement_orb(self, current_gray: np.ndarray) -> Tuple[float, float, float]:
        """Calculate movement using ORB feature matching"""
        if self.ref_keypoints is None or self.ref_descriptors is None:
            return 0.0, 0.0, 0.0
        
        # Detect features in current frame
        current_kp, current_desc = self.orb.detectAndCompute(current_gray, None)
        
        if current_desc is None or len(current_kp) < 10:
            return 0.0, 0.0, 0.0
        
        # Match features
        matches = self.matcher.match(self.ref_descriptors, current_desc)
        
        if len(matches) < 10:
            return 0.0, 0.0, 0.0
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Take top matches (filter outliers)
        good_matches = matches[:min(50, len(matches))]
        
        # Calculate average movement
        dx_list = []
        dy_list = []
        distances = []
        
        for match in good_matches:
            ref_pt = self.ref_keypoints[match.queryIdx].pt
            cur_pt = current_kp[match.trainIdx].pt
            dx = cur_pt[0] - ref_pt[0]
            dy = cur_pt[1] - ref_pt[1]
            dx_list.append(dx)
            dy_list.append(dy)
            distances.append(match.distance)
        
        # Use median to avoid outliers
        dx = np.median(dx_list)
        dy = np.median(dy_list)
        confidence = np.mean(distances) / 100.0  # Normalize confidence
        
        return dx, dy, confidence
    
    def _calculate_movement_phase_correlation(self, current_gray: np.ndarray) -> Tuple[float, float, float]:
        """Calculate movement using phase correlation (fast method)"""
        try:
            # Use phase correlation for translation
            shift, _ = cv2.phaseCorrelate(self.ref_gray.astype(np.float32), 
                                         current_gray.astype(np.float32))
            dx, dy = shift[0], shift[1]
            
            # Calculate confidence based on response
            confidence = min(1.0, abs(dx) + abs(dy)) / self.max_movement
            
            return dx, dy, confidence
        
        except Exception as e:
            return 0.0, 0.0, 0.0
    
    def detect_movement(self, frame: np.ndarray) -> Tuple[float, float, float]:
        """Detect movement between current frame and reference frame"""
        if not self.enable_compensation or self.ref_gray is None:
            return 0.0, 0.0, 1.0
        
        # Convert to grayscale
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate movement using different methods
        if self.use_orb_features:
            dx, dy, confidence = self._calculate_movement_orb(current_gray)
        else:
            dx, dy, confidence = self._calculate_movement_phase_correlation(current_gray)
        
        # Limit maximum movement
        dx = np.clip(dx, -self.max_movement, self.max_movement)
        dy = np.clip(dy, -self.max_movement, self.max_movement)
        
        # Smooth movement using history
        self.movement_history.append((dx, dy, confidence))
        
        # Calculate smoothed movement (weighted average)
        if self.movement_history:
            weights = []
            movements_x = []
            movements_y = []
            
            for i, (mx, my, conf) in enumerate(self.movement_history):
                weight = conf * (0.95 ** i)  # Recent movements have higher weight
                weights.append(weight)
                movements_x.append(mx)
                movements_y.append(my)
            
            # Normalize weights
            if sum(weights) > 0:
                weights = [w / sum(weights) for w in weights]
                dx_smooth = sum(mx * w for mx, w in zip(movements_x, weights))
                dy_smooth = sum(my * w for my, w in zip(movements_y, weights))
            else:
                dx_smooth, dy_smooth = dx, dy
        else:
            dx_smooth, dy_smooth = dx, dy
        
        self.current_movement = (dx, dy)
        self.movement_smoothed = (dx_smooth, dy_smooth)
        self.frame_idx += 1
        
        # Check if movement is significant
        movement_magnitude = np.sqrt(dx_smooth**2 + dy_smooth**2)
        is_significant = movement_magnitude > self.movement_threshold
        
        return dx_smooth, dy_smooth, 1.0 if is_significant else 0.0
    
    def adjust_roi_coordinates(self, roi_coords: List[Dict], dx: float, dy: float) -> List[Dict]:
        """Adjust ROI coordinates based on detected movement"""
        if not roi_coords or (abs(dx) < 0.1 and abs(dy) < 0.1):
            return roi_coords
        
        adjusted_coords = []
        
        for roi in roi_coords:
            # Adjust rectangle coordinates
            if 'rect' in roi:
                rect = roi['rect'].copy()
                rect['x'] = int(rect['x'] + dx)
                rect['y'] = int(rect['y'] + dy)
                
                # Ensure coordinates are within bounds
                roi['rect'] = rect
            
            # Adjust polygon coordinates if present
            if 'polygon' in roi:
                polygon = roi['polygon'].copy()
                adjusted_polygon = [(int(x + dx), int(y + dy)) for x, y in polygon]
                roi['polygon'] = adjusted_polygon
            
            adjusted_coords.append(roi)
        
        return adjusted_coords
    
    def get_movement_summary(self) -> Dict:
        """Get movement analysis summary"""
        if not self.movement_history:
            return {}
        
        movements = np.array([(m[0], m[1]) for m in self.movement_history])
        dx_values = movements[:, 0]
        dy_values = movements[:, 1]
        
        return {
            'current_dx': float(self.current_movement[0]),
            'current_dy': float(self.current_movement[1]),
            'smoothed_dx': float(self.movement_smoothed[0]),
            'smoothed_dy': float(self.movement_smoothed[1]),
            'avg_dx': float(np.mean(dx_values)),
            'avg_dy': float(np.mean(dy_values)),
            'max_dx': float(np.max(np.abs(dx_values))),
            'max_dy': float(np.max(np.abs(dy_values))),
            'movement_magnitude': float(np.sqrt(self.current_movement[0]**2 + self.current_movement[1]**2)),
            'total_frames': self.frame_idx,
            'history_length': len(self.movement_history)
        }
    
    def reset(self):
        """Reset movement compensation"""
        self.movement_history.clear()
        self.current_movement = (0.0, 0.0)
        self.movement_smoothed = (0.0, 0.0)
        self.frame_idx = 0
    
    def update_reference_frame(self, frame: np.ndarray):
        """Update reference frame (e.g., after large movement or periodic refresh)"""
        self.set_reference_frame(frame)



class EnhancedAnomalyInference:
    """
    Enhanced anomaly detection with golden frame comparison for all model types
    - FIXED: Proper coordinate handling for multi-ROI mode
    - ADDED: Camera movement compensation
    """
    
    def __init__(self, 
                 model_path: str,
                 # Original parameters
                 threshold: Optional[float] = None,
                 imagesize: Optional[Tuple[int, int]] = None,
                 skip_frames: int = 3,
                 mask_alpha: float = 0.5,
                 proximity_on_gpu: bool = False,
                 tile_rows: int = 2,
                 tile_cols: int = 2, 
                 tile_overlap: float = 0.10,
                 parallel_tiles: bool = True,
                 num_workers: int = 4,
                 resource_sample_interval: float = 2.0,
                 output_path: str = "output",
                 
                 # Enhanced parameters
                 enable_golden_comparison: bool = False,
                 golden_image_path: Optional[str] = None,
                 comparison_padding: int = 20,
                 enable_change_analysis: bool = True,
                 change_threshold: float = 0.3,
                 golden_similarity_threshold: float = 0.85,
                 enable_visualization: bool = True,
                 save_comparison_results: bool = True,
                 min_change_area: int = 100,
                 feature_match_threshold: float = 0.3,
                 show_all_anomalies: bool = False,
                 structural_change_threshold: float = 0.8,
                 
                 # NEW: Camera movement compensation parameters
                 enable_movement_compensation: bool = True,
                 movement_threshold: float = 5.0,
                 max_movement: float = 50.0,
                 reference_frame_idx: int = 3,
                 auto_update_reference: bool = False,
                 update_reference_interval: int = 100):
        
        # Store enhanced parameters
        self.enable_golden_comparison = enable_golden_comparison
        self.golden_image_path = golden_image_path
        self.comparison_padding = comparison_padding
        self.enable_change_analysis = enable_change_analysis
        self.change_threshold = change_threshold
        self.golden_similarity_threshold = golden_similarity_threshold
        self.enable_visualization = enable_visualization
        self.save_comparison_results = save_comparison_results
        self.min_change_area = min_change_area
        self.feature_match_threshold = feature_match_threshold
        self.structural_change_threshold = structural_change_threshold
        self.show_all_anomalies = show_all_anomalies
        
        # NEW: Camera movement compensation parameters
        self.enable_movement_compensation = enable_movement_compensation
        self.movement_threshold = movement_threshold
        self.max_movement = max_movement
        self.reference_frame_idx = reference_frame_idx
        self.auto_update_reference = auto_update_reference
        self.update_reference_interval = update_reference_interval
        
        # Initialize golden image
        self.golden_image = None
        self.golden_features = {}
        
        if self.enable_golden_comparison and golden_image_path:
            self.load_golden_image(golden_image_path)
        
        # NEW: Initialize camera movement compensator
        self.movement_compensator = None
        if self.enable_movement_compensation:
            self.movement_compensator = CameraMovementCompensator(
                enable_compensation=True,
                movement_threshold=self.movement_threshold,
                max_movement=self.max_movement,
                reference_frame_idx=self.reference_frame_idx
            )
        
        # Initialize the original anomaly inference
        self.anomaly_infer = AnomalyInference(
            model_path=model_path,
            threshold=threshold,
            imagesize=imagesize,
            skip_frames=3,
            mask_alpha=mask_alpha,
            proximity_on_gpu=proximity_on_gpu,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            tile_overlap=tile_overlap,
            parallel_tiles=parallel_tiles,
            num_workers=num_workers,
            resource_sample_interval=resource_sample_interval,
            output_path=output_path
        )
        
        # Load the model
        # self.model = self.anomaly_infer.load_model()
        
        # Setup Multi-ROI if needed
        self.multi_mgr = None
        self.original_roi_coords = {}  # Store original ROI coordinates
        
        if _is_multi_model_root(model_path):
            self.multi_mgr = MultiROIManager(model_path, self.anomaly_infer)
            
            # Enable parallel ROI processing
            self.multi_mgr.parallel_rois = False
            self.multi_mgr.roi_workers = None
            
            # Also enable parallel tiles within each ROI
            for det in self.multi_mgr.detectors:
                det['infer'].parallel_tiles = True
                det['infer'].num_workers = 4
            
            # Store original ROI coordinates for movement compensation
            self._store_original_roi_coordinates()
        
        self.anomaly_infer.multi_mgr = self.multi_mgr
        
        # Additional optimizations
        self.anomaly_infer.use_compile = True
        self.anomaly_infer.use_pinned_memory = True
        
        # Movement compensation state
        self.compensated_frames = 0
        self.total_movement_detected = 0
    
    def _store_original_roi_coordinates(self):
        """Store original ROI coordinates for movement compensation"""
        if not self.multi_mgr:
            return
        
        for det in self.multi_mgr.detectors:
            roi_name = det["name"]
            roi_dir = det["dir"]
            
            # Read ROI metadata
            meta_path = os.path.join(roi_dir, "roi_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                
                rect = meta.get("rectangle", {})
                polygon = meta.get("polygon", [])
                
                self.original_roi_coords[roi_name] = {
                    "rect": rect.copy(),
                    "polygon": polygon.copy() if polygon else []
                }
    
    def set_reference_frame_for_movement(self, frame: np.ndarray):
        """Set reference frame for camera movement compensation"""
        if self.movement_compensator:
            self.movement_compensator.set_reference_frame(frame)
            print("📊 [Movement Compensation] Reference frame set")
            return True
        return False
    
    def set_reference_from_video_frame(self, video_path: str, frame_idx: int = 3):
        """Set reference frame from specific frame in video"""
        if self.movement_compensator:
            success = self.movement_compensator.set_reference_from_idx(video_path, frame_idx)
            if success:
                print(f"📊 [Movement Compensation] Reference frame set from video frame {frame_idx}")
            return success
        return False
    
    def _adjust_rois_for_movement(self, frame: np.ndarray) -> Tuple[bool, float, float]:
        """Detect camera movement and adjust ROI coordinates"""
        if not self.enable_movement_compensation or not self.movement_compensator or not self.multi_mgr:
            return False, 0.0, 0.0
        
        # Detect movement
        dx, dy, confidence = self.movement_compensator.detect_movement(frame)
        
        # Check if movement is significant
        movement_magnitude = np.sqrt(dx**2 + dy**2)
        is_significant = movement_magnitude > self.movement_threshold
        
        if not is_significant:
            return False, dx, dy
        
        # Update ROI coordinates in MultiROIManager
        for det in self.multi_mgr.detectors:
            roi_name = det["name"]
            
            if roi_name in self.original_roi_coords:
                original_coords = self.original_roi_coords[roi_name]
                
                # Adjust rectangle
                if 'rect' in original_coords:
                    rect = original_coords['rect'].copy()
                    rect['x'] = int(rect['x'] + dx)
                    rect['y'] = int(rect['y'] + dy)
                    
                    # Store adjusted coordinates back in detector metadata
                    meta_path = os.path.join(det["dir"], "roi_meta.json")
                    if os.path.exists(meta_path):
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        
                        meta['rectangle'] = rect
                        
                        # Also adjust polygon if present
                        if 'polygon' in original_coords and original_coords['polygon']:
                            polygon = original_coords['polygon'].copy()
                            adjusted_polygon = [(int(x + dx), int(y + dy)) for x, y in polygon]
                            meta['polygon'] = adjusted_polygon
                        
                        # Save back
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(meta, f, indent=2)
        
        self.compensated_frames += 1
        self.total_movement_detected += movement_magnitude
        
        # Auto-update reference frame if enabled
        if (self.auto_update_reference and 
            self.compensated_frames % self.update_reference_interval == 0):
            self.movement_compensator.update_reference_frame(frame)
            print(f"🔄 [Movement Compensation] Reference frame updated at frame {self.compensated_frames}")
        
        return True, dx, dy
    
    def _restore_original_roi_coordinates(self):
        """Restore original ROI coordinates"""
        if not self.multi_mgr:
            return
        
        for det in self.multi_mgr.detectors:
            roi_name = det["name"]
            
            if roi_name in self.original_roi_coords:
                original_coords = self.original_roi_coords[roi_name]
                
                # Restore original coordinates
                meta_path = os.path.join(det["dir"], "roi_meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    
                    meta['rectangle'] = original_coords['rect'].copy()
                    
                    if 'polygon' in original_coords:
                        meta['polygon'] = original_coords['polygon'].copy()
                    
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
    
    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Dict:
        """Enhanced frame processing with camera movement compensation"""
        total_start_time = time.time()
        timing_breakdown = {}
        
        # NEW: Camera movement compensation
        movement_info = {}
        if self.enable_movement_compensation and frame_idx >= self.reference_frame_idx:
            movement_start = time.time()
            movement_detected, dx, dy = self._adjust_rois_for_movement(frame)
            movement_info = {
                'movement_detected': movement_detected,
                'dx': float(dx),
                'dy': float(dy),
                'magnitude': float(np.sqrt(dx**2 + dy**2))
            }
            
            if self.movement_compensator:
                movement_summary = self.movement_compensator.get_movement_summary()
                movement_info.update(movement_summary)
            
            timing_breakdown['movement_compensation_ms'] = float((time.time() - movement_start) * 1000)
        
        # Time anomaly detection
        anomaly_start = time.time()
        
        # Use unified processing for all model types
        if self.multi_mgr is None:
            # Single-detector path
            result = self.anomaly_infer.process_single_frame(frame)
            anomaly_score = result.anomaly_score
            is_anom = result.is_anomaly
            anomaly_areas = result.anomaly_areas
            segmentation_map = result.segmentation_map
            original_overlay = result.processed_frame
        else:
            # Multi-ROI path with potential movement compensation
            try:
                # Clear temporal buffers to prevent shape mismatches
                if hasattr(self.multi_mgr, '_roi_buffers'):
                    for buffer in self.multi_mgr._roi_buffers.values():
                        buffer.clear()
                
                # Also clear temporal buffers in child detectors
                for det in self.multi_mgr.detectors:
                    child_infer = det['infer']
                    if hasattr(child_infer, 'temporal_buffer'):
                        child_infer.temporal_buffer.clear()
                    if hasattr(child_infer, '_sf_last_segmentation'):
                        child_infer._sf_last_segmentation = None
                
                result = self.multi_mgr.process_frame_all(frame)
                anomaly_score = result.anomaly_score
                is_anom = result.is_anomaly
                anomaly_areas = result.anomaly_areas
                segmentation_map = None
                original_overlay = result.processed_frame
            except Exception as e:
                print(f"❌ Multi-ROI processing failed: {e}")
                # Restore original coordinates and retry
                self._restore_original_roi_coordinates()
                
                # Fallback to single detector if multi-ROI fails
                result = self.anomaly_infer.process_single_frame(frame)
                anomaly_score = result.anomaly_score
                is_anom = result.is_anomaly
                anomaly_areas = result.anomaly_areas
                segmentation_map = result.segmentation_map
                original_overlay = result.processed_frame
                
        timing_breakdown['anomaly_detection_ms'] = float((time.time() - anomaly_start) * 1000)
        
        # Perform golden image comparison if enabled
        change_analysis = {}
        if self.enable_golden_comparison and self.golden_image is not None:
            change_start = time.time()
            change_analysis = self._analyze_anomaly_areas_with_golden(
                frame, anomaly_areas, segmentation_map
            )
            timing_breakdown['change_analysis_ms'] = float((time.time() - change_start) * 1000)
        
        # Determine final anomaly status based on both score and actual changes
        has_significant_changes = change_analysis.get('significant_changes_found', False)
        false_positives_filtered = change_analysis.get('false_positives_filtered', 0)
        final_is_anomaly = is_anom and has_significant_changes
        
        # Create enhanced visualization with movement info
        viz_start = time.time()
        if self.enable_visualization:
            enhanced_overlay = self._create_visualization_with_movement(
                frame, change_analysis, anomaly_score, anomaly_areas, movement_info
            )
        else:
            enhanced_overlay = original_overlay
        timing_breakdown['visualization_ms'] = float((time.time() - viz_start) * 1000)
        
        # Calculate total time
        total_time = time.time() - total_start_time
        timing_breakdown['total_processing_ms'] = float(total_time * 1000)
        
        # Prepare results
        results = {
            'frame_index': int(frame_idx),
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anom),  # Original anomaly detection
            'final_is_anomaly': bool(final_is_anomaly),  # Final judgment after change analysis
            'has_significant_changes': bool(has_significant_changes),
            'false_positives_filtered': int(false_positives_filtered),
            'processing_time': float(total_time),
            'timing_breakdown': timing_breakdown,
            'change_analysis': self._convert_to_serializable(change_analysis),
            'anomaly_areas': anomaly_areas,  # Include original anomaly areas
            'movement_compensation': movement_info,  # NEW: Include movement info
            'overlay_frame': enhanced_overlay
            
        }
        
        # Save results if enabled
        if self.save_comparison_results:
            self._save_results(results, frame_idx)
        
        return results
    
    def _create_visualization_with_movement(self, current_frame: np.ndarray, 
                                           analysis: Dict, anomaly_score: float,
                                           original_anomaly_areas: List[Dict] = None,
                                           movement_info: Dict = None) -> np.ndarray:
        """Create enhanced visualization with movement compensation info"""
        if not self.enable_visualization:
            return current_frame
        
        viz_frame = current_frame.copy()
        h, w = viz_frame.shape[:2]
        
        # Determine final status based on anomaly score AND actual changes found
        has_significant_changes = analysis.get('significant_changes_found', False)
        false_positives_filtered = analysis.get('false_positives_filtered', 0)
        is_anomaly_by_score = anomaly_score >= self.anomaly_infer.threshold
        
        # IMPROVED STATUS LOGIC with show_all_anomalies flag
        if self.show_all_anomalies and is_anomaly_by_score:
            status = "ANOMALY (All Shown)"
            status_color = (0, 165, 255)  # Orange for forced display
            should_draw_boxes = True
        elif is_anomaly_by_score and not has_significant_changes and false_positives_filtered > 0:
            status = "PASS (False Positives Filtered)"
            status_color = (0, 255, 255)  # Yellow for pass
            should_draw_boxes = False
        elif is_anomaly_by_score and has_significant_changes:
            status = "ANOMALY"
            status_color = (0, 0, 255)  # Red for anomaly
            should_draw_boxes = True
        else:
            status = "NORMAL" 
            status_color = (0, 255, 0)  # Green for normal
            should_draw_boxes = False
        
        # Draw regions based on should_visualize flag AND show_all_anomalies flag
        visualized_regions = 0
        if should_draw_boxes:
            for region in analysis.get('anomaly_regions', []):
                # NEW: Show all if flag is set, otherwise use original logic
                if self.show_all_anomalies or region.get('should_visualize', False):
                    x1, y1, x2, y2 = region['bbox']
                    region_analysis = region.get('analysis', {})
                    similarity_score = region_analysis.get('similarity_score', 0.0)
                    change_type = region.get('change_type', 'UNKNOWN')
                    
                    # Color code based on change type
                    color_map = {
                        'FALSE_POSITIVE_MATCHES_GOLDEN': (128, 128, 128),  # Gray for false positives
                        'OBJECT_MISSING': (0, 0, 255),        # Red
                        'NEW_OBJECT': (255, 0, 0),            # Blue
                        'OBJECT_MOVED': (0, 255, 255),        # Yellow
                        'OBJECT_MODIFIED': (255, 255, 0),     # Cyan
                        'STRUCTURAL_CHANGE': (0, 165, 255),   # Orange
                        'COLOR_APPEARANCE_CHANGE': (255, 0, 255),  # Magenta
                        'MINOR_CHANGE': (128, 128, 0),        # Olive for minor changes
                    }
                    
                    color = color_map.get(change_type, (128, 128, 128))  # Gray for unknown
                    thickness = 2
                    
                    # Draw bounding box
                    cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Add change type and similarity label
                    label = f"{change_type}: {similarity_score:.2f}"
                    cv2.putText(viz_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    visualized_regions += 1
        
        # Add movement compensation info if available
        if movement_info and movement_info.get('movement_detected', False):
            dx = movement_info.get('dx', 0)
            dy = movement_info.get('dy', 0)
            magnitude = movement_info.get('magnitude', 0)
            
            # Draw movement vector
            center_x, center_y = w // 2, h // 2
            end_x = int(center_x + dx * 5)  # Scale for visibility
            end_y = int(center_y + dy * 5)
            
            # Draw movement arrow
            cv2.arrowedLine(viz_frame, (center_x, center_y), (end_x, end_y),
                          (0, 255, 255), 2, tipLength=0.3)
            
            # Draw movement info text
            movement_text = f"Movement: dx={dx:.1f}, dy={dy:.1f}, mag={magnitude:.1f}"
            cv2.putText(viz_frame, movement_text, (10, h - 100),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw movement indicator
            cv2.circle(viz_frame, (center_x, center_y), 10, (0, 255, 255), -1)
            cv2.putText(viz_frame, "M", (center_x - 5, center_y + 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add analysis information overlay
        change_metrics = analysis.get('change_metrics', {})
        
        # Top-left: Status 
        cv2.putText(viz_frame, status, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Top-right: Timing and processing info
        timing_info = f"Change Analysis: {analysis.get('timing_breakdown', {}).get('region_analysis_total_ms', 0):.0f}ms"
        if movement_info:
            timing_info += f" | Movement: {movement_info.get('magnitude', 0):.1f}px"
        cv2.putText(viz_frame, timing_info, (w - 400, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Bottom-left: Detailed metrics
        y_offset = h - 10
        metrics_text = [
            f"Anomaly Score: {anomaly_score:.1f}",
            f"Visualized Regions: {visualized_regions}",
            f"False Positives Filtered: {false_positives_filtered}",
        ]
        
        # Add movement info if available
        if movement_info and movement_info.get('movement_detected', False):
            metrics_text.append(f"Camera Movement: {movement_info.get('magnitude', 0):.1f}px")
            metrics_text.append(f"Compensated Frames: {self.compensated_frames}")
        
        # Show timing breakdown if available
        timing_breakdown = analysis.get('timing_breakdown', {})
        if 'feature_extraction_ms' in timing_breakdown:
            metrics_text.append(f"Feature Extraction: {timing_breakdown['feature_extraction_ms']:.0f}ms")
        if 'regions_processed' in timing_breakdown:
            metrics_text.append(f"Regions Processed: {timing_breakdown['regions_processed']}")
        
        for text in reversed(metrics_text):
            cv2.putText(viz_frame, text, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset -= 20
        
        return viz_frame
    
    def get_movement_summary(self) -> Dict:
        """Get summary of camera movement compensation"""
        if not self.movement_compensator:
            return {}
        
        movement_stats = self.movement_compensator.get_movement_summary()
        movement_stats.update({
            'compensated_frames': self.compensated_frames,
            'total_movement_detected': self.total_movement_detected,
            'avg_movement_per_frame': self.total_movement_detected / max(self.compensated_frames, 1)
        })
        
        return movement_stats
    
    def enable_movement_compensation_toggle(self, enable: bool):
        """Enable or disable movement compensation"""
        self.enable_movement_compensation = enable
        if self.movement_compensator:
            self.movement_compensator.enable_compensation = enable
        
        if enable:
            print("📊 [Movement Compensation] Enabled")
        else:
            print("📊 [Movement Compensation] Disabled")
    
    def reset_movement_compensation(self):
        """Reset movement compensation to initial state"""
        if self.movement_compensator:
            self.movement_compensator.reset()
        
        self.compensated_frames = 0
        self.total_movement_detected = 0
        
        # Restore original ROI coordinates
        self._restore_original_roi_coordinates()
        
        print("📊 [Movement Compensation] Reset to initial state")

    def set_golden_image_from_frame(self, frame: np.ndarray) -> bool:
        """Set golden image from numpy array frame"""
        try:
            self.golden_image = frame.copy()
            # Convert to RGB for feature extraction
            self.golden_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract features from golden image
            self._extract_golden_features()
            print(f"✅ Golden image set from frame - Shape: {frame.shape}")
            return True
            
        except Exception as e:
            print(f"❌ [Golden Image] Error setting from frame: {e}")
            self.golden_image = None
            return False
    
    def load_golden_image(self, image_path: str):
        """Load and preprocess the golden image"""
        try:
            self.golden_image = cv2.imread(image_path)
            if self.golden_image is None:
                raise ValueError(f"Failed to load golden image from {image_path}")
            
            # Convert to RGB for feature extraction
            self.golden_image_rgb = cv2.cvtColor(self.golden_image, cv2.COLOR_BGR2RGB)
            
            # Extract features from golden image
            self._extract_golden_features()
            print(f"✅ Golden image loaded: {image_path}")
            return True
            
        except Exception as e:
            print(f"❌ [Golden Image] Error loading: {e}")
            self.golden_image = None
            return False
    
    def _extract_golden_features(self):
        """Extract features from golden image for comparison"""
        if self.golden_image_rgb is None:
            return
        
        try:
            # Initialize feature detectors
            self.orb = cv2.ORB_create(500)
            
            # Extract keypoints and descriptors from golden image
            gray_golden = cv2.cvtColor(self.golden_image_rgb, cv2.COLOR_RGB2GRAY)
            
            # ORB features
            self.golden_kp_orb, self.golden_desc_orb = self.orb.detectAndCompute(gray_golden, None)
            
            # Store golden image dimensions
            self.golden_h, self.golden_w = gray_golden.shape
            
            print(f"📊 [Golden Features] ORB: {len(self.golden_kp_orb) if self.golden_kp_orb else 0}")
        
        except Exception as e:
            print(f"❌ [Golden Features] Error extracting features: {e}")
            self.golden_kp_orb, self.golden_desc_orb = [], None

    def _calculate_region_similarity(self, current_region: np.ndarray, golden_region: np.ndarray) -> Dict:
        """Calculate comprehensive similarity metrics between regions"""
        similarity_metrics = {}
        
        try:
            # Resize regions to same dimensions if needed
            if current_region.shape != golden_region.shape:
                current_region = cv2.resize(current_region, (golden_region.shape[1], golden_region.shape[0]))
            
            # 1. SSIM Score
            from skimage.metrics import structural_similarity as ssim
            current_gray = cv2.cvtColor(current_region, cv2.COLOR_BGR2GRAY)
            golden_gray = cv2.cvtColor(golden_region, cv2.COLOR_BGR2GRAY)
            ssim_score, _ = ssim(current_gray, golden_gray, full=True)
            similarity_metrics['ssim'] = float(ssim_score)
            
            # 2. Feature Matching
            kp1, desc1 = self.orb.detectAndCompute(current_gray, None)
            kp2, desc2 = self.orb.detectAndCompute(golden_gray, None)
            
            if desc1 is not None and desc2 is not None and len(desc1) > 0 and len(desc2) > 0:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(desc1, desc2)
                match_ratio = len(matches) / max(len(desc1), len(desc2), 1)
                similarity_metrics['feature_match_ratio'] = float(match_ratio)
            else:
                similarity_metrics['feature_match_ratio'] = 0.0
            
            # 3. Histogram Correlation
            hist_current = cv2.calcHist([current_region], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
            hist_golden = cv2.calcHist([golden_region], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist_current, hist_current, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_golden, hist_golden, 0, 1, cv2.NORM_MINMAX)
            hist_corr = cv2.compareHist(hist_current, hist_golden, cv2.HISTCMP_CORREL)
            similarity_metrics['histogram_correlation'] = float(hist_corr)
            
            # 4. Overall Similarity Score
            overall_similarity = (
                similarity_metrics['ssim'] * 0.4 +
                similarity_metrics['feature_match_ratio'] * 0.4 + 
                max(0, similarity_metrics['histogram_correlation']) * 0.2
            )
            similarity_metrics['overall_similarity'] = float(overall_similarity)
            
            # 5. Similarity classification
            if overall_similarity > 0.95:
                similarity_metrics['similarity_class'] = "VERY_HIGH"
            elif overall_similarity > 0.85:
                similarity_metrics['similarity_class'] = "HIGH"
            elif overall_similarity > 0.70:
                similarity_metrics['similarity_class'] = "MEDIUM"
            else:
                similarity_metrics['similarity_class'] = "LOW"
                
        except Exception as e:
            print(f"❌ [Similarity Calculation] Error: {e}")
            similarity_metrics['error'] = str(e)
        
        return similarity_metrics

    def _classify_change_type(self, current_region: np.ndarray, golden_region: np.ndarray, 
                            analysis: Dict) -> str:
        """Classify the type of change detected with golden frame comparison"""
        try:
            # Extract metrics from analysis
            ssim_score = analysis.get('ssim_score', 0)
            orb_match_ratio = analysis.get('orb_match_ratio', 0)
            current_features = analysis.get('current_features', 0)
            golden_features = analysis.get('golden_features', 0)
            hist_correlation = analysis.get('hist_correlation_overall', 0)
            mse = analysis.get('mse', 0)
            
            # Check if region matches golden frame (overkill pass)
            similarity_metrics = self._calculate_region_similarity(current_region, golden_region)
            overall_similarity = similarity_metrics.get('overall_similarity', 0)
            
            # GOLDEN FRAME MATCHING: Skip regions that match golden frame
            if overall_similarity > self.golden_similarity_threshold:
                return "FALSE_POSITIVE_MATCHES_GOLDEN"
            
            # Calculate intensity differences
            current_gray = cv2.cvtColor(current_region, cv2.COLOR_BGR2GRAY)
            golden_gray = cv2.cvtColor(golden_region, cv2.COLOR_BGR2GRAY)
            intensity_diff = abs(np.mean(current_gray) - np.mean(golden_gray))
            
            # Calculate feature presence ratios
            current_feature_ratio = current_features / max(golden_features, 1)
            golden_feature_ratio = golden_features / max(current_features, 1)
            
            # IMPROVED Classification logic
            if current_features == 0 and golden_features > 15:
                return "OBJECT_MISSING"
            elif current_features > 15 and golden_features == 0:
                return "NEW_OBJECT"
            elif current_feature_ratio < 0.2 and orb_match_ratio < 0.15:
                return "MULTIPLE_OBJECTS_MISSING"
            elif current_feature_ratio > 4.0 and orb_match_ratio < 0.15:
                return "MULTIPLE_NEW_OBJECTS"
            elif ssim_score < 0.6 and orb_match_ratio < 0.3:
                return "STRUCTURAL_CHANGE"
            elif hist_correlation < 0.7:
                return "COLOR_APPEARANCE_CHANGE"
            elif orb_match_ratio < 0.5 and intensity_diff > 30:
                return "OBJECT_MOVED"
            elif orb_match_ratio < 0.7:
                return "OBJECT_MODIFIED"
            elif analysis.get('change_score', 0) < self.change_threshold:
                return "MINOR_CHANGE_NO_ACTION"
            else:
                return "UNKNOWN_CHANGE"
                
        except Exception as e:
            return "CLASSIFICATION_ERROR"
    
    def _should_visualize_region(self, analysis: Dict, change_type: str, area: float, 
                               similarity_metrics: Dict) -> bool:
        """Enhanced logic to filter out regions that match golden frame"""
        
        # NEVER visualize regions that match golden frame
        if change_type == "FALSE_POSITIVE_MATCHES_GOLDEN":
            return False
            
        # Extract similarity metrics
        overall_similarity = similarity_metrics.get('overall_similarity', 0)
        ssim_score = similarity_metrics.get('ssim', 0)
        feature_match_ratio = similarity_metrics.get('feature_match_ratio', 0)
        
        # HIGH CONFIDENCE: Region matches golden frame (false positive)
        if (overall_similarity > 0.95 or 
            (ssim_score > 0.95 and feature_match_ratio > 0.85)):
            return False  # Skip visualization - overkill pass
        
        # MEDIUM CONFIDENCE: Very similar to golden frame
        if (overall_similarity > self.golden_similarity_threshold or
            (ssim_score > 0.90 and feature_match_ratio > 0.75)):
            return False  # Skip visualization - likely false positive
        
        # OBJECT-SPECIFIC CHECKS: Always visualize these (unless they match golden frame)
        if change_type in ["OBJECT_MISSING", "NEW_OBJECT", "MULTIPLE_OBJECTS_MISSING", "MULTIPLE_NEW_OBJECTS"]:
            return True
        
        # STRUCTURAL CHANGES: Always visualize
        if change_type == "STRUCTURAL_CHANGE":
            return True
        
        # For other changes, use threshold-based approach
        change_score = analysis.get('change_score', 0)
        if change_score > self.change_threshold:
            return True
        
        # Area-based filtering for minor changes
        if area > self.min_change_area * 2:
            return True
        
        return False

    def _analyze_anomaly_areas_with_golden(self, current_frame: np.ndarray, 
                                        anomaly_areas: List[Dict],
                                        segmentation_map: Optional[np.ndarray] = None) -> Dict:
        """
        OPTIMIZED: Only process regions where anomalies are actually detected
        """
        if not self.enable_golden_comparison or self.golden_image is None or not anomaly_areas:
            return {
                'anomaly_regions': [],
                'total_anomaly_area': 0,
                'change_metrics': {},
                'timing_breakdown': {},
                'significant_changes_found': False,
                'false_positives_filtered': 0
            }
        
        analysis = {
            'anomaly_regions': [],
            'total_anomaly_area': 0,
            'change_metrics': {},
            'timing_breakdown': {},
            'significant_changes_found': False,
            'false_positives_filtered': 0
        }
        
        region_start_time = time.time()
        
        try:
            current_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            
            # PRE-COMPUTE: Only compute features once for the entire frame
            feature_start = time.time()
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            current_kp, current_desc = self.orb.detectAndCompute(current_gray, None)
            analysis['timing_breakdown']['feature_extraction_ms'] = float((time.time() - feature_start) * 1000)
            
            processed_regions = 0
            for i, area in enumerate(anomaly_areas):
                # EARLY EXIT: If we've already found significant changes and don't need to process more
                if processed_regions > 10:  # Limit to max 10 regions for performance
                    break
                    
                contour_start_time = time.time()
                
                # Extract bounding box from anomaly area
                x, y, w, h = area['bbox']
                area_size = area['area']
                
                if area_size < self.min_change_area:  # Filter small regions
                    continue
                
                # Get bounding box with padding - but limit padding for performance
                padding = min(self.comparison_padding, 10)  # Reduce padding for speed
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(current_frame.shape[1], x + w + padding)
                y2 = min(current_frame.shape[0], y + h + padding)
                
                # Extract region from current frame
                current_region = current_frame[y1:y2, x1:x2]
                
                # Scale coordinates to golden image dimensions
                scale_x = self.golden_w / current_frame.shape[1]
                scale_y = self.golden_h / current_frame.shape[0]
                
                golden_x1 = int(x1 * scale_x)
                golden_y1 = int(y1 * scale_y)
                golden_x2 = int(x2 * scale_x)
                golden_y2 = int(y2 * scale_y)
                
                # Ensure coordinates are within golden image bounds
                golden_x1 = max(0, golden_x1)
                golden_y1 = max(0, golden_y1)
                golden_x2 = min(self.golden_w, golden_x2)
                golden_y2 = min(self.golden_h, golden_y2)
                
                # Extract corresponding region from golden image
                golden_region = self.golden_image[golden_y1:golden_y2, golden_x1:golden_x2]
                
                if golden_region.size == 0 or current_region.size == 0:
                    continue
                
                # RESIZE OPTIMIZATION: Resize to smaller size for faster processing
                target_size = (100, 100)  # Smaller size for faster processing
                current_region_small = cv2.resize(current_region, target_size)
                golden_region_small = cv2.resize(golden_region, target_size)
                
                # FAST SIMILARITY CHECK: Use simple histogram comparison first (fastest)
                similarity_start = time.time()
                
                # Method 1: Fast histogram comparison (quick rejection)
                hist_current = cv2.calcHist([current_region_small], [0], None, [32], [0, 256])
                hist_golden = cv2.calcHist([golden_region_small], [0], None, [32], [0, 256])
                hist_corr = cv2.compareHist(hist_current, hist_golden, cv2.HISTCMP_CORREL)
                
                # QUICK REJECTION: If histograms are very similar, skip detailed analysis
                if hist_corr > 0.95 and not self.show_all_anomalies:
                    change_type = "FALSE_POSITIVE_MATCHES_GOLDEN"
                    should_visualize = False
                else:
                    # Method 2: Only compute SSIM if needed (slower but more accurate)
                    try:
                        from skimage.metrics import structural_similarity as ssim
                        current_gray_small = cv2.cvtColor(current_region_small, cv2.COLOR_BGR2GRAY)
                        golden_gray_small = cv2.cvtColor(golden_region_small, cv2.COLOR_BGR2GRAY)
                        ssim_score, _ = ssim(current_gray_small, golden_gray_small, full=True)
                        
                        # COMBINED SIMILARITY SCORE
                        overall_similarity = (ssim_score * 0.7 + max(0, hist_corr) * 0.3)
                        
                        # Classify change type based on simple thresholds
                        if overall_similarity > self.golden_similarity_threshold:
                            change_type = "FALSE_POSITIVE_MATCHES_GOLDEN"
                            should_visualize = self.show_all_anomalies
                        elif ssim_score < 0.7:
                            change_type = "STRUCTURAL_CHANGE"
                            should_visualize = True
                        elif hist_corr < 0.8:
                            change_type = "COLOR_APPEARANCE_CHANGE" 
                            should_visualize = True
                        else:
                            change_type = "MINOR_CHANGE"
                            should_visualize = False
                            
                    except Exception as e:
                        change_type = "COMPARISON_ERROR"
                        should_visualize = True
                        overall_similarity = 0.0
                
                similarity_time = time.time() - similarity_start
                
                # Track statistics
                if should_visualize:
                    analysis['significant_changes_found'] = True
                elif change_type == "FALSE_POSITIVE_MATCHES_GOLDEN":
                    analysis['false_positives_filtered'] += 1
                
                region_info = {
                    'region_id': i,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'area': int(area_size),
                    'analysis': {
                        'similarity_score': float(overall_similarity) if 'overall_similarity' in locals() else float(hist_corr),
                        'histogram_correlation': float(hist_corr),
                        'change_type': change_type
                    },
                    'similarity_metrics': {
                        'overall_similarity': float(overall_similarity) if 'overall_similarity' in locals() else float(hist_corr),
                        'histogram_correlation': float(hist_corr)
                    },
                    'change_type': change_type,
                    'should_visualize': should_visualize,
                    'golden_bbox': (int(golden_x1), int(golden_y1), int(golden_x2), int(golden_y2)),
                    'timing': {
                        'similarity_ms': float(similarity_time * 1000),
                        'total_ms': float((time.time() - contour_start_time) * 1000)
                    }
                }
                
                analysis['anomaly_regions'].append(region_info)
                analysis['total_anomaly_area'] += int(area_size)
                processed_regions += 1
            
            # Calculate overall change metrics
            if analysis['anomaly_regions']:
                analysis['change_metrics'] = self._calculate_change_metrics(analysis['anomaly_regions'])
            
            analysis['timing_breakdown']['region_analysis_total_ms'] = float((time.time() - region_start_time) * 1000)
            analysis['timing_breakdown']['regions_processed'] = processed_regions
            
        except Exception as e:
            print(f"❌ [Region Analysis] Error: {e}")
        
        return analysis
    
    def _compare_regions(self, current_region: np.ndarray, current_region_rgb: np.ndarray,
                        golden_region: np.ndarray, anomaly_segmentation: np.ndarray) -> Dict:
        """Compare current region with golden image region"""
        comparison = {}
        
        try:
            # Convert to grayscale for some analyses
            current_gray = cv2.cvtColor(current_region, cv2.COLOR_BGR2GRAY)
            golden_gray = cv2.cvtColor(golden_region, cv2.COLOR_BGR2GRAY)
            
            # 1. Structural Similarity (SSIM)
            from skimage.metrics import structural_similarity as ssim
            ssim_score, ssim_diff = ssim(current_gray, golden_gray, full=True)
            comparison['ssim_score'] = float(ssim_score)
            
            # 2. Mean Squared Error
            mse = np.mean((current_gray.astype(float) - golden_gray.astype(float)) ** 2)
            comparison['mse'] = float(mse)
            
            # 3. Feature matching
            feature_comparison = self._compare_features_optimized(current_region_rgb, golden_region)
            comparison.update(feature_comparison)
            
            # 4. Color histogram comparison
            hist_comparison = self._compare_histograms_simple(current_region, golden_region)
            comparison.update(hist_comparison)
            
            # 5. Anomaly intensity in this region
            if anomaly_segmentation is not None and anomaly_segmentation.size > 0:
                anomaly_intensity = np.max(anomaly_segmentation) if anomaly_segmentation.size > 0 else 0
                comparison['anomaly_intensity'] = float(anomaly_intensity)
            
            # 6. Overall change score (combined metric)
            change_score = self._calculate_change_score(comparison)
            comparison['change_score'] = float(change_score)
            comparison['significant_change'] = bool(change_score > self.change_threshold)
            
        except Exception as e:
            print(f"❌ [Region Comparison] Error: {e}")
            comparison['error'] = str(e)
        
        return comparison
    
    def _compare_features_optimized(self, current_region: np.ndarray, golden_region: np.ndarray) -> Dict:
        """Optimized feature comparison for speed"""
        feature_results = {}
        
        try:
            current_gray = cv2.cvtColor(current_region, cv2.COLOR_BGR2GRAY)
            golden_gray = cv2.cvtColor(golden_region, cv2.COLOR_BGR2GRAY)
            
            # Use existing orb detector
            kp_current, desc_current = self.orb.detectAndCompute(current_gray, None)
            kp_golden, desc_golden = self.orb.detectAndCompute(golden_gray, None)
            
            if desc_current is not None and desc_golden is not None and len(desc_current) > 0 and len(desc_golden) > 0:
                # Faster matcher with crossCheck
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(desc_current, desc_golden)
                
                # Calculate match quality
                good_matches = [m for m in matches if m.distance < 50]
                match_ratio = len(good_matches) / max(len(desc_current), len(desc_golden), 1)
                
                feature_results.update({
                    'orb_matches': int(len(good_matches)),
                    'orb_match_ratio': float(match_ratio),
                    'current_features': int(len(kp_current)),
                    'golden_features': int(len(kp_golden))
                })
            else:
                # Handle case where no features are found
                feature_results.update({
                    'orb_matches': 0,
                    'orb_match_ratio': 0.0,
                    'current_features': int(len(kp_current) if kp_current else 0),
                    'golden_features': int(len(kp_golden) if kp_golden else 0)
                })
            
        except Exception as e:
            print(f"❌ [Feature Comparison] Error: {e}")
        
        return feature_results
    
    def _compare_histograms_simple(self, current_region: np.ndarray, golden_region: np.ndarray) -> Dict:
        """Simplified histogram comparison for speed"""
        hist_results = {}
        
        try:
            # Use simpler histogram with fewer bins for speed
            hist_current = cv2.calcHist([current_region], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
            hist_golden = cv2.calcHist([golden_region], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
            
            cv2.normalize(hist_current, hist_current, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_golden, hist_golden, 0, 1, cv2.NORM_MINMAX)
            
            overall_correlation = cv2.compareHist(hist_current, hist_golden, cv2.HISTCMP_CORREL)
            hist_results['hist_correlation_overall'] = float(overall_correlation)
            
        except Exception as e:
            print(f"❌ [Histogram Comparison] Error: {e}")
        
        return hist_results
    
    def _calculate_change_score(self, comparison: Dict) -> float:
        """Calculate overall change score from multiple metrics"""
        weights = {
            'ssim_score': 0.3,
            'mse': 0.2,
            'orb_match_ratio': 0.3,
            'hist_correlation_overall': 0.2
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in comparison:
                if metric == 'ssim_score':
                    # Lower SSIM = more change
                    score += (1 - comparison[metric]) * weight
                elif metric == 'mse':
                    # Higher MSE = more change (normalized)
                    normalized_mse = min(comparison[metric] / 10000.0, 1.0)
                    score += normalized_mse * weight
                elif metric == 'orb_match_ratio':
                    # Lower match ratio = more change
                    score += (1 - comparison[metric]) * weight
                elif metric == 'hist_correlation_overall':
                    # Lower correlation = more change
                    score += (1 - max(comparison[metric], 0)) * weight
                
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_change_metrics(self, regions: List[Dict]) -> Dict:
        """Calculate overall change metrics across all regions"""
        if not regions:
            return {}
        
        metrics = {
            'total_regions': len(regions),
            'regions_with_significant_changes': 0,
            'false_positives_filtered': 0,
            'average_change_score': 0.0,
            'max_change_score': 0.0,
            'total_change_area': 0,
            'change_types': {}
        }
        
        total_score = 0.0
        max_score = 0.0
        significant_changes = 0
        false_positives = 0
        total_area = 0
        change_type_counts = {}
        
        for region in regions:
            analysis = region.get('analysis', {})
            change_score = analysis.get('change_score', 0.0)
            change_type = region.get('change_type', 'UNKNOWN')
            
            total_score += change_score
            max_score = max(max_score, change_score)
            total_area += region.get('area', 0)
            
            if analysis.get('significant_change', False):
                significant_changes += 1
            
            if change_type == "FALSE_POSITIVE_MATCHES_GOLDEN":
                false_positives += 1
            
            # Count change types
            change_type_counts[change_type] = change_type_counts.get(change_type, 0) + 1
        
        metrics.update({
            'regions_with_significant_changes': int(significant_changes),
            'false_positives_filtered': int(false_positives),
            'average_change_score': float(total_score / len(regions)),
            'max_change_score': float(max_score),
            'total_change_area': int(total_area),
            'change_types': change_type_counts
        })
        
        return metrics
    
    def _create_visualization(self, current_frame: np.ndarray, 
                            analysis: Dict, anomaly_score: float,
                            original_anomaly_areas: List[Dict] = None) -> np.ndarray:
        """Create enhanced visualization with change classification"""
        if not self.enable_visualization:
            return current_frame
        
        viz_frame = current_frame.copy()
        h, w = viz_frame.shape[:2]
        
        # Determine final status based on anomaly score AND actual changes found
        has_significant_changes = analysis.get('significant_changes_found', False)
        false_positives_filtered = analysis.get('false_positives_filtered', 0)
        is_anomaly_by_score = anomaly_score >= self.anomaly_infer.threshold
        
        # IMPROVED STATUS LOGIC with show_all_anomalies flag
        if self.show_all_anomalies and is_anomaly_by_score:
            status = "ANOMALY (All Shown)"
            status_color = (0, 165, 255)  # Orange for forced display
            should_draw_boxes = True
        elif is_anomaly_by_score and not has_significant_changes and false_positives_filtered > 0:
            status = "PASS (False Positives Filtered)"
            status_color = (0, 255, 255)  # Yellow for pass
            should_draw_boxes = False
        elif is_anomaly_by_score and has_significant_changes:
            status = "ANOMALY"
            status_color = (0, 0, 255)  # Red for anomaly
            should_draw_boxes = True
        else:
            status = "NORMAL" 
            status_color = (0, 255, 0)  # Green for normal
            should_draw_boxes = False
        
        # Draw regions based on should_visualize flag AND show_all_anomalies flag
        visualized_regions = 0
        if should_draw_boxes:
            for region in analysis.get('anomaly_regions', []):
                # NEW: Show all if flag is set, otherwise use original logic
                if self.show_all_anomalies or region.get('should_visualize', False):
                    x1, y1, x2, y2 = region['bbox']
                    region_analysis = region.get('analysis', {})
                    similarity_score = region_analysis.get('similarity_score', 0.0)
                    change_type = region.get('change_type', 'UNKNOWN')
                    
                    # Color code based on change type
                    color_map = {
                        'FALSE_POSITIVE_MATCHES_GOLDEN': (128, 128, 128),  # Gray for false positives
                        'OBJECT_MISSING': (0, 0, 255),        # Red
                        'NEW_OBJECT': (255, 0, 0),            # Blue
                        'OBJECT_MOVED': (0, 255, 255),        # Yellow
                        'OBJECT_MODIFIED': (255, 255, 0),     # Cyan
                        'STRUCTURAL_CHANGE': (0, 165, 255),   # Orange
                        'COLOR_APPEARANCE_CHANGE': (255, 0, 255),  # Magenta
                        'MINOR_CHANGE': (128, 128, 0),        # Olive for minor changes
                    }
                    
                    color = color_map.get(change_type, (128, 128, 128))  # Gray for unknown
                    thickness = 2
                    
                    # Draw bounding box
                    cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Add change type and similarity label
                    label = f"{change_type}: {similarity_score:.2f}"
                    cv2.putText(viz_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    visualized_regions += 1
        
        # Rest of the visualization code remains the same...
        # Add analysis information overlay
        change_metrics = analysis.get('change_metrics', {})
        
        # Top-left: Status 
        cv2.putText(viz_frame, status, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Top-right: Timing and processing info
        timing_info = f"Change Analysis: {analysis.get('timing_breakdown', {}).get('region_analysis_total_ms', 0):.0f}ms"
        cv2.putText(viz_frame, timing_info, (w - 300, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Bottom-left: Detailed metrics
        y_offset = h - 10
        metrics_text = [
            f"Anomaly Score: {anomaly_score:.1f}",
            f"Visualized Regions: {visualized_regions}",
            f"False Positives Filtered: {false_positives_filtered}",
        ]
        
        # Show timing breakdown if available
        timing_breakdown = analysis.get('timing_breakdown', {})
        if 'feature_extraction_ms' in timing_breakdown:
            metrics_text.append(f"Feature Extraction: {timing_breakdown['feature_extraction_ms']:.0f}ms")
        if 'regions_processed' in timing_breakdown:
            metrics_text.append(f"Regions Processed: {timing_breakdown['regions_processed']}")
        
        for text in reversed(metrics_text):
            cv2.putText(viz_frame, text, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset -= 20
        
        return viz_frame

    # In the process_frame method, modify the multi-ROI path to handle temporal buffer issues:
    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Dict:
        """Enhanced frame processing with golden image comparison for all model types"""
        total_start_time = time.time()
        timing_breakdown = {}
        
        # Time anomaly detection
        anomaly_start = time.time()
        
        # Use unified processing for all model types
        if self.multi_mgr is None:
            # Single-detector path
            result = self.anomaly_infer.process_single_frame(frame)
            anomaly_score = result.anomaly_score
            is_anom = result.is_anomaly
            anomaly_areas = result.anomaly_areas
            segmentation_map = result.segmentation_map
            original_overlay = result.processed_frame
        else:
            # Multi-ROI path - FIXED: Clear temporal buffers to prevent shape inconsistencies
            try:
                # Clear temporal buffers to prevent shape mismatches
                if hasattr(self.multi_mgr, '_roi_buffers'):
                    for buffer in self.multi_mgr._roi_buffers.values():
                        buffer.clear()
                
                # Also clear temporal buffers in child detectors
                for det in self.multi_mgr.detectors:
                    child_infer = det['infer']
                    if hasattr(child_infer, 'temporal_buffer'):
                        child_infer.temporal_buffer.clear()
                    if hasattr(child_infer, '_sf_last_segmentation'):
                        child_infer._sf_last_segmentation = None
                
                result = self.multi_mgr.process_frame_all(frame)
                anomaly_score = result.anomaly_score
                is_anom = result.is_anomaly
                anomaly_areas = result.anomaly_areas
                segmentation_map = None
                original_overlay = result.processed_frame
            except Exception as e:
                print(f"❌ Multi-ROI processing failed: {e}")
                # Fallback to single detector if multi-ROI fails
                result = self.anomaly_infer.process_single_frame(frame)
                anomaly_score = result.anomaly_score
                is_anom = result.is_anomaly
                anomaly_areas = result.anomaly_areas
                segmentation_map = result.segmentation_map
                original_overlay = result.processed_frame
                
        timing_breakdown['anomaly_detection_ms'] = float((time.time() - anomaly_start) * 1000)
    
        
        # Perform golden image comparison if enabled
        change_analysis = {}
        if self.enable_golden_comparison and self.golden_image is not None:
            change_start = time.time()
            change_analysis = self._analyze_anomaly_areas_with_golden(
                frame, anomaly_areas, segmentation_map
            )
            timing_breakdown['change_analysis_ms'] = float((time.time() - change_start) * 1000)
        
        # Determine final anomaly status based on both score and actual changes
        has_significant_changes = change_analysis.get('significant_changes_found', False)
        false_positives_filtered = change_analysis.get('false_positives_filtered', 0)
        final_is_anomaly = is_anom and has_significant_changes
        
        # Create enhanced visualization
        viz_start = time.time()
        if self.enable_visualization:
            enhanced_overlay = self._create_visualization(
                frame, change_analysis, anomaly_score, anomaly_areas
            )
        else:
            enhanced_overlay = original_overlay
        timing_breakdown['visualization_ms'] = float((time.time() - viz_start) * 1000)
        
        # Calculate total time
        total_time = time.time() - total_start_time
        timing_breakdown['total_processing_ms'] = float(total_time * 1000)
        
        # Prepare results
        results = {
            'frame_index': int(frame_idx),
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anom),  # Original anomaly detection
            'final_is_anomaly': bool(final_is_anomaly),  # Final judgment after change analysis
            'has_significant_changes': bool(has_significant_changes),
            'false_positives_filtered': int(false_positives_filtered),
            'processing_time': float(total_time),
            'timing_breakdown': timing_breakdown,
            'change_analysis': self._convert_to_serializable(change_analysis),
            'anomaly_areas': anomaly_areas,  # Include original anomaly areas
            'overlay_frame': enhanced_overlay
            
        }
        
        # Save results if enabled
        if self.save_comparison_results:
            self._save_results(results, frame_idx)
        # self._print_change_summary()
        return results

    def process_frame_with_golden_comparison(self, frame: np.ndarray, frame_idx: int = 0) -> AnomalyDetectionResult:
        """
        Process frame with golden comparison and return AnomalyDetectionResult
        Works with all model types (full frame, single ROI, multi-ROI)
        FIXED: Proper coordinate handling for multi-ROI mode
        """
        results = self.process_frame(frame, frame_idx)
        
        # Convert to AnomalyDetectionResult format with filtered areas
        filtered_anomaly_areas = []
        for area in results['anomaly_areas']:
            # Check if this area passed golden comparison
            should_include = True
            if self.enable_golden_comparison and self.golden_image is not None:
                # Find corresponding region in change analysis
                x, y, w, h = area['bbox']
                for region in results['change_analysis'].get('anomaly_regions', []):
                    region_bbox = region['bbox']
                    if (abs(region_bbox[0] - x) < 10 and abs(region_bbox[1] - y) < 10 and
                        abs(region_bbox[2] - (x + w)) < 10 and abs(region_bbox[3] - (y + h)) < 10):
                        should_include = region.get('should_visualize', True)
                        break
            
            if should_include:
                filtered_anomaly_areas.append(area)
        
        return AnomalyDetectionResult(
            processed_frame=results['overlay_frame'],
            anomaly_score=results['anomaly_score'],
            is_anomaly=results['final_is_anomaly'],
            anomaly_areas=filtered_anomaly_areas,
            segmentation_map=None  # Not available after golden comparison
        )
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to JSON-serializable types"""
        if obj is None:
            return None
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def _save_results(self, results: Dict, frame_idx: int):
        """Save analysis results with detailed timing"""
        try:
            output_dir = Path(self.anomaly_infer.output_path)
            output_dir.mkdir(exist_ok=True)
            
            # Save visualization
            if self.enable_visualization and 'overlay_frame' in results and results['overlay_frame'] is not None:
                viz_path = output_dir / f"frame_{frame_idx:06d}_analysis.jpg"
                cv2.imwrite(str(viz_path), results['overlay_frame'])
                print(f"💾 Enhanced visualization saved to: {viz_path}")
            
            # Save JSON results
            json_path = output_dir / f"frame_{frame_idx:06d}_analysis.json"
            
            # Create detailed results with timing
            detailed_results = {
                'frame_index': results['frame_index'],
                'anomaly_score': results['anomaly_score'],
                'original_anomaly': results['is_anomaly'],
                'final_anomaly': results['final_is_anomaly'],
                'has_significant_changes': results['has_significant_changes'],
                'false_positives_filtered': results['false_positives_filtered'],
                'processing_time_seconds': results['processing_time'],
                'timing_breakdown_ms': results['timing_breakdown'],
                'change_analysis': results['change_analysis'],
                'anomaly_areas': results['anomaly_areas']
            }
            
            with open(json_path, 'w') as f:
                json.dump(detailed_results, f, indent=2, default=str)
                
            print(f"📄 Detailed analysis with timing saved to: {json_path}")
            
            # Print timing summary
            self._print_timing_summary(results['timing_breakdown'])
            
            # Print change detection summary
            self._print_change_summary(results)
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def _print_timing_summary(self, timing: Dict):
        """Print timing breakdown summary"""
        print("⏱️  Timing Breakdown:")
        for task, time_ms in timing.items():
            task_name = task.replace('_', ' ').title()
            print(f"   - {task_name}: {time_ms:.1f}ms")
    
    def _print_change_summary(self, results: Dict):
        """Print change detection summary"""
        print("🔍 Change Detection Summary:")
        print(f"   - Anomaly Score: {results['anomaly_score']:.3f}")
        print(f"   - Original Detection: {'ANOMALY' if results['is_anomaly'] else 'NORMAL'}")
        print(f"   - Final Judgment: {'ANOMALY' if results['final_is_anomaly'] else 'PASS' if results['is_anomaly'] else 'NORMAL'}")
        print(f"   - Significant Changes Found: {results['has_significant_changes']}")
        print(f"   - False Positives Filtered: {results['false_positives_filtered']}")
        
        if results['change_analysis']:
            change_metrics = results['change_analysis'].get('change_metrics', {})
            if change_metrics:
                print(f"   - Total Change Regions: {change_metrics.get('total_regions', 0)}")
                print(f"   - Significant Change Regions: {change_metrics.get('regions_with_significant_changes', 0)}")
                
                change_types = change_metrics.get('change_types', {})
                if change_types:
                    print(f"   - Change Types:")
                    for change_type, count in change_types.items():
                        if change_type != "FALSE_POSITIVE_MATCHES_GOLDEN":
                            readable_type = change_type.replace('_', ' ').title()
                            print(f"     * {readable_type}: {count}")

    def set_golden_image(self, image_path: str):
        """Set golden image after initialization"""
        self.enable_golden_comparison = True
        return self.load_golden_image(image_path)
    
    def update_comparison_parameters(self, padding: Optional[int] = None, 
                                   change_threshold: Optional[float] = None,
                                   golden_similarity_threshold: Optional[float] = None,
                                   min_change_area: Optional[int] = None):
        """Update comparison parameters"""
        if padding is not None:
            self.comparison_padding = padding
        if change_threshold is not None:
            self.change_threshold = change_threshold
        if golden_similarity_threshold is not None:
            self.golden_similarity_threshold = golden_similarity_threshold
        if min_change_area is not None:
            self.min_change_area = min_change_area

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.anomaly_infer, 'cleanup'):
            self.anomaly_infer.cleanup()
        if self.multi_mgr and hasattr(self.multi_mgr, 'cleanup'):
            self.multi_mgr.cleanup()


# Unified interface for all model types with golden comparison
def create_enhanced_anomaly_detector(model_path: str, 
                                   golden_image_path: Optional[str] = None,
                                   **kwargs) -> EnhancedAnomalyInference:
    """
    Create enhanced anomaly detector with golden image comparison
    Works with all model types: full frame, single ROI, multi-ROI
    """
    detector = EnhancedAnomalyInference(
        model_path=model_path,
        enable_golden_comparison=golden_image_path is not None,
        golden_image_path=golden_image_path,
        **kwargs
    )
    return detector


# Example usage for different model types
def example_usage():
    """Example usage for different model types"""
    
    # 1. Full Frame Model
    print("🚀 Testing Full Frame Model with Golden Comparison")
    full_frame_detector = create_enhanced_anomaly_detector(
        model_path="/path/to/full_frame_model",
        golden_image_path="/path/to/golden_image.jpg",
        threshold=0.5,
        enable_visualization=True
    )
    
    # 2. Single ROI Model  
    print("🚀 Testing Single ROI Model with Golden Comparison")
    single_roi_detector = create_enhanced_anomaly_detector(
        model_path="/path/to/single_roi_model", 
        golden_image_path="/path/to/golden_image.jpg",
        threshold=0.5,
        enable_visualization=True
    )
    
    # 3. Multi-ROI Model
    print("🚀 Testing Multi-ROI Model with Golden Comparison")
    multi_roi_detector = create_enhanced_anomaly_detector(
        model_path="/path/to/multi_roi_model",
        golden_image_path="/path/to/golden_image.jpg", 
        threshold=0.5,
        enable_visualization=True
    )
    
    # Process frame with any detector
    frame = cv2.imread("test_frame.jpg")
    
    # Method 1: Get detailed results
    results = full_frame_detector.process_frame(frame)
    print(f"Anomaly Score: {results['anomaly_score']}")
    print(f"Final Judgment: {'ANOMALY' if results['final_is_anomaly'] else 'NORMAL'}")
    
    # Method 2: Get AnomalyDetectionResult with filtered areas
    result_obj = full_frame_detector.process_frame_with_golden_comparison(frame)
    print(f"Filtered Anomaly Areas: {len(result_obj.anomaly_areas)}")
    
    return full_frame_detector


if __name__ == "__main__":
    example_usage()
