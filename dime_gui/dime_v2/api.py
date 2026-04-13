"""
Dime_v2 - Anomaly Detection Library
Simple API for frame-by-frame anomaly detection with coordinate extraction
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json
from pathlib import Path

from .core.detector import DimeDetector


class DimeAnomalyDetector:
    """
    Main API class for Dime_v2 anomaly detection
    
    Simple interface:
    - Initialize once with model path
    - Process frames with process_frame()
    - Get anomaly status and coordinates
    """
    
    def __init__(self, 
                 model_path: str,
                 threshold: Optional[float] = None,
                 enable_golden_comparison: bool = False,
                 golden_image_path: Optional[str] = None,
                 enable_movement_compensation: bool = False,
                 **kwargs):
        """
        Initialize Dime_v2 anomaly detector
        
        Args:
            model_path: Path to trained model directory
            threshold: Anomaly detection threshold (auto-detected if None)
            enable_golden_comparison: Use golden image for false positive reduction
            golden_image_path: Path to golden reference image
            enable_movement_compensation: Compensate for camera movement
            **kwargs: Additional parameters passed to underlying detector
        """
        
        # Store configuration
        self.model_path = model_path
        self.threshold = threshold
        self.enable_golden_comparison = enable_golden_comparison
        self.golden_image_path = golden_image_path
        self.enable_movement_compensation = enable_movement_compensation
        
        # Initialize the detector
        self.detector = DimeDetector(
            model_path=model_path,
            threshold=threshold,
            enable_golden_comparison=enable_golden_comparison,
            golden_image_path=golden_image_path,
            enable_movement_compensation=enable_movement_compensation,
            **kwargs
        )
        
        print(f"✅ Dime_v2 initialized with model: {model_path}")
        if enable_golden_comparison and golden_image_path:
            print(f"   Using golden image: {golden_image_path}")
        if enable_movement_compensation:
            print("   Camera movement compensation: ENABLED")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame for anomaly detection
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Dictionary containing:
            - is_anomaly: bool (True if anomaly detected)
            - anomaly_areas: list of dicts with bounding boxes and metadata
            - anomaly_score: float anomaly confidence score
            - processing_time_ms: float processing time in milliseconds
            - frame_with_overlay: np.ndarray (optional, with visualizations)
        """
        # Convert BGR to RGB if needed (detector expects BGR)
        # Our existing code works with BGR, so no conversion needed
        
        # Process frame
        result = self.detector.process_frame(frame)
        
        # Format the response
        response = {
            'is_anomaly': result['final_is_anomaly'] if 'final_is_anomaly' in result else result['is_anomaly'],
            'anomaly_areas': result.get('anomaly_areas', []),
            'anomaly_score': float(result['anomaly_score']),
            'processing_time_ms': float(result.get('processing_time', 0) * 1000),
            'detector_type': 'multi_roi' if hasattr(self.detector, 'multi_mgr') and self.detector.multi_mgr else 'single',
            'frame_index': result.get('frame_index', 0)
        }
        
        # Add frame with overlay if available
        if 'overlay_frame' in result and result['overlay_frame'] is not None:
            response['frame_with_overlay'] = result['overlay_frame']
        
        # Extract simplified coordinates if requested
        if response['is_anomaly'] and response['anomaly_areas']:
            response['coordinates'] = self._extract_coordinates(response['anomaly_areas'])
        else:
            response['coordinates'] = []
        
        return response
    
    def _extract_coordinates(self, anomaly_areas: List[Dict]) -> List[Dict]:
        """Extract simplified coordinate information from anomaly areas"""
        coordinates = []
        
        for area in anomaly_areas:
            if 'bbox' in area:
                x, y, w, h = area['bbox']
                coord = {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'confidence': float(area.get('confidence', 0)),
                    'area': int(area.get('area', 0))
                }
                
                # Add centroid if available
                if 'centroid' in area:
                    coord['centroid'] = {
                        'x': int(area['centroid'][0]),
                        'y': int(area['centroid'][1])
                    }
                
                # Add ROI name for multi-ROI mode
                if 'roi_name' in area:
                    coord['roi_name'] = area['roi_name']
                
                coordinates.append(coord)
        
        return coordinates
    
    def process_video(self, video_path: str, output_path: Optional[str] = None, 
                     max_frames: Optional[int] = None) -> List[Dict]:
        """
        Process a video file and return anomaly results for each frame
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save results JSON
            max_frames: Optional limit on number of frames to process
            
        Returns:
            List of results for each processed frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        results = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_idx >= max_frames:
                break
            
            # Process frame
            result = self.process_frame(frame)
            result['video_frame_index'] = frame_idx
            results.append(result)
            
            frame_idx += 1
            
            # Show progress
            if frame_idx % 100 == 0:
                print(f"📊 Processed {frame_idx} frames...")
        
        cap.release()
        
        # Save results if output path provided
        if output_path:
            self._save_video_results(results, output_path, video_path)
        
        print(f"✅ Video processing complete: {len(results)} frames processed")
        return results
    
    def _save_video_results(self, results: List[Dict], output_path: str, video_path: str):
        """Save video processing results to JSON file"""
        summary = {
            'video_file': video_path,
            'total_frames': len(results),
            'anomaly_frames': sum(1 for r in results if r['is_anomaly']),
            'max_anomaly_score': max((r['anomaly_score'] for r in results), default=0),
            'average_processing_time_ms': np.mean([r['processing_time_ms'] for r in results]),
            'frame_results': results
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        summary = convert_to_serializable(summary)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
    
    def set_golden_image(self, image_path: str) -> bool:
        """Set golden image after initialization"""
        return self.detector.set_golden_image(image_path)
    
    def set_golden_image_from_frame(self, frame: np.ndarray) -> bool:
        """Set golden image from numpy frame"""
        return self.detector.set_golden_image_from_frame(frame)
    
    def update_threshold(self, new_threshold: float):
        """Update anomaly detection threshold"""
        self.detector.anomaly_infer.threshold = new_threshold
        print(f"📊 Threshold updated to: {new_threshold}")
    
    def get_detector_info(self) -> Dict:
        """Get information about the loaded detector"""
        info = {
            'model_path': self.model_path,
            'threshold': float(self.detector.anomaly_infer.threshold),
            'detector_type': 'multi_roi' if hasattr(self.detector, 'multi_mgr') and self.detector.multi_mgr else 'single',
            'has_golden_image': self.detector.golden_image is not None,
            'movement_compensation_enabled': self.enable_movement_compensation
        }
        
        if hasattr(self.detector, 'multi_mgr') and self.detector.multi_mgr:
            info['roi_count'] = len(self.detector.multi_mgr.detectors)
            info['roi_names'] = [det['name'] for det in self.detector.multi_mgr.detectors]
        
        return info
    
    def cleanup(self):
        """Clean up resources"""
        self.detector.cleanup()
        print("🧹 Dime_v2 resources cleaned up")


# Convenience function for quick initialization
def create_detector(model_path: str, **kwargs) -> DimeAnomalyDetector:
    """
    Create a DimeAnomalyDetector instance
    
    Example:
        detector = dime_v2.create_detector('/path/to/model')
        result = detector.process_frame(frame)
    """
    return DimeAnomalyDetector(model_path, **kwargs)