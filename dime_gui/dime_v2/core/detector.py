"""
Simplified detector wrapper that exposes only essential functionality
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any
import os

from ..enhanced_inference import EnhancedAnomalyInference


class DimeDetector:
    """
    Simplified wrapper around EnhancedAnomalyInference
    Exposes only the essential functionality needed by the API
    """
    
    def __init__(self, 
                 model_path: str,
                 threshold: Optional[float] = None,
                 enable_golden_comparison: bool = False,
                 golden_image_path: Optional[str] = None,
                 enable_movement_compensation: bool = False,
                 **kwargs):
        
        # Default parameters optimized for performance
        default_params = {
            'threshold': threshold,
            'skip_frames': 3,
            'mask_alpha': 0.5,
            'tile_rows': 2,
            'tile_cols': 2,
            'tile_overlap': 0.1,
            'parallel_tiles': True,
            'num_workers': 4,
            'enable_visualization': False,  # Disable by default for API
            'output_path': 'dime_output',
            
            # Enhanced features
            'enable_golden_comparison': enable_golden_comparison,
            'golden_image_path': golden_image_path,
            'enable_movement_compensation': enable_movement_compensation,
            'movement_threshold': 5.0,
            'max_movement': 50.0,
        }
        
        # Override defaults with user-provided kwargs
        default_params.update(kwargs)
        
        # Initialize the enhanced inference engine
        self.engine = EnhancedAnomalyInference(
            model_path=model_path,
            **default_params
        )
        
        # Expose important attributes for API access
        self.anomaly_infer = self.engine.anomaly_infer
        self.golden_image = self.engine.golden_image
        self.multi_mgr = self.engine.multi_mgr
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame"""
        return self.engine.process_frame(frame)
    
    def set_golden_image(self, image_path: str) -> bool:
        """Set golden image from file"""
        return self.engine.set_golden_image(image_path)
    
    def set_golden_image_from_frame(self, frame: np.ndarray) -> bool:
        """Set golden image from numpy array"""
        return self.engine.set_golden_image_from_frame(frame)
    
    def cleanup(self):
        """Clean up resources"""
        self.engine.cleanup()