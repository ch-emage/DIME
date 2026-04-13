"""Motion processing utilities for enhanced anomaly detection."""
import torch
import torch.nn as nn
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MotionProcessor(nn.Module):
    """Process motion features for anomaly detection"""
    
    def __init__(self, input_channels=1, output_dim=128):
        super(MotionProcessor, self).__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        
        # Convolutional layers for motion feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Final projection layer
        self.projection = nn.Linear(128, output_dim)
    
    def forward(self, motion_features):
        """
        Process motion features
        Args:
            motion_features: Tensor of shape (batch_size, channels, height, width)
        Returns:
            Processed motion features of shape (batch_size, output_dim)
        """
        batch_size = motion_features.shape[0]
        
        # Pass through convolutional layers
        x = self.conv_layers(motion_features)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Project to output dimension
        x = self.projection(x)
        
        return x

class OpticalFlowProcessor:
    """Optical flow processing for motion detection"""
    
    def __init__(self, method="farneback"):
        self.method = method
        self.prev_frame = None
        
    def compute_optical_flow(self, frame):
        """
        Compute optical flow between consecutive frames
        Args:
            frame: Current frame (BGR)
        Returns:
            Flow magnitude map
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros_like(gray, dtype=np.float32)
        
        if self.method == "farneback":
            flow = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, gray, 
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            if flow is not None and len(flow) == 3:
                flow_vectors = flow[0]
                status = flow[1]
                
                # Create flow magnitude map
                flow_mag = np.zeros(gray.shape, dtype=np.float32)
                
                if flow_vectors is not None and status is not None:
                    # Filter good points
                    good_points = status.ravel() == 1
                    
                    if np.any(good_points):
                        # Calculate magnitude of motion vectors
                        dx = flow_vectors[good_points, 0]
                        dy = flow_vectors[good_points, 1]
                        magnitude = np.sqrt(dx*dx + dy*dy)
                        
                        # Create a simple flow magnitude map
                        # This is a simplified approach - you might want to improve this
                        flow_mag = cv2.GaussianBlur(
                            np.ones(gray.shape, dtype=np.float32) * np.mean(magnitude),
                            (15, 15), 0
                        )
            else:
                flow_mag = np.zeros_like(gray, dtype=np.float32)
                
        else:
            # Fallback: simple frame difference
            flow_mag = cv2.absdiff(self.prev_frame.astype(np.float32), 
                                 gray.astype(np.float32))
        
        self.prev_frame = gray
        return flow_mag
    
    def reset(self):
        """Reset the flow processor"""
        self.prev_frame = None

def extract_motion_features(frame, prev_frame=None, method="farneback"):
    """
    Extract motion features from frames
    Args:
        frame: Current frame
        prev_frame: Previous frame (optional)
        method: Motion detection method
    Returns:
        Motion feature tensor
    """
    if prev_frame is None:
        # Return zeros if no previous frame
        h, w = frame.shape[:2]
        return torch.zeros(1, 1, h, w)
    
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_curr = frame
    
    if len(prev_frame.shape) == 3:
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_prev = prev_frame
    
    if method == "farneback":
        # Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        # Calculate magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    elif method == "diff":
        # Simple frame difference
        magnitude = cv2.absdiff(gray_prev.astype(np.float32), 
                              gray_curr.astype(np.float32))
    else:
        # Default to frame difference
        magnitude = cv2.absdiff(gray_prev.astype(np.float32), 
                              gray_curr.astype(np.float32))
    
    # Normalize
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()
    
    # Convert to tensor
    motion_tensor = torch.from_numpy(magnitude).float().unsqueeze(0).unsqueeze(0)
    
    return motion_tensor