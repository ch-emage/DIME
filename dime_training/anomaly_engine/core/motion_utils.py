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

class MotionStats:
    def __init__(self):
        self.values = []

    def update(self, flow_mag):
        self.values.append(float(flow_mag))

    def finalize(self):
        import numpy as np
        return {
            "mean": float(np.mean(self.values)),
            "std":  float(np.std(self.values) + 1e-6)
        }



class MotionSpeedAnalyzer:
    """Analyze motion speed for linear conveyor belt scenarios"""
    
    def __init__(self, method="farneback", roi_mask=None):
        self.method = method
        self.prev_frame = None
        self.prev_flow = None
        self.roi_mask = roi_mask
        self.speed_history = []
        self.max_history = 100
        self.speed_threshold_factor = 0.3  # 30% slowdown threshold
        
    def compute_flow_speed(self, current_frame, roi=None):
        """
        Compute flow and calculate average speed in the ROI
        Returns: avg_speed, flow_mag, flow_angle
        """
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray_current
            return 0.0, np.zeros_like(gray_current), np.zeros_like(gray_current)
        
        # Compute optical flow
        if self.method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame, gray_current, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
        else:
            # Lucas-Kanade method
            feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            lk_params = dict(winSize=(15, 15), maxLevel=2,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            
            p0 = cv2.goodFeaturesToTrack(self.prev_frame, mask=None, **feature_params)
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, gray_current, p0, None, **lk_params)
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    flow_vectors = good_new - good_old
                    # Convert sparse to dense flow (simplified)
                    flow = np.zeros((gray_current.shape[0], gray_current.shape[1], 2), dtype=np.float32)
                    for (x, y), (dx, dy) in zip(good_old.astype(int), flow_vectors):
                        if 0 <= x < flow.shape[1] and 0 <= y < flow.shape[0]:
                            flow[y, x] = [dx, dy]
                else:
                    flow = np.zeros((gray_current.shape[0], gray_current.shape[1], 2), dtype=np.float32)
            else:
                flow = np.zeros((gray_current.shape[0], gray_current.shape[1], 2), dtype=np.float32)
        
        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Apply ROI mask if provided
        if roi is not None:
            mask = np.zeros_like(magnitude)
            x, y, w, h = roi
            mask[y:y+h, x:x+w] = 1
            magnitude = magnitude * mask
        
        # Calculate average speed in ROI
        if roi is not None:
            x, y, w, h = roi
            roi_magnitude = magnitude[y:y+h, x:x+w]
            avg_speed = np.mean(roi_magnitude) if roi_magnitude.size > 0 else 0.0
        else:
            avg_speed = np.mean(magnitude)
        
        # Store for next frame
        self.prev_frame = gray_current
        self.prev_flow = flow
        
        return avg_speed, magnitude, angle
    
    def detect_speed_anomaly(self, current_speed, min_samples=10):
        """
        Detect if current speed is anomalous (too slow)
        Returns: is_anomalous, anomaly_score
        """
        # Add to history
        self.speed_history.append(current_speed)
        if len(self.speed_history) > self.max_history:
            self.speed_history.pop(0)
        
        # Need enough samples for baseline
        if len(self.speed_history) < min_samples:
            return False, 0.0
        
        # Calculate baseline (normal speed)
        baseline_speed = np.percentile(self.speed_history, 75)  # Use 75th percentile as normal
        
        # Calculate slowdown ratio
        if baseline_speed > 0:
            slowdown_ratio = 1.0 - (current_speed / baseline_speed)
        else:
            slowdown_ratio = 0.0
        
        # Check if slowdown exceeds threshold
        is_anomalous = slowdown_ratio > self.speed_threshold_factor
        
        return is_anomalous, slowdown_ratio
    
    def reset(self):
        """Reset analyzer"""
        self.prev_frame = None
        self.prev_flow = None
        self.speed_history = []

def extract_motion_features_with_speed(frame, prev_frame=None, method="farneback", roi=None):
    """
    Extract motion features including speed analysis
    """
    # Create analyzer instance
    analyzer = MotionSpeedAnalyzer(method)
    
    if prev_frame is None:
        analyzer.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return {
            'motion_features': torch.zeros(1, 1, frame.shape[0], frame.shape[1]),
            'speed': 0.0,
            'is_slow': False,
            'slowdown_ratio': 0.0
        }
    
    # Compute flow and speed
    avg_speed, magnitude, angle = analyzer.compute_flow_speed(frame, roi)
    
    # Detect speed anomaly
    is_slow, slowdown_ratio = analyzer.detect_speed_anomaly(avg_speed)
    
    # Create motion tensor
    motion_tensor = torch.from_numpy(magnitude).float().unsqueeze(0).unsqueeze(0)
    
    return {
        'motion_features': motion_tensor,
        'speed': avg_speed,
        'is_slow': is_slow,
        'slowdown_ratio': slowdown_ratio,
        'flow_magnitude': magnitude,
        'flow_angle': angle
    }