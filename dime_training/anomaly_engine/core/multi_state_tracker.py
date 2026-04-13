import numpy as np
from collections import defaultdict
from torch import nn 
import torch

class MultiObjectStateTracker:
    def __init__(self, state_dim=128, transition_tolerance=0.1):
        self.state_classifiers = {}
        self.state_embeddings = {}
        self.transition_matrices = {}
        self.current_states = defaultdict(int)
        self.transition_tolerance = transition_tolerance
        self.state_dim = state_dim
        
    def register_object_class(self, class_name, states, transition_matrix):
        """Register a new object class with its states and transitions"""
        # Create state classifier
        classifier = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, len(states))
        )
        
        self.state_classifiers[class_name] = classifier
        self.state_embeddings[class_name] = states
        self.transition_matrices[class_name] = transition_matrix
        
    def update_states(self, detections, features):
        """Update states for all detected objects"""
        anomalies = []
        
        for detection in detections:
            class_name = detection["class"]
            obj_id = self._get_object_id(detection)
            
            if class_name not in self.state_classifiers:
                # Unknown object class - skip
                continue
                
            # Extract features for this object
            obj_features = self._extract_object_features(features, detection["box"])
            
            # Classify state
            state_probs = torch.softmax(
                self.state_classifiers[class_name](obj_features), 
                dim=-1
            )
            
            # Check transition validity
            current_state = self.current_states[obj_id]
            valid_transitions = self.transition_matrices[class_name][current_state]
            valid_mask = (state_probs * valid_transitions) > self.transition_tolerance
            
            # Check for anomaly
            if not torch.any(valid_mask):
                anomalies.append({
                    "object": class_name,
                    "id": obj_id,
                    "previous_state": self.state_embeddings[class_name][current_state],
                    "current_probabilities": state_probs.tolist()
                })
            
            # Update state
            if valid_mask.any():
                new_state = torch.argmax(state_probs * valid_transitions).item()
                self.current_states[obj_id] = new_state
            else:
                # Maintain current state if no valid transition
                pass
                
        return anomalies
        
    def _get_object_id(self, detection):
        """Simple object ID based on position (for demo)"""
        box = detection["box"]
        return f"{detection['class']}-{int(box[0])}-{int(box[1])}"
        
    def _extract_object_features(self, features, box):
        """Extract features for a specific object region"""
        x1, y1, x2, y2 = map(int, box)
        object_features = features[:, :, y1:y2, x1:x2]
        return torch.mean(object_features, dim=(2, 3))  # Global average pooling