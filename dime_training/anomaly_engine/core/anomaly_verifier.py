import torch.nn as nn

class AnomalyVerifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, appearance_feat, motion_feat, anomaly_map, residual):
        # Flatten inputs
        app_flat = appearance_feat.flatten(start_dim=1)
        res_flat = residual.flatten(start_dim=1)
        map_flat = anomaly_map.flatten(start_dim=1)
        
        # Combine features
        if motion_feat is not None:
            mot_flat = motion_feat.flatten(start_dim=1)
            combined = torch.cat([app_flat, mot_flat, res_flat, map_flat], dim=1)
        else:
            combined = torch.cat([app_flat, res_flat, map_flat], dim=1)
        
        return self.net(combined)