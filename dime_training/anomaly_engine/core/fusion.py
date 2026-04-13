import torch.nn as nn
import torch
class SpatioTemporalFusion(nn.Module):
    def __init__(self, appearance_dim, motion_dim, fused_dim):
        super().__init__()
        self.appearance_proj = nn.Linear(appearance_dim, fused_dim)
        self.motion_proj = nn.Linear(motion_dim, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(appearance_dim + motion_dim, fused_dim),
            nn.Sigmoid()
        )
        
    def forward(self, appearance_feat, motion_feat):
        # Project features to common space
        proj_app = self.appearance_proj(appearance_feat.permute(0, 2, 3, 1))
        proj_mot = self.motion_proj(motion_feat.permute(0, 2, 3, 1))
        
        # Compute gating mechanism
        combined = torch.cat([appearance_feat.permute(0, 2, 3, 1), 
                              motion_feat.permute(0, 2, 3, 1)], dim=-1)
        gate = self.gate(combined)
        
        # Fuse features
        fused = gate * proj_app + (1 - gate) * proj_mot
        return fused.permute(0, 3, 1, 2)