import torch
import numpy as np 

class FeatureSlicer:
    """Optimized FeatureSlicer with GPU acceleration and caching."""
    
    def __init__(self, feature_window, stride=None):
        """Slices features into smaller regions for anomaly detection."""
        self.feature_window = feature_window
        self.stride = stride or 1
        self.padding = int((self.feature_window - 1) / 2)
        
        # Cache unfolders to avoid recreation
        self._unfolder_cache = {}
        
    def _get_unfolder(self, device):
        """Get cached unfolder for device."""
        device_key = str(device) if device is not None else "cpu"
        if device_key not in self._unfolder_cache:
            unfolder = torch.nn.Unfold(
                kernel_size=self.feature_window, 
                stride=self.stride, 
                padding=self.padding, 
                dilation=1
            )
            if device is not None and device.type == 'cuda':
                unfolder = unfolder.to(device)
            self._unfolder_cache[device_key] = unfolder
        return self._unfolder_cache[device_key]

    def slice_features(self, features, return_spatial_info=False):
        """Converts a tensor into a tensor of feature regions with optimization."""
        device = features.device if hasattr(features, 'device') else None
        unfolder = self._get_unfolder(device)
        
        # Perform unfolding (optimized to stay on GPU)
        unfolded_features = unfolder(features)
        
        # Calculate number of patches more efficiently
        h, w = features.shape[-2:]
        h_patches = (h + 2 * self.padding - self.feature_window) // self.stride + 1
        w_patches = (w + 2 * self.padding - self.feature_window) // self.stride + 1
        
        # Reshape more efficiently
        batch_size, channels = features.shape[:2]
        unfolded_features = unfolded_features.view(
            batch_size, channels, self.feature_window, self.feature_window, h_patches * w_patches
        ).permute(0, 4, 1, 2, 3)
        
        if return_spatial_info:
            return unfolded_features, (h_patches, w_patches)
        return unfolded_features

    def unslice_scores(self, x, batchsize):
        """Reconstructs scores from sliced regions."""
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        """Computes final anomaly score from region scores."""
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        x = torch.max(torch.max(x, dim=-1).values, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
