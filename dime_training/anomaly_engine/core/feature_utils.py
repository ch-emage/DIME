import torch
import numpy as np 
class FeatureSlicer:
    def __init__(self, feature_window, stride=None):
        """Slices features into smaller regions for anomaly detection."""
        self.feature_window = feature_window
        self.stride = stride

    def slice_features(self, features, return_spatial_info=False):
        """Converts a tensor into a tensor of feature regions."""
        padding = int((self.feature_window - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.feature_window, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        unfolded_features = unfolded_features * 1.0  # Dummy normalization
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.feature_window - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.feature_window, self.feature_window, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
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
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x