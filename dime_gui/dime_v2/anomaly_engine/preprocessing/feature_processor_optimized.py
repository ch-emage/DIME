import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        """Preprocesses features for anomaly detection with optimizations."""
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        """Optimized forward pass with autocast support."""
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            with autocast('cuda', enabled=feature.device.type == 'cuda'):
                _features.append(module(feature))
        return torch.stack(_features, dim=1)

class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        """Optimized mean mapping with improved memory efficiency."""
        batch_size = features.shape[0]
        # More efficient reshaping
        features_flat = features.view(batch_size, 1, -1)
        return F.adaptive_avg_pool1d(features_flat, self.preprocessing_dim).squeeze(1)

class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        """Aggregates features to a target dimension with optimizations."""
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Optimized aggregation with improved memory efficiency."""
        batch_size = features.shape[0]
        features_flat = features.view(batch_size, 1, -1)
        aggregated = F.adaptive_avg_pool1d(features_flat, self.target_dim)
        return aggregated.view(batch_size, -1)

class FeatureEnhancer(torch.nn.Module):
    def __init__(self):
        """Placeholder for feature enhancement with potential optimizations."""
        super(FeatureEnhancer, self).__init__()

    def forward(self, features):
        """No-op for now, but optimized for potential future enhancements."""
        return features
