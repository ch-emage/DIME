import torch
import torch.nn.functional as F

class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        """Preprocesses features for anomaly detection."""
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)

class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)

class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        """Aggregates features to a target dimension."""
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)

class FeatureEnhancer(torch.nn.Module):
    def __init__(self):
        """Placeholder for feature enhancement."""
        super(FeatureEnhancer, self).__init__()

    def forward(self, features):
        return features  # No-op for now