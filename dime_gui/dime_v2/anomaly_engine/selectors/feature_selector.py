import abc
from typing import Union
import numpy as np
import torch
import tqdm

class IdentitySelector:
    def run(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Returns features unchanged."""
        return features

class BaseSelector(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)

class GreedyCoresetSelector(BaseSelector):
    def __init__(self, percentage: float, device: torch.device, dimension_to_project_features_to=128):
        """Greedy coreset selection for feature sampling."""
        super().__init__(percentage)
        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        
        # Memory-efficient processing for DDP
        mapper = torch.nn.Linear(features.shape[1], self.dimension_to_project_features_to, bias=False)
        mapper = mapper.to(self.device)
        
        # Process in chunks to avoid large memory allocation
        chunk_size = min(1000, features.shape[0])  # Process in smaller chunks
        results = []
        
        for i in range(0, features.shape[0], chunk_size):
            chunk = features[i:i+chunk_size].to(self.device)
            with torch.no_grad():  # Reduce memory usage
                result_chunk = mapper(chunk)
                results.append(result_chunk.cpu())  # Move back to CPU immediately
            
            # Clear GPU cache between chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate results on CPU, then move final result to device
        final_result = torch.cat(results, dim=0)
        return final_result.to(self.device)

    def run(self, features: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        select_indices = self._compute_greedy_coreset_indices(reduced_features)
        features = features[select_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)
        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)
            coreset_select_distance = distance_matrix[:, select_idx:select_idx + 1]
            coreset_anchor_distances = torch.cat([coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1)
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)

class ApproximateGreedyCoresetSelector(GreedyCoresetSelector):
    def __init__(self, percentage: float, device: torch.device, number_of_starting_points: int = 10, dimension_to_project_features_to: int = 128):
        """Approximate greedy coreset selection."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        number_of_starting_points = np.clip(self.number_of_starting_points, None, len(features))
        start_points = np.random.choice(len(features), number_of_starting_points, replace=False).tolist()
        approximate_distance_matrix = self._compute_batchwise_differences(features, features[start_points])
        approximate_coreset_anchor_distances = torch.mean(approximate_distance_matrix, axis=-1).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Selecting features..."):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(features, features[select_idx:select_idx + 1])
                approximate_coreset_anchor_distances = torch.cat([approximate_coreset_anchor_distances, coreset_select_distance], dim=-1)
                approximate_coreset_anchor_distances = torch.min(approximate_coreset_anchor_distances, dim=1).values.reshape(-1, 1)

        return np.array(coreset_indices)