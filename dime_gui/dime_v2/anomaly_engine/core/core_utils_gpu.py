import csv
import logging
import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm
import faiss
from skimage import morphology
from skimage.segmentation import mark_boundaries
from sklearn import metrics
from typing import List, Union
import torch.nn.functional as F
import scipy.ndimage as ndimage
import pickle
import logging
from typing import Tuple, Optional
import cv2


LOGGER = logging.getLogger(__name__)

def plot_anomaly_maps(
    savefolder,
    image_paths,
    anomaly_maps,
    categories,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
):
    """Generate and save anomaly maps with original and superimposed images."""
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    category_stats = {}
    combined_statistics = []

    for category, image_path, anomaly_map, anomaly_score, mask_path in tqdm.tqdm(
        zip(categories, image_paths, anomaly_maps, anomaly_scores, mask_paths),
        total=len(image_paths),
        desc="Generating anomaly maps...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        if len(anomaly_map.shape) == 3:
            anomaly_map = np.mean(anomaly_map, axis=0)

        anomaly_score_raw = np.max(anomaly_map)
        anomaly_score_val = round(anomaly_score_raw, 2)

        if category not in category_stats:
            category_stats[category] = []
        category_stats[category].append(anomaly_score_val)

        result_dir = os.path.join(savefolder)
        os.makedirs(result_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        original_path = os.path.join(result_dir, f"{base_name}_original.png")
        image.save(original_path)

        # heatmap = cv2.applyColorMap(np.float32(255 * anomaly_map), cv2.COLORMAP_VIRIDIS)
        heatmap = cv2.applyColorMap(np.uint8(255 * anomaly_map), cv2.COLORMAP_VIRIDIS)
        heatmap = cv2.resize(heatmap, (image.width, image.height))

        heatmap_path = os.path.join(result_dir, f"{base_name}_anomaly_map.png")
        cv2.imwrite(heatmap_path, heatmap)

        superimposed_image = cv2.addWeighted(heatmap, 0.3, image_np, 0.7, 0)
        superimposed_path = os.path.join(result_dir, f"{base_name}_combined.png")
        cv2.imwrite(superimposed_path, superimposed_image)

    for category, scores in category_stats.items():
        scores_less_than_1 = [score for score in scores if score < 1]
        max_score_less_than_1 = max(scores_less_than_1, default=0)
        avg_score = round(np.mean(scores), 2)
        combined_statistics.append([category, avg_score, max_score_less_than_1])

    return combined_statistics

def compute_optical_flow(prev_frame, current_frame):
    """Compute dense optical flow between consecutive frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    return magnitude, angle

def compute_motion_histogram(magnitude, angle, bins=8):
    """Convert optical flow to motion histogram."""
    hist = np.zeros(bins)
    bin_edges = np.linspace(0, 2*np.pi, bins+1)
    
    for i in range(bins):
        mask = (angle >= bin_edges[i]) & (angle < bin_edges[i+1])
        hist[i] = np.mean(magnitude[mask])
                                        
    return hist / (np.sum(hist) + 1e-6)

def create_storage_folder(main_folder_path, project_folder, group_folder, mode="iterate"):
    """Creates a folder for storing results."""
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)
    return save_path

def set_torch_device(gpu_ids):
    """Sets the appropriate torch device."""
    if len(gpu_ids):
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")

def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixes seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def compute_imagewise_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels):
    """Computes image-level anomaly detection metrics."""
    fpr, tpr, thresholds = metrics.roc_curve(anomaly_ground_truth_labels, anomaly_prediction_weights)
    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}

def compute_pixelwise_metrics(anomaly_segmentations, ground_truth_masks):
    """Computes pixel-level anomaly detection metrics."""
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)
    auroc = metrics.roc_auc_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)

    precision, recall, thresholds = metrics.precision_recall_curve(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }

def compute_custom_score(segmentations):
    """Computes a custom metric (mean intensity) for anomaly maps."""
    return np.mean(segmentations)

class ProximitySearcher:
    def __init__(self, on_gpu: bool = False, num_workers: int = 4) -> None:
        import faiss

        # configure FAISS threading
        faiss.omp_set_num_threads(num_workers)

        self.num_workers = num_workers
        self.search_index = None

        # what the caller *asked* for
        self.on_gpu_requested = bool(on_gpu)

        # check if this faiss build even has GPU helpers
        self._faiss_has_gpu = (
            hasattr(faiss, "StandardGpuResources")
            and (hasattr(faiss, "index_cpu_to_gpu") or hasattr(faiss, "index_cpu_to_all_gpus"))
        )

        # final flag we actually use
        self.on_gpu = self.on_gpu_requested and self._faiss_has_gpu


    def _gpu_cloner_options(self):
        gpu_cloner_options = faiss.GpuClonerOptions()
        gpu_cloner_options.useFloat16CoarseQuantizer = True
        return gpu_cloner_options

    # def _index_to_gpu(self, index):
    #     if self.on_gpu:
    #         return faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index, self._gpu_cloner_options())
    #     return index

    def _index_to_gpu(self, index):
        """
        Try to move a CPU FAISS index to GPU if possible.
        Falls back to CPU index if GPU helpers are missing or fail.
        """
        import faiss

        # if we decided not to use GPU, just return CPU index
        if not self.on_gpu:
            return index

        # Prefer index_cpu_to_gpu if present
        try:
            if hasattr(faiss, "index_cpu_to_gpu"):
                res = faiss.StandardGpuResources()
                return faiss.index_cpu_to_gpu(
                    res,
                    0,  # GPU id
                    index,
                    self._gpu_cloner_options() if hasattr(self, "_gpu_cloner_options") else None,
                )

            # Newer builds (like yours) may only expose index_cpu_to_all_gpus
            if hasattr(faiss, "index_cpu_to_all_gpus"):
                return faiss.index_cpu_to_all_gpus(
                    index,
                    self._gpu_cloner_options() if hasattr(self, "_gpu_cloner_options") else None,
                )

        except Exception as e:
            print(f"[ProximitySearcher] Failed to move index to GPU ({e}); using CPU index instead.")

        # If we reach here, something went wrong → stay on CPU
        self.on_gpu = False
        return index

    def _index_to_cpu(self, index):
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig())
        return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray) -> None:
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self._train(self.search_index, features)
        self.search_index.add(features)

    def _train(self, _index, _features):
        pass

    def run(self, n_nearest_neighbours, query_features: np.ndarray, index_features: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)
        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None

# class ApproximateProximitySearcher(ProximitySearcher):
#     def _train(self, index, features):
#         index.train(features)

#     def _gpu_cloner_options(self):
#         cloner = faiss.GpuClonerOptions()
#         cloner.useFloat16 = True
#         cloner.useFloat16LookupTables = True
#         if hasattr(faiss, 'INDICES_32'):
#             cloner.indicesOptions = faiss.INDICES_32
#         return cloner

#     def _create_index(self, dimension):
#         index = faiss.IndexIVFPQ(faiss.IndexFlatL2(dimension), dimension, 512, 32, 8)
#         return self._index_to_gpu(index)
                
class ApproximateProximitySearcher(ProximitySearcher):
    def _train(self, index, features):
        index.train(features)

    def _gpu_cloner_options(self):
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        cloner.useFloat16LookupTables = True
        if hasattr(faiss, 'INDICES_32'):
            cloner.indicesOptions = faiss.INDICES_32
        return cloner

    # def _create_index(self, dimension):
    #     quantizer = faiss.IndexFlatL2(dimension)
    #     # Reduce nlist (number of clusters) to match your data size
    #     # index = faiss.IndexIVFFlat(quantizer, dimension, min(15, dimension//4))
    #     index = faiss.IndexIVFFlat(quantizer, dimension, min(256, dimension//4))
    #     index.nprobe = 5  # Can also reduce this
    #     return self._index_to_gpu(index)

    def _create_index(self, dimension):
        nlist = min(256, max(10, dimension // 4))  # Ensure valid nlist
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.nprobe = min(5, nlist // 2)  # Ensure valid nprobe
        return self._index_to_gpu(index)

class FeatureExtractor(torch.nn.Module):
    def __init__(self, backbone, layers_to_extract_from, device):
        """Extracts features from specified network layers."""
        super(FeatureExtractor, self).__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}
        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(self.outputs, extract_layer, layers_to_extract_from[-1])
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]
            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(network_layer[-1].register_forward_hook(forward_hook))
            else:
                self.backbone.hook_handles.append(network_layer.register_forward_hook(forward_hook))
        self.to(self.device)

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]

class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = layer_name == last_layer_to_extract

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None

class LastLayerToExtractReachedException(Exception):
    pass

class AnomalyRater:
    def __init__(self, n_nearest_neighbours: int, nn_method=ProximitySearcher(False, 4)) -> None:
        """Rates anomalies based on proximity to known features."""
        self.feature_merger = ConcatMerger()
        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method
        self.imagelevel_nn = lambda query: self.nn_method.run(n_nearest_neighbours, query)
        self.pixelwise_nn = lambda query, index: self.nn_method.run(1, query, index)

    def fit(self, detection_features: List[np.ndarray]) -> None:
        self.detection_features = self.feature_merger.merge(detection_features)
        self.nn_method.fit(self.detection_features)

    def predict(self, query_features: List[np.ndarray]) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        query_features = self.feature_merger.merge(query_features)
        query_distances, query_nns = self.imagelevel_nn(query_features)
        anomaly_scores = np.mean(query_distances, axis=-1)
        return anomaly_scores, query_distances, query_nns

    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "anomaly_rater_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        # nnscorer_search_index.faiss
        # return os.path.join(folder, prepend + "anomaly_rater_index.faiss")
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(self, save_folder: str, save_features_separately: bool = False, prepend: str = "") -> None:
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(self._detection_file(save_folder, prepend), self.detection_features)

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(self._detection_file(load_folder, prepend))

class ConcatMerger:
    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)

    @staticmethod
    def _reduce(features):
        return features.reshape(len(features), -1)

class RescaleSegmentor:
    def __init__(self, device, target_size=224):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, region_scores):
        with torch.no_grad():
            if isinstance(region_scores, np.ndarray):
                region_scores = torch.from_numpy(region_scores)
            _scores = region_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(_scores, size=self.target_size, mode="bilinear", align_corners=False)
            _scores = _scores.squeeze(1)
            region_scores = _scores.cpu().numpy()
        return [ndimage.gaussian_filter(score, sigma=self.smoothing) for score in region_scores]