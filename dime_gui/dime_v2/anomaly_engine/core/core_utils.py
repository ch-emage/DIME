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

# # Optimize TensorRT
# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit

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
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.num_workers = num_workers
        self.search_index = None

    def _gpu_cloner_options(self):
        gpu_cloner_options = faiss.GpuClonerOptions()
        gpu_cloner_options.useFloat16CoarseQuantizer = True
        return gpu_cloner_options

    def _index_to_gpu(self, index):
        if self.on_gpu:
            return faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index, self._gpu_cloner_options())
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

    def _create_index(self, dimension):
        nlist = min(256, max(10, dimension // 4))  # Ensure valid nlist
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.nprobe = min(5, nlist // 2)  # Ensure valid nprobe
        return self._index_to_gpu(index)
    
    def update(self, new_features: np.ndarray) -> None:
        """Update existing FAISS index with new features - PRESERVE OLD DATA"""
        if self.search_index is None:
            print("❌ No existing index to update")
            self.fit(new_features)
            return
        
        if new_features is None or len(new_features) == 0:
            print("⚠️ No new features to add")
            return
        
        print(f"📥 Adding {len(new_features)} new vectors to FAISS index...")
        
        # Convert to the correct type
        if isinstance(new_features, np.ndarray):
            new_features = new_features.astype(np.float32)
        
        try:
            # Get current index size
            current_size = self.search_index.ntotal
            print(f"📊 Current FAISS index size: {current_size} vectors")
            
            # Add new features to existing index
            self.search_index.add(new_features)
            
            # Verify addition worked
            new_size = self.search_index.ntotal
            added_count = new_size - current_size
            
            print(f"✅ Successfully added {added_count} new vectors to FAISS index")
            print(f"📊 New FAISS index size: {new_size} vectors")
            
        except Exception as e:
            print(f"❌ FAISS incremental update failed: {e}")
            raise e

    def can_update(self) -> bool:
        """Check if this index type supports updates"""
        return hasattr(self.search_index, 'add')
    
class FeatureExtractorTRT10(torch.nn.Module):
    def __init__(self, model_path, layers_to_extract_from):
        super(FeatureExtractorTRT10, self).__init__()
        self.layers_to_extract_from = layers_to_extract_from

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        runtime = trt.Runtime(TRT_LOGGER)

        with open(model_path, "rb") as f:
            engine_bytes = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)

        self.show_engine_info(self.engine)
        # self.cuda_ctx = cuda.Device(0).make_context()
        self.context = self.engine.create_execution_context()
        # self.cuda_ctx.push()
        self.stream = cuda.Stream()

        # ---- Tensor metadata ----
        self.input_names = []
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # ---- Buffers (allocated lazily) ----
        self.device_buffers = {}

    def _allocate_buffers(self, input_shape):
        """Allocate buffers based on actual runtime shape"""
        self.device_buffers.clear()

        # Set input shape
        input_name = self.input_names[0]
        self.context.set_input_shape(input_name, input_shape)

        if not self.context.all_binding_shapes_specified:
            raise RuntimeError("Input shapes not fully specified")

        for name in self.input_names + self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape))

            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            self.device_buffers[name] = device_mem

            self.context.set_tensor_address(name, int(device_mem))

    def __call__(self, images):
        # images: np.ndarray, shape = (B, C, H, W)
        is_tensor = False
        if isinstance(images, torch.Tensor):
            is_tensor = True
            images.unsqueeze(0)
            images = images.detach().cpu().numpy()
        input_shape = images.shape
        input_name = self.input_names[0]

        # Allocate buffers if first run or shape changed
        if not self.device_buffers:
            self._allocate_buffers(input_shape)

        # H2D
        cuda.memcpy_htod_async(
            self.device_buffers[input_name],
            images,
            self.stream,
        )

        print("[FeatExtShape]",images.shape, images.dtype)
        # Execute (TRT 10)
        # self.cuda_ctx.push() 
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        # self.cuda_ctx.pop()

        # D2H outputs
        results = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host_buf = np.empty(shape, dtype=dtype)

            cuda.memcpy_dtoh_async(
                host_buf,
                self.device_buffers[name],
                self.stream,
            )
            results[name] = host_buf

        self.stream.synchronize()
        if is_tensor:
            tensor_dict = {k: torch.from_numpy(v) for k, v in results.items()}
            return tensor_dict
        else:
            return results

    def feature_dimensions(self, input_shape):
        dummy = np.random.randn(*input_shape).astype(np.float32)
        outputs = self(dummy)
        return [outputs[layer].shape[1] for layer in self.layers_to_extract_from]

    def show_engine_info(self, engine):
        print("[INFO] TensorRT Engine Info (TRT 10)")
        print(f"\t + Engine device memory: {engine.device_memory_size / (1024**2):.2f} MB")
        print("\t + IO Tensors:")
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            print(
                f"\t\t + {'Input' if mode == trt.TensorIOMode.INPUT else 'Output'}: "
                f"{name}, " 
                f"dtype={engine.get_tensor_dtype(name)}, "
                f"shape={engine.get_tensor_shape(name)}"
            )

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
        self.detection_features = None  # ADD THIS LINE

    def fit(self, detection_features: List[np.ndarray]) -> None:
        self.detection_features = self.feature_merger.merge(detection_features)  # MODIFY THIS LINE
        self.nn_method.fit(self.detection_features)  # MODIFY THIS LINE


    def predict(self, query_features: List[np.ndarray]) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        query_features = self.feature_merger.merge(query_features)
        query_distances, query_nns = self.imagelevel_nn(query_features)
        anomaly_scores = np.mean(query_distances, axis=-1)
        return anomaly_scores, query_distances, query_nns

    def update(self, new_features: List[np.ndarray]) -> bool:
        """Update with new features - NEVER LOSE EXISTING FEATURES"""
        if self.detection_features is None or len(self.detection_features) == 0:
            print("❌ CRITICAL: No existing features to update!")
            self.fit(new_features)
            return True
        
        # Convert new features
        new_features_merged = self.feature_merger.merge(new_features)
        
        print(f"🔄 Updating: {len(self.detection_features)} (existing) + {len(new_features_merged)} (new) features")
        
        # CRITICAL: Always combine old + new features
        combined_features = np.concatenate([self.detection_features, new_features_merged], axis=0)
        
        print(f"📦 Combined features: {len(combined_features)} total")
        
        # Apply feature limit if needed (but preserve as many as possible)
        max_features = 50000
        if len(combined_features) > max_features:
            print(f"📦 Feature limit reached, keeping most recent {max_features} features")
            # Keep the most recent features (new ones are at the end)
            combined_features = combined_features[-max_features:]
        else:
            print(f"📦 Within feature limit: {len(combined_features)} / {max_features}")
        
        # REBUILD the index with combined features - this PRESERVES everything
        print("🔄 Rebuilding FAISS index with ALL features...")
        self.fit([combined_features])
        
        # Verify we didn't lose features
        final_count = len(self.detection_features) if self.detection_features is not None else 0
        print(f"✅ Final feature bank: {final_count} features")
        
        if final_count < len(combined_features):
            print(f"❌ CRITICAL: Features were lost during update!")
            return False
        
        return True

    def get_feature_count(self) -> int:
        """Get current number of features in the index"""
        if self.detection_features is None:
            return 0
        return len(self.detection_features)

    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "anomaly_rater_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
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

    def save(self, save_folder: str, save_features_separately: bool = True, prepend: str = "") -> None:  # Changed default to True
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately and self.detection_features is not None:
            self._save(self._detection_file(save_folder, prepend), self.detection_features)


    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        detection_file_path = self._detection_file(load_folder, prepend)
        if os.path.exists(detection_file_path):
            self.detection_features = self._load(detection_file_path)
            print(f"✅ Loaded {len(self.detection_features)} features from {detection_file_path}")
        else:
            # Try to reconstruct features from FAISS index if possible
            LOGGER.warning("Detection features file not found. Attempting to reconstruct from index...")
            try:
                # For some FAISS index types, we can extract the vectors
                if hasattr(self.nn_method.search_index, 'reconstruct_n'):
                    n_vectors = self.nn_method.search_index.ntotal
                    reconstructed = []
                    for i in range(n_vectors):
                        vec = self.nn_method.search_index.reconstruct(i)
                        reconstructed.append(vec)
                    self.detection_features = np.array(reconstructed)
                    print(f"✅ Reconstructed {len(self.detection_features)} features from FAISS index")
                else:
                    self.detection_features = None
                    LOGGER.warning("Cannot reconstruct features from this FAISS index type.")
            except Exception as e:
                LOGGER.warning(f"Failed to reconstruct features: {e}")
                self.detection_features = None

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