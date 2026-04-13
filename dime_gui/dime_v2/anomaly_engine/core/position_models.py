from __future__ import annotations

import os
import glob
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from anomaly_engine.core.anomaly_net import AnomalyNet
from anomaly_engine.models import network_models
from anomaly_engine.selectors.feature_selector import IdentitySelector
from anomaly_engine.core.core_utils import ProximitySearcher


__all__ = ["PositionAwareAnomalyNet"]

# Used only to locate the correct posK folder; AnomalyRater.load() constructs this internally.
_INDEX_FILENAME = "nnscorer_search_index.faiss"


class PositionAwareAnomalyNet:
    """
    Per-position AnomalyNet cache with FAISS index attached via AnomalyRater.
    - One model per pos_id
    - Rebuild only when tile (H, W) changes
    - Keep anomaly-net logic untouched
    """

    def __init__(
        self,
        base_cfg: dict,
        models_root: str,
        device: Union[str, torch.device] = "cuda",
        dataset_name: Optional[str] = None,
    ) -> None:
        self.base_cfg = dict(base_cfg or {})
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

        # If dataset_name is provided, look under models_root/dataset_name
        self.models_root = os.path.join(models_root, dataset_name) if dataset_name else models_root

        self._pos_models: Dict[int, AnomalyNet] = {}
        self._pos_hw: Dict[int, Tuple[int, int]] = {}

        if not os.path.isdir(self.models_root):
            raise FileNotFoundError(f"[PositionAwareAnomalyNet] models_root does not exist: {self.models_root}")

    # --------------------- filesystem helpers ---------------------

    def _pos_dir(self, pos_id: int) -> str:
        """
        Find a directory .../pos{pos_id}/ that contains the FAISS index.
        Works with layouts like:
          <root>/rank*/posK/nnscorer_search_index.faiss
          <root>/posK/nnscorer_search_index.faiss
        """
        pattern = os.path.join(self.models_root, "**", f"pos{pos_id}", _INDEX_FILENAME)
        hits = glob.glob(pattern, recursive=True)
        if not hits:
            raise FileNotFoundError(
                f"[PositionAwareAnomalyNet] FAISS index not found for pos{pos_id} under {self.models_root}."
            )
        return os.path.dirname(hits[0])

    def _load_projector(self, pos_dir: str) -> dict:
        """Optionally load projector.npz (mean/std/proj_W/proj_b). Non-fatal if missing/corrupt."""
        pz = os.path.join(pos_dir, "projector.npz")
        if os.path.isfile(pz):
            try:
                return dict(np.load(pz))
            except Exception:
                return {}
        return {}

    # --------------------- build + attach ---------------------

    def _build_model_for_infer(self, tile_hw: Tuple[int, int]) -> AnomalyNet:
        """
        Build and configure AnomalyNet for a given (H, W).
        Matches your AnomalyNet API EXACTLY:
          AnomalyNet(device) + .load(backbone, layers_to_extract_from, device, input_shape, ...)
        """
        h, w = map(int, tile_hw)
        cfg = self.base_cfg

        backbone_name = cfg.get("backbone_name") or cfg.get("backbone_names", ["wideresnet50"])[0]
        layers = list(cfg.get("layers_to_extract_from", ["layer2"]))
        pretrain_embed_dimension = int(cfg.get("pretrain_embed_dimension", 512))
        target_embed_dimension   = int(cfg.get("target_embed_dimension", 512))
        feature_window           = int(cfg.get("feature_window", 3))
        window_step              = int(cfg.get("window_step", 1))
        anomaly_score_num_nn     = int(cfg.get("anomaly_score_num_nn", cfg.get("anomaly_scorer_num_nn", 1)))
        motion_config            = cfg.get("motion_config", None)
        sequence_config          = cfg.get("sequence_config", None)
        object_detection_config  = cfg.get("object_detection_config", None)

        # Backbone (your network_models.load takes only 'name')
        backbone = network_models.load(backbone_name).to(self.device)

        # Feature sampler and NN method per your defaults
        featuresampler = IdentitySelector()
        nn_method = ProximitySearcher(on_gpu=False, num_workers=4)

        # NOTE: your AnomalyNet requires device in __init__
        model = AnomalyNet(self.device)
        model.load(
            backbone=backbone,
            layers_to_extract_from=layers,
            device=self.device,
            input_shape=(3, h, w),
            pretrain_embed_dimension=pretrain_embed_dimension,
            target_embed_dimension=target_embed_dimension,
            feature_window=feature_window,
            window_step=window_step,
            anomaly_score_num_nn=anomaly_score_num_nn,
            featuresampler=featuresampler,
            nn_method=nn_method,  # stored inside model.anomaly_rater
            motion_config=motion_config,
            sequence_config=sequence_config,
            object_detection_config=object_detection_config,
        )
        if hasattr(model, "set_input_size"):
            model.set_input_size((h, w))
        return model

    def _attach_faiss(self, model: AnomalyNet, pos_dir: str) -> None:
        """
        Attach FAISS by delegating to AnomalyRater (correct for your repo):
          model.anomaly_rater.load(<folder>)
        This reads '<folder>/nnscorer_search_index.faiss' into the searcher.
        """
        # projector (optional)
        for k, v in self._load_projector(pos_dir).items():
            try:
                setattr(model, k, v)
            except Exception:
                pass  # optional; ignore if model lacks these attrs

        # attach index via AnomalyRater (this calls nn_method.load internally)
        if not hasattr(model, "anomaly_rater") or not hasattr(model.anomaly_rater, "load"):
            raise AttributeError("[PositionAwareAnomalyNet] Model has no anomaly_rater.load; cannot attach FAISS index.")
        model.anomaly_rater.load(pos_dir)

    # --------------------- public API ---------------------

    def load_position(self, pos_id: int) -> Optional[AnomalyNet]:
        return self._pos_models.get(pos_id, None)

    @torch.no_grad()
    def infer_tile(self, pos_id: int, tile_tensor: torch.Tensor):
        """
        Run inference for a single tile tensor of shape [1, C, H, W].
        Returns: {"heatmap": HxW float32, "score": float}
        """
        if tile_tensor.ndim != 4 or tile_tensor.shape[0] != 1:
            raise ValueError(f"[PositionAwareAnomalyNet] Expected [1,C,H,W], got {tuple(tile_tensor.shape)}")

        # Channel normalization
        c = tile_tensor.shape[1]
        if c == 1:
            tile_tensor = tile_tensor.repeat(1, 3, 1, 1)
        elif c == 4:
            tile_tensor = tile_tensor[:, :3, :, :]
        elif c != 3:
            raise ValueError(f"[PositionAwareAnomalyNet] Expected 1/3/4 channels, got C={c}")

        h, w = int(tile_tensor.shape[-2]), int(tile_tensor.shape[-1])

        model = self.load_position(pos_id)
        if model is None or self._pos_hw.get(pos_id) != (h, w):
            pos_dir = self._pos_dir(pos_id)
            model = self._build_model_for_infer((h, w))
            self._attach_faiss(model, pos_dir)
            self._pos_models[pos_id] = model
            self._pos_hw[pos_id] = (h, w)

        scores, masks = model._predict(tile_tensor.to(self.device), None)  # returns ([scores], [masks])
        heat = masks[0]
        if isinstance(heat, torch.Tensor):
            heat = heat.detach().cpu().numpy().astype(np.float32)
        score = float(scores[0]) if isinstance(scores, (list, tuple, np.ndarray)) else float(scores)
        return {"heatmap": heat, "score": score}
