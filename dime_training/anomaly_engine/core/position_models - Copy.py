# anomaly_engine/core/position_models.py
# FAISS per‑position loader (no fallbacks). Matches the train pipeline that saves
# models/<dataset>/rank*/pos*/index.faiss (+projector.npz, meta.json).

import os
import glob
import json
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from anomaly_engine.core.anomaly_net import AnomalyNet
from anomaly_engine.models import network_models
from anomaly_engine.selectors.feature_selector import IdentitySelector
import anomaly_engine.core.core_utils as core_utils


__all__ = ["PositionAwareAnomalyNet"]


def _split_layers_for_backbones(backbone_names, layers_to_extract_from):
    if len(backbone_names) > 1:
        coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            lyr = ".".join(layer.split(".")[1:])
            coll[idx].append(lyr)
        return coll
    else:
        return [layers_to_extract_from]


class PositionAwareAnomalyNet:
    """
    Lazily loads a **FAISS-backed** AnomalyNet per tile position.

    Expected on-disk layout (written by train.py):
        <models_root>/<dataset>/rank*/pos{P}/index.faiss
        <models_root>/<dataset>/rank*/pos{P}/projector.npz   (optional)
        <models_root>/<dataset>/rank*/pos{P}/meta.json       (optional)

    Also tolerated (flat, no ranks):
        <models_root>/<dataset>/pos{P}/index.faiss
        <models_root>/pos{P}/index.faiss

    No single‑model fallbacks. If a required position is missing, we raise.
    """

    def __init__(
        self,
        base_cfg: dict,
        models_root: str,
        device: Union[str, torch.device] = "cuda",
        dataset_name: Optional[str] = None,
    ) -> None:
        self.base_cfg = dict(base_cfg or {})
        self.models_root = models_root
        self.dataset_name = dataset_name  # if None, we search recursively under models_root
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self._pos_models: Dict[int, AnomalyNet] = {}

        if not self.models_root or not os.path.isdir(self.models_root):
            raise FileNotFoundError(f"models_root does not exist: {self.models_root}")

    # -------------------------
    # Filesystem helpers
    # -------------------------
    def _pos_dir(self, pos_id: int) -> str:
        # Prefer dataset-scoped
        if self.dataset_name:
            root = os.path.join(self.models_root, self.dataset_name)
            # rank-scoped first
            for d in sorted(glob.glob(os.path.join(root, "rank*"))):
                cand = os.path.join(d, f"pos{pos_id}")
                if os.path.isdir(cand):
                    return cand
            # flat under dataset
            cand = os.path.join(root, f"pos{pos_id}")
            if os.path.isdir(cand):
                return cand
        # Fallback: search recursively under models_root
        for path in sorted(glob.glob(os.path.join(self.models_root, "**", f"pos{pos_id}"), recursive=True)):
            if os.path.isdir(path):
                return path
        raise FileNotFoundError(
            f"No model directory for pos{pos_id} under {self.models_root}"
            + (f"/{self.dataset_name}" if self.dataset_name else "")
        )

    def _index_path(self, pos_dir: str) -> str:
        path = os.path.join(pos_dir, "nnscorer_search_index.faiss")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing FAISS index at: {path}")
        return path

    def _load_projector(self, pos_dir: str):
        pz = os.path.join(pos_dir, "projector.npz")
        if os.path.isfile(pz):
            return dict(np.load(pz))
        return {}

    # -------------------------
    # Model construction & FAISS attach
    # -------------------------
    def _build_model_for_infer(self, tile_hw: Tuple[int, int]) -> AnomalyNet:
        # Ensure 3 input channels for CNN backbones (e.g., ResNet family)
        h, w = int(tile_hw[0]), int(tile_hw[1])
        input_shape = (3, h, w)  # (C,H,W)

        anomaly_cfg = self.base_cfg
        backbone_names = anomaly_cfg.get("backbone_names", ["wideresnet50"])  # default
        layers_all = anomaly_cfg.get("layers_to_extract_from", ["layer2", "layer3"])
        layers_coll = _split_layers_for_backbones(backbone_names, layers_all)

        # NOTE: We construct a single model (first backbone) for inference here.
        # If you trained an ensemble, instantiate multiple and average externally.
        backbone_name = backbone_names[0]
        layers_to_extract_from = layers_coll[0]

        # Seeded backbones (optional naming pattern: "name.seed-<n>")
        backbone_seed = None
        if ".seed-" in backbone_name:
            backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(backbone_name.split("-")[-1])
        backbone = network_models.load(backbone_name)
        backbone.name, backbone.seed = backbone_name, backbone_seed

        # Identity selector and FAISS proximity searcher
        selector = IdentitySelector()
        nn_method = core_utils.ApproximateProximitySearcher(
            anomaly_cfg.get("proximity_on_gpu", True),
            anomaly_cfg.get("proximity_num_workers", 12),
        )

        model = AnomalyNet(self.device)
        model.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=self.device,
            input_shape=input_shape,
            pretrain_embed_dimension=anomaly_cfg.get("pretrain_embed_dimension", 512),
            target_embed_dimension=anomaly_cfg.get("target_embed_dimension", 512),
            patchsize=anomaly_cfg.get("feature_window", 2),
            window_step=anomaly_cfg.get("window_step", 1),
            featuresampler=selector,
            anomaly_scorer_num_nn=anomaly_cfg.get("anomaly_scorer_num_nn", 3),
            nn_method=nn_method,
            motion_config={"enable": False},
            sequence_config={"enable": False},
            object_detection_config={"enable": False},
        )
        model.eval()
        return model

    def _attach_faiss(self, model: AnomalyNet, model_dir: str):
        idx_path = self._index_path(model_dir)
        attached = False
        # Try nn_method then scorer, then model-level path
        obj = getattr(model, "nn_method", None)
        if obj is not None and hasattr(obj, "load"):
            obj.load(idx_path, device=self.device)
            attached = True
        if not attached:
            scorer = getattr(model, "anomaly_scorer", None)
            if scorer is not None and hasattr(scorer, "load"):
                scorer.load(idx_path, device=self.device)
                attached = True
        if not attached and hasattr(model, "load_from_path"):
            model.load_from_path(model_dir)
            attached = True
        if not attached:
            raise RuntimeError(f"Could not attach FAISS index from {idx_path}")

    # -------------------------
    # Public API
    # -------------------------
    def load_position(self, pos_id: int) -> AnomalyNet:
        if pos_id in self._pos_models:
            return self._pos_models[pos_id]

        # Build fresh model with correct tile geometry, then attach FAISS
        # Use a dummy size; we will rebuild per actual tile in infer path when needed
        model = self._build_model_for_infer(tile_hw=(512, 512))
        self._attach_faiss(model, self._pos_dir(pos_id))
        self._pos_models[pos_id] = model
        return model

    @torch.no_grad()
    def infer_tile(self, pos_id: int, tile_tensor: torch.Tensor):
        """
        Run inference for a single tile tensor of shape [1, C, H, W].
        Ensures C=3 by channel replication if grayscale.
        Returns a dict: {"heatmap": HxW float32, "score": float}
        """
        if tile_tensor.ndim != 4 or tile_tensor.shape[0] != 1:
            raise ValueError(f"Expected tile tensor shape [1, C, H, W], got {tuple(tile_tensor.shape)}")

        # Make sure channels match backbone (3)
        if tile_tensor.shape[1] == 1:
            tile_tensor = tile_tensor.repeat(1, 3, 1, 1)
        elif tile_tensor.shape[1] != 3:
            raise ValueError(f"Backbone expects 1 or 3 channels, got C={tile_tensor.shape[1]}")

        # (Re)build a model with the exact tile geometry so feature shapes align
        h, w = int(tile_tensor.shape[-2]), int(tile_tensor.shape[-1])
        model = self._build_model_for_infer((h, w))
        self._attach_faiss(model, self._pos_dir(pos_id))

        scores, maps = model._predict(tile_tensor.to(self.device), None)
        heat = maps[0]
        if isinstance(heat, torch.Tensor):
            heat = heat.detach().cpu().numpy().astype(np.float32)
        score = float(scores[0]) if isinstance(scores, (list, tuple, np.ndarray)) else float(scores)
        return {"heatmap": heat, "score": score}
