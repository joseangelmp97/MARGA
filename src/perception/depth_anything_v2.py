# src/perception/depth_anything_v2.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..core.interfaces import DepthEstimator
from ..core.types import FramePacket

ROINorm = Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max) in [0..1]


@dataclass(frozen=True)
class DepthAnythingV2Config:
    repo_dir: str = "models/Depth-Anything-V2"
    encoder: str = "vits"  # 'vits' | 'vitb' | 'vitl' | 'vitg'
    checkpoint_path: str = "models/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"
    device: str = "cpu"  # 'cpu' | 'cuda'
    roi: ROINorm = (0.33, 0.67, 0.20, 0.95)
    roi_percentile: float = 5.0  # 5th percentile: "closest-ish" pixels


class DepthAnythingV2Estimator(DepthEstimator):
    """
    Minimal wrapper for Depth Anything V2.
    Returns a single scalar depth score extracted from the ROI.

    NOTE: Default models output relative depth (not meters).
    """

    def __init__(self, cfg: DepthAnythingV2Config):
        self.cfg = cfg
        self._model = None

        repo_abs = os.path.abspath(cfg.repo_dir)
        if repo_abs not in sys.path:
            sys.path.insert(0, repo_abs)

        import torch
        from depth_anything_v2.dpt import DepthAnythingV2

        self._torch = torch

        # Choose device
        if cfg.device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        if not os.path.exists(cfg.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint_path}")

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
        }

        if cfg.encoder not in model_configs:
            raise ValueError(f"Unsupported encoder: {cfg.encoder}")

        model = DepthAnythingV2(**model_configs[cfg.encoder])
        state = torch.load(cfg.checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        model = model.to(self.device).eval()

        self._model = model

    def estimate_roi_depth(self, packet: FramePacket) -> Optional[float]:
        if self._model is None:
            return None

        depth = self._model.infer_image(packet.frame)  # numpy HxW
        if depth is None:
            return None

        h, w = depth.shape[:2]
        x_min, x_max, y_min, y_max = self.cfg.roi

        x1 = int(x_min * w)
        x2 = int(x_max * w)
        y1 = int(y_min * h)
        y2 = int(y_max * h)

        roi_depth = depth[y1:y2, x1:x2]
        if roi_depth.size == 0:
            return None

        return float(np.percentile(roi_depth, self.cfg.roi_percentile))

    def close(self) -> None:
        self._model = None
