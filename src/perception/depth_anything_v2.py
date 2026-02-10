# src/perception/depth_anything_v2.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

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
    obstacle_norm_threshold: float = 0.2  # normalized threshold (0..1) below which we flag an obstacle


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
        print(f"[DEBUG] Depth map shape: {depth.shape}, ROI pixels: x[{x_min*w:.0f}:{x_max*w:.0f}], y[{y_min*h:.0f}:{y_max*h:.0f}]")

        x1 = int(x_min * w)
        x2 = int(x_max * w)
        y1 = int(y_min * h)
        y2 = int(y_max * h)

        roi_depth = depth[y1:y2, x1:x2]
        if roi_depth.size == 0:
            return None

        print(f"[DEBUG] ROI depth values (sample): {roi_depth.flatten()[:10]}")

        return float(np.percentile(roi_depth, self.cfg.roi_percentile))

    def visualize_depth(self,
                        packet: FramePacket,
                        window_name: str = "Depth Map",
                        alpha: float = 0.6,
                        annotate_roi: bool = True,
                        block: bool = True,
                        save_path: Optional[str] = None) -> None:
        """Show heatmap overlay of the depth map and draw ROI rectangle on the image."""
        if self._model is None:
            return

        depth = self._model.infer_image(packet.frame)  # numpy HxW
        if depth is None:
            return

        # Normalize depth to 0-255 for colormap
        dmin, dmax = float(depth.min()), float(depth.max())
        norm = (depth - dmin) / (dmax - dmin + 1e-8)
        depth_8u = (255.0 * norm).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_INFERNO)  # BGR colormap

        # Ensure color map same size as frame
        frame = packet.frame
        if frame.ndim == 2:
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_color = frame.copy()

        h_d, w_d = depth_color.shape[:2]
        h_f, w_f = frame_color.shape[:2]
        if (h_d, w_d) != (h_f, w_f):
            depth_color = cv2.resize(depth_color, (w_f, h_f), interpolation=cv2.INTER_LINEAR)

        overlay = cv2.addWeighted(frame_color, 1.0 - alpha, depth_color, alpha, 0)

               # Draw ROI rectangle and compute ROI score
        status_text = None
        status_color = (0, 255, 0)
        if annotate_roi:
            x_min, x_max, y_min, y_max = self.cfg.roi
            x1 = int(x_min * w_f)
            x2 = int(x_max * w_f)
            y1 = int(y_min * h_f)
            y2 = int(y_max * h_f)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # extract ROI from original depth and compute percentile score
            roi_depth = depth[y1:y2, x1:x2] if (0 <= x1 < x2 and 0 <= y1 < y2) else np.array([])
            if roi_depth.size:
                roi_score = float(np.percentile(roi_depth, self.cfg.roi_percentile))
                # normalize the roi_score to the same range used for visualization
                roi_norm = (roi_score - dmin) / (dmax - dmin + 1e-8)
                # decide obstacle presence (lower normalized depth -> closer -> obstacle)
                if roi_norm < self.cfg.obstacle_norm_threshold:
                    status_text = f"OBSTACLE (dist:{roi_norm:.2f})"
                    status_color = (0, 0, 255)
                else:
                    status_text = f"CLEAR (dist:{roi_norm:.2f})"
                    status_color = (0, 255, 0)

        # Overlay status text (top-left of ROI if present, else top-left of image)
        if status_text:
            # choose position
            try:
                tx, ty = x1, max(10, y1 - 10)
            except UnboundLocalError:
                tx, ty = 10, 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.7
            th = 2
            (text_w, text_h), _ = cv2.getTextSize(status_text, font, scale, th)
            # background rectangle for readability
            bg_tl = (tx - 5, ty - text_h - 5)
            bg_br = (tx + text_w + 5, ty + 5)
            cv2.rectangle(overlay, bg_tl, bg_br, status_color, -1)
            cv2.putText(overlay, status_text, (tx, ty), font, scale, (255, 255, 255), th, cv2.LINE_AA)


        # Optionally save and/or show
        if save_path:
            cv2.imwrite(save_path, overlay)

        cv2.imshow(window_name, overlay)
        if block:
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)
        else:
            cv2.waitKey(1)
    
    def close(self) -> None:
        self._model = None
