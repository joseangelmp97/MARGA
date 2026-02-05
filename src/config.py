"""
Central configuration file for the project.
Goal:
- Keep all "knobs" (URLs, thresholds, cooldown times, etc.) in one place for easy tuning and experimentation.
- Avoid hardcoding values in the core logic, making it easier to maintain and update.
- Make it easy to switch components without changing the core code.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, Tuple

# Region of Interest (ROI) defined as normalized coordinates (x_min, y_min, x_max, y_max) where values are between 0 and 1.
ROINorm = Tuple[float, float, float, float]

# Capture configuration.
@dataclass(frozen=True)
class CaptureConfig:
    """
    Defines where frames come from.

    For now (MVP): IP Webcam MJPEG stream from the phone.
    Later: webcam device index, video file, RTSP, etc.
    """
    source_type: str = "ipwebcam"  # "ipwebcam" | "webcam" | "file" (future)
    ipwebcam_url: str = "http://127.0.0.1:8080/video"

    # If you later want to force a maximum resolution, you can add:
    # max_width: Optional[int] = None
    # max_height: Optional[int] = None

# Detector configuration.
@dataclass(frozen=True)
class DetectorConfig:
    """
    Defines which perception model to use and its main thresholds.

    MVP: MobileNet-SSD (Caffe) via OpenCV DNN.
    Future: YOLO ONNX, TensorRT, etc.
    """
    detector_type: str = "mobilenetssd_caffe"  # "mobilenetssd_caffe" | "yolo_onnx" (future)

    # Model paths (only used by MobileNet-SSD Caffe detector)
    proto_path: str = "models/MobileNetSSD_deploy.prototxt"
    model_path: str = "models/MobileNetSSD_deploy.caffemodel"

    # Runtime thresholds
    conf_threshold: float = 0.50 
    nms_threshold: float = 0.40 

    # Speed knobs
    process_every_n_frames: int = 2   # e.g. 1 for max accuracy, 2-3 for speed

# Decision / Hazard logic configuration.
@dataclass(frozen=True)
class HazardConfig:
    """
    Defines the rules that convert detections into a hazard.

    We keep it model-agnostic:
    - It should not care whether detections come from MobileNet or YOLO.
    """
    # Which classes are considered obstacles (by label or by class_id mapping).
    # For now, we'll assume we will map class_id -> label in the detector,
    # and the hazard logic will operate on labels.
    obstacle_labels: Set[str] = frozenset({
        "person", "car", "bus", "bicycle", "motorbike", "chair", "sofa"
    })

    # A simple "central corridor" ROI to reduce noisy alerts.
    # (x_min, x_max, y_min, y_max) normalized to [0..1]
    roi: ROINorm = (0.33, 0.67, 0.20, 0.95)

    # Anti-spam / stability rules
    persist_frames: int = 3     # require hazard to be true for N processed frames
    cooldown_s: float = 1.5     # minimum seconds between alerts

    # Optional: how to choose "top" object for message
    # "highest_conf" | "largest_area" (future)
    top_selection: str = "highest_conf"


# ---------------------------
# Feedback configuration
# ---------------------------

@dataclass(frozen=True)
class FeedbackConfig:
    """
    Defines output channels.

    MVP: Termux TTS (phone speaks).
    Also useful: save debug frames since Termux has no GUI.
    """
    enable_tts: bool = True
    tts_language: Optional[str] = None  # keep None unless you need to force it

    # Debug artifacts on phone storage:
    enable_debug_frames: bool = True
    debug_dir: str = "/sdcard/eta_debug"
    save_every_n_processed: int = 10  # save 1 out of N processed frames

    # Console logging
    log_fps_every_n_processed: int = 30


# ---------------------------
# App configuration (the bundle)
# ---------------------------

@dataclass(frozen=True)
class AppConfig:
    """
    High-level config: just groups all sub-configs.
    """
    capture: CaptureConfig = CaptureConfig()
    detector: DetectorConfig = DetectorConfig()
    hazard: HazardConfig = HazardConfig()
    feedback: FeedbackConfig = FeedbackConfig()


def default_config() -> AppConfig:
    """
    Factory function so the app can import a single function and start running.
    """
    return AppConfig()