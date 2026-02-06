"""
Application entry point.

Responsibilities:
- Load configuration.
- Build the pipeline modules (source -> detector -> hazard -> notifier).
- Run the main loop.
- Handle shutdown cleanly.

Non-responsibilities (IMPORTANT):
- No model details here.
- No hazard rules here.
- No Termux commands here.
"""
from __future__ import annotations

import time
from typing import List

from .config import AppConfig, default_config
from .core.interfaces import FrameSource, Detector, HazardLogic, Notifier
from .core.types import FramePacket, Detection, HazardState

# Factory functions to create module instances based on config. This keeps the app.py clean and focused on orchestration.

def build_source(cfg: AppConfig) -> FrameSource:
    """
    Docstring for build_source

    Builds the frame source module based on the configuration. This allows us to easily switch between different input sources (e.g., IP webcam, video file, camera) without changing the main loop or the other modules. For the MVP, we just have an IP webcam source, but in the future we could add more sources and select them here.
    
    :param cfg: Description
    :type cfg: AppConfig
    :return: Description
    :rtype: FrameSource
    """
    if cfg.capture.source_type == "ipwebcam":
        from .capture.ipwebcam import IPWebcamSource
        return IPWebcamSource(cfg.capture.ipwebcam_url)
    raise ValueError(f"Unsupported source type: {cfg.capture.source_type}")

def build_detector(cfg: AppConfig) -> Detector:
    """
    Docstring for build_detector

    builds the detection module based on the config. This allows us to easily switch between different models (e.g., MobileNet-SSD, YOLO) without changing the main loop or the hazard logic. For the MVP, we just have MobileNet-SSD via OpenCV DNN, but in the future we could add more models and select them here.
    
    :param cfg: Description
    :type cfg: AppConfig
    :return: Description
    :rtype: Detector
    """
    if cfg.detector.detector_type == "mobilenetssd_caffe":
        from .perception.mobilenetssd_caffe import MobileNetSSDDetector
        return MobileNetSSDDetector(
            proto_path=cfg.detector.proto_path,
            model_path=cfg.detector.model_path,
            conf_threshold=cfg.detector.conf_threshold,
            nms_threshold=cfg.detector.nms_threshold,
            process_every_n_frames=cfg.detector.process_every_n_frames
        )
    raise ValueError(f"Unsupported detector type: {cfg.detector.detector_type}")

def build_hazard_logic(cfg: AppConfig) -> HazardLogic:
    """
    Docstring for build_hazard_logic

    build_hazard_logic creates an instance of the hazard logic module based on the configuration. This allows us to easily switch between different decision rules without changing the main loop or the detector. For the MVP, we have a simple rule-based logic, but in the future we could add more complex logic (e.g., ML-based) and just select it here.
    
    :param cfg: Description
    :type cfg: AppConfig
    :return: Description
    :rtype: HazardLogic
    """
    from .decision.hazard import SimpleHazardLogic
    return SimpleHazardLogic(
        obstacle_labels=set(cfg.hazard.obstacle_labels),
        roi=cfg.hazard.roi,
        persist_frames=cfg.hazard.persist_frames,
        cooldown_s=cfg.hazard.cooldown_s,
        top_selection=cfg.hazard.top_selection,)

def build_notifier(cfg: AppConfig) -> Notifier:
    """
    Docstring for build_notifier

    Builds a composite notifier based on the config. This allows us to easily add multiple feedback channels (console, TTS, debug frames) without cluttering the main loop. 
    
    :param cfg: Description
    :type cfg: AppConfig
    :return: Description
    :rtype: Notifier
    """
    from .feedback.composite import CompositeNotifier
    from .feedback.console import ConsoleNotifier

    notifiers: List[Notifier] = []

    # Always useful: console output for debugging
    notifiers.append(ConsoleNotifier(log_fps_every_n_processed=cfg.feedback.log_fps_every_n_processed))

    if cfg.feedback.enable_tts:
        from .feedback.termux_tts import TermuxTTSNotifier
        notifiers.append(TermuxTTSNotifier())

    if cfg.feedback.enable_debug_frames:
        from .feedback.debug_writer import DebugFrameWriterNotifier
        notifiers.append(
            DebugFrameWriterNotifier(
                debug_dir=cfg.feedback.debug_dir,
                save_every_n_processed=cfg.feedback.save_every_n_processed,
            )
        )

    return CompositeNotifier(notifiers)
