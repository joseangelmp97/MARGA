# This file defines how detections, frames and hazard states are represented in the system.

# src/core/types.py
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List, Tuple
import time

BBoxXYXY = Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max) in pixels

@dataclass(frozen=True)
class Detection:
    """Represents a detected object in a frame."""

    bbox: BBoxXYXY  # Bounding box of the detected object
    label : str  # Class label of the detected object (e.g., "person", "car")
    class_id: Optional[int] = None  # Class ID of the detected object
    confidence: float  # Confidence score of the detection
    tracking_id: Optional[int] = None  # Optional tracking ID for the detected object

class Severity(IntEnum):
    """Enumeration of hazard severity levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

@dataclass(frozen=True)
class FrameInfo:
    """Metadata about a frame, without forcing a specific image type here."""

    width: int  # Width of the frame in pixels
    height: int  # Height of the frame in pixels
    timestamp: float  # Timestamp of the frame in seconds

@dataclass(frozen=True)
class HazardState:
    """Represents the state of a detected hazard."""

    hazard: bool  # Whether a hazard is detected
    severity: Severity  # Severity level of the hazard
    message: Optional[str] = None  # Optional description of the hazard
    affected_detections: Optional[List[Detection]] = None  # Detections related to the hazard

@dataclass(frozen=True)
class FramePacket:
    """Combines frame information with its associated hazard state."""

    frame: object  # The actual frame data
    frame_info: FrameInfo  # Metadata about the frame

    @staticmethod
    def from_frame(frame: object) -> FramePacket:
        """Factory method to create a FramePacket from raw frame data. The actual type of 'frame' can be defined by the implementation (e.g., numpy array, PIL image, etc.).
        """
        h, w = frame.shape[:2]
        return FramePacket(
            frame=frame,
            info=FrameInfo(width=w, height=h, timestamp=time.time()),
        )

