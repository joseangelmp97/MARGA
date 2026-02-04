# This file defines how detections, frames and hazard states are represented in the system.

# src/core/types.py
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List, Tuple

BBoxXYXY = Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max) in pixels

@dataclass(frozen=True)
class Detection:
    """Represents a detected object in a frame."""

    bbox: BBoxXYXY  # Bounding box of the detected object
    class_id: int  # Class ID of the detected object
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
    detections: List[Detection]  # List of detections in the frame

@dataclass(frozen=True)
class HazardState:
    """Represents the state of a detected hazard."""

    hazard: bool  # Whether a hazard is detected
    severity: Severity  # Severity level of the hazard
    description: Optional[str] = None  # Optional description of the hazard
    affected_detections: Optional[List[Detection]] = None  # Detections related to the hazard

@dataclass(frozen=True)
class FramePacket:
    """Combines frame information with its associated hazard state."""

    frame: object  # The actual frame data
    frame_info: FrameInfo  # Metadata about the frame
    hazard_state: HazardState  # Hazard state associated with the frame

    @staticmethod
    def from_frame(frame: object) -> FramePacket:
        """Creates a FramePacket from a given frame with default metadata and hazard state.
        """
        default_frame_info = FrameInfo(width=0, height=0, timestamp=0.0, detections=[])
        default_hazard_state = HazardState(hazard=False, severity=Severity.NONE)
        return FramePacket(frame=frame, frame_info=default_frame_info, hazard_state=default_hazard_state)

