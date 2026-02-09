""" This file decides which methods must implement the different modules of the system. """

#src/core/interfaces.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .types import FramePacket, Detection, HazardState

class FrameSource(ABC):
    """Interface for frame source modules."""

    @abstractmethod
    def get_frame_packet(self) -> Optional[FramePacket]:
        """Retrieves the next FramePacket from the source or None if no more frames are available."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Cleans up any resources held by the frame source."""
        pass

class Detector(ABC):
    """ Runs perception on a frame and outputs detections.
        Must NOT contain hazard logic, cooldown, TTS, etc."""

    @abstractmethod
    def detect(self, frame_packet: FramePacket) -> List[Detection]:
        """Detects objects in a given frame packet."""
        pass

class DepthEstimator(ABC):

    """
    Module to estimate depth in the ROI. This can be used by hazard logic to improve decision making. If not implemented, depth_roi in FrameInfo will be None.
    """

    @abstractmethod
    def estimate_roi_depth(self, packet: FramePacket) -> float | None:
        """Returns a single depth score for ROI (relative)."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class HazardLogic(ABC):
    """
    Converts detections + frame info into hazard states.
    Must be pure decision logic without side effects.
    """

    @abstractmethod
    def update(self, packet: FramePacket, detections: List[Detection]) -> HazardState:
        """Updates the hazard state based on the provided frame packet and detections."""
        pass

class Notifier(ABC):
    
    """
    Consumes HazardState and triggers side effects (TTS/beep/logs/debug frames).
    Should handle its own cooldown/rate limiting if needed.
    """

    @abstractmethod
    def notify(self, packet: FramePacket,hazard_state: HazardState) -> None:
        """Sends a notification based on the provided hazard state."""
        pass
    @abstractmethod
    def close(self) -> None:
        pass