# src/decision/hazard.py
"""
Hazard decision logic (MVP).

Responsibility:
- Convert detections into a HazardState using simple rules:
  ROI + obstacle labels + persistence + cooldown.

No side effects here (no TTS, no file writes).
"""

from __future__ import annotations

import time
from typing import List, Optional, Set, Tuple

from ..core.interfaces import HazardLogic
from ..core.types import Detection, FramePacket, HazardState, Severity


ROINorm = Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max) in [0..1]


class SimpleHazardLogic(HazardLogic):
    def __init__(
        self,
        obstacle_labels: Set[str],
        roi: ROINorm = (0.33, 0.67, 0.20, 0.95),
        persist_frames: int = 3,
        cooldown_s: float = 1.5,
        top_selection: str = "highest_conf",
    ):
        self.obstacle_labels = set(obstacle_labels)
        self.roi = roi
        self.persist_frames = max(1, int(persist_frames))
        self.cooldown_s = float(cooldown_s)
        self.top_selection = top_selection

        # Internal state (kept across frames)
        self._persist_counter = 0
        self._last_trigger_t = 0.0

    def update(self, packet: FramePacket, detections: List[Detection]) -> HazardState:
        """
        Decide if current frame is hazardous.

        The hazard condition is "true" if at least one obstacle detection
        is inside ROI.
        """
        w = packet.info.width
        h = packet.info.height

        # 1) Filter detections to obstacles inside ROI
        candidates: List[Detection] = []
        for d in detections:
            if d.label not in self.obstacle_labels:
                continue

            if self._in_roi(d, w, h):
                candidates.append(d)

        hazard_now = len(candidates) > 0

        # 2) Persistence logic (stability)
        if hazard_now:
            self._persist_counter += 1
        else:
            self._persist_counter = 0

        # 3) If not persistent enough yet, return "no hazard"
        if self._persist_counter < self.persist_frames:
            return HazardState(
                hazard=False,
                severity=Severity.NONE,
                message="",
                affected_detections=None,
            )

        # 4) Cooldown logic (anti-spam)
        now = time.time()
        if (now - self._last_trigger_t) < self.cooldown_s:
            # We consider hazard still true, but we avoid triggering a "new" alert.
            # For MVP, we can report hazard=True but keep message empty to avoid repeated TTS.
            return HazardState(
                hazard=True,
                severity=self._estimate_severity(candidates, w, h),
                message="",
                affected_detections=candidates,
            )

        # 5) Select the "top" detection for messaging
        top = self._select_top(candidates, w, h)

        # 6) Build message
        msg = f"{top.label} ahead" if top is not None else "Obstacle ahead"

        # 7) Update trigger time
        self._last_trigger_t = now

        return HazardState(
            hazard=True,
            severity=self._estimate_severity(candidates, w, h),
            message=msg,
            affected_detections=candidates,
        )

    # ---------------------------
    # Helpers
    # ---------------------------

    def _in_roi(self, d: Detection, w: int, h: int) -> bool:
        """
        Checks if detection center is inside the normalized ROI corridor.
        """
        x1, y1, x2, y2 = d.bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        x_min, x_max, y_min, y_max = self.roi
        return (x_min * w <= cx <= x_max * w) and (y_min * h <= cy <= y_max * h)

    def _select_top(self, detections: List[Detection], w: int, h: int) -> Optional[Detection]:
        """
        Choose the main detection to describe in the message.

        MVP default: highest confidence.
        Alternative (future): largest area.
        """
        if not detections:
            return None

        if self.top_selection == "largest_area":
            def area(d: Detection) -> int:
                x1, y1, x2, y2 = d.bbox
                return max(0, x2 - x1) * max(0, y2 - y1)
            return max(detections, key=area)

        # Default: highest confidence
        return max(detections, key=lambda d: d.confidence)

    def _estimate_severity(self, detections: List[Detection], w: int, h: int) -> Severity:
        """
        MVP severity estimate:
        - Use bbox area as a very rough proxy for closeness.
        - This is NOT real distance, but helps prioritize "big/close" objects.

        You can refine later with calibration or depth estimation.
        """
        if not detections:
            return Severity.NONE

        # Largest area ratio among candidate detections
        max_ratio = 0.0
        frame_area = max(1, w * h)

        for d in detections:
            x1, y1, x2, y2 = d.bbox
            a = max(0, x2 - x1) * max(0, y2 - y1)
            ratio = a / frame_area
            if ratio > max_ratio:
                max_ratio = ratio

        # Heuristic thresholds (tune later)
        if max_ratio > 0.18:
            return Severity.HIGH
        if max_ratio > 0.08:
            return Severity.MEDIUM
        return Severity.LOW
