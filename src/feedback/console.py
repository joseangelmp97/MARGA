"""
Console notifier for debugging.
"""

# src/feedback/console.py
from __future__ import annotations

import time

from ..core.interfaces import Notifier
from ..core.types import FramePacket, HazardState


class ConsoleNotifier(Notifier):
    """
    Console logger for debugging.

    - Prints hazard events (message + severity).
    - Prints FPS (ticks/second) every N processed frames.
    """

    def __init__(self, log_fps_every_n_processed: int = 30):
        self.log_fps_every_n_processed = max(1, int(log_fps_every_n_processed))
        self._processed = 0
        self._t0 = time.time()

    def notify(self, packet: FramePacket, hazard_state: HazardState) -> None:
        self._processed += 1

        # Print hazard events
        if hazard_state.hazard:
            print(f"[HAZARD] severity={int(hazard_state.severity)} msg='{hazard_state.message}'")

        # Periodic FPS info (rough, per notify calls)
        if self._processed % self.log_fps_every_n_processed == 0:
            t1 = time.time()
            dt = max(1e-6, t1 - self._t0)
            fps = self.log_fps_every_n_processed / dt
            print(f"[PERF] approx FPS: {fps:.2f}")
            self._t0 = t1

    def close(self) -> None:
        # Nothing to release for console
        pass
