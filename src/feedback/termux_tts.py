# src/feedback/termux_tts.py
"""
Termux TTS notifier.

Responsibility:
- Speak hazard messages using Termux API (termux-tts-speak).

Notes:
- This notifier should NOT decide *when* a hazard happens.
  It only reacts to HazardState produced by HazardLogic.
- It implements a simple rate limiter (cooldown) to avoid spam.
"""

from __future__ import annotations

import os
import time

from ..core.interfaces import Notifier
from ..core.types import FramePacket, HazardState


class TermuxTTSNotifier(Notifier):
    def __init__(self, cooldown_s: float = 1.5):
        # Cooldown prevents speaking too frequently even if hazard stays true.
        self.cooldown_s = float(cooldown_s)
        self._last_spoken_t = 0.0

    def notify(self, packet: FramePacket, hazard_state: HazardState) -> None:
        # Only speak on hazards (no chatter on safe state).
        if not hazard_state.hazard:
            return

        now = time.time()
        if (now - self._last_spoken_t) < self.cooldown_s:
            return

        msg = hazard_state.message.strip()
        if not msg:
            msg = "Obstacle ahead"

        # Termux API command (requires: pkg install termux-api)
        # Keep it simple for MVP: no language/engine options unless needed.
        os.system(f'termux-tts-speak "{msg}"')

        self._last_spoken_t = now

    def close(self) -> None:
        # Nothing persistent to close for Termux TTS.
        pass
