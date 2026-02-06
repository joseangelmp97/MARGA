"""
Composite notifier that forwards notifications to multiple notifiers.
"""

# src/feedback/composite.py
from __future__ import annotations

from typing import List

from ..core.interfaces import Notifier
from ..core.types import FramePacket, HazardState


class CompositeNotifier(Notifier):
    """
    A notifier that forwards notifications to multiple notifiers.
    """

    def __init__(self, notifiers: List[Notifier]):
        self.notifiers = notifiers

    def notify(self, packet: FramePacket, hazard_state: HazardState) -> None:
        for n in self.notifiers:
            n.notify(packet, hazard_state)

    def close(self) -> None:
        for n in self.notifiers:
            try:
                n.close()
            except Exception:
                pass
