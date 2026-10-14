"""Tactile reader: per-step taxel forces -> per-finger summaries at a boundary.

Input is a mapping (hand, finger) -> world-frame force on that link [N]; it is
produced by tdex.sim.taxels on the cluster and by fakes in tests.
"""

from __future__ import annotations

import numpy as np

from tdex.types import FINGERS, HANDS, FingerSummary, Hand

Readings = dict[tuple[str, str], np.ndarray]


class TactileReader:
    def __init__(self, contact_threshold_n: float = 0.5):
        self.contact_threshold_n = contact_threshold_n
        self.reset()

    def reset(self) -> None:
        self.latest: Readings = {}
        self._peak = {(h, f): 0.0 for h in HANDS for f in FINGERS}
        self._contact_at_prev_boundary = {(h, f): False for h in HANDS for f in FINGERS}

    def update(self, readings: Readings) -> None:
        self.latest = readings
        for key, force in readings.items():
            self._peak[key] = max(self._peak.get(key, 0.0), float(np.linalg.norm(force)))

    def summarize(self, hand: Hand) -> list[FingerSummary]:
        """Summaries for one hand; closes the window for that hand."""
        out = []
        for finger in FINGERS:
            key = (hand, finger)
            force = np.asarray(self.latest.get(key, np.zeros(3)), dtype=float)
            mag = float(np.linalg.norm(force))
            in_contact = mag >= self.contact_threshold_n
            out.append(
                FingerSummary(
                    hand=hand,
                    finger=finger,
                    force_now=tuple(float(x) for x in force),
                    magnitude_now=mag,
                    peak_magnitude=max(self._peak.get(key, 0.0), mag),
                    in_contact_now=in_contact,
                    contact_lost=self._contact_at_prev_boundary[key] and not in_contact,
                )
            )
            self._contact_at_prev_boundary[key] = in_contact
            self._peak[key] = 0.0
        return out

    def flags(self, hand: Hand, summaries: list[FingerSummary]) -> dict:
        """Rule-based flags for the touch-only gate ablation."""
        tips = [s for s in summaries if s.finger != "palm"]
        n_contact = sum(s.in_contact_now for s in tips)
        return {
            "n_fingertips_in_contact": n_contact,
            "thumb_in_contact": any(s.in_contact_now for s in tips if s.finger == "thumb"),
            "any_lost": any(s.contact_lost for s in summaries),
        }
