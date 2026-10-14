"""Touch-only gate ablation: tactile-reader flags through the same gate table, no VLM."""

from __future__ import annotations

from tdex.types import BoundaryPacket, Verdict


class TouchOnlyJudge:
    name = "touch-only"

    def __init__(self, min_fingertips: int = 2):
        self.min_fingertips = min_fingertips

    def __call__(self, packet: BoundaryPacket) -> Verdict:
        tips = [s for s in packet.finger_summaries if s.finger != "palm"]
        n = sum(s.in_contact_now for s in tips)
        thumb = any(s.in_contact_now for s in tips if s.finger == "thumb")
        if packet.boundary.boundary_type == "close":
            ok = thumb and n >= self.min_fingertips
            why = f"{n} fingertips in contact, thumb {'on' if thumb else 'off'}"
        else:
            ok = n == 0
            why = f"{n} fingertips still in contact after opening"
        return Verdict("success" if ok else "fail", "yes", why, self.name)
