"""Data passed across the boundary interface. Nothing here carries privileged state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

Hand = Literal["left", "right"]
BoundaryType = Literal["close", "open"]
Outcome = Literal["success", "fail", "unclear"]

FINGERS = ("thumb", "index", "middle", "ring", "pinky", "palm")
HANDS: tuple[Hand, Hand] = ("left", "right")


@dataclass(frozen=True)
class Boundary:
    hand: Hand
    boundary_type: BoundaryType
    step: int  # env step at which the settle window ended


@dataclass
class FingerSummary:
    """Per-link taxel summary over the window since the previous boundary of this hand."""

    hand: Hand
    finger: str
    force_now: tuple[float, float, float]  # world-frame force on the link at the boundary [N]
    magnitude_now: float
    peak_magnitude: float  # max over the window
    in_contact_now: bool
    contact_lost: bool  # was in contact at the previous boundary, not now

    def to_text(self) -> str:
        state = "contact" if self.in_contact_now else "no contact"
        lost = ", LOST since last boundary" if self.contact_lost else ""
        return (
            f"{self.hand} {self.finger}: {state}, |F|={self.magnitude_now:.1f} N "
            f"(peak {self.peak_magnitude:.1f} N){lost}"
        )


@dataclass
class FingerMark:
    """One overlay marker in image coordinates (produced by the projector, consumed by render)."""

    hand: Hand
    finger: str
    uv: tuple[float, float]  # anchor pixel (x right, y down)
    direction: tuple[float, float] | None  # unit image-space direction of the force, None if no force
    magnitude: float  # [N]
    occluded: bool
    lost: bool
    in_view: bool = True


@dataclass
class Verdict:
    outcome: Outcome
    plan_still_valid: Literal["yes", "no"]
    reason: str
    source: str = ""  # judge name, for logs

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BoundaryPacket:
    """Everything a judge may see at one boundary."""

    boundary: Boundary
    frames: dict[str, np.ndarray]  # view name -> HxWx3 uint8 (plain RGB)
    overlay_frames: dict[str, np.ndarray]  # same views with taxel overlay
    finger_summaries: list[FingerSummary]
    task_instruction: str
    retry_count: int
    expected_subtask: str = ""
    prev_overlay_frame: np.ndarray | None = None
    prev_plain_frame: np.ndarray | None = None
    prev_verdict: Verdict | None = None
    meta: dict = field(default_factory=dict)  # episode/task ids for logging only
