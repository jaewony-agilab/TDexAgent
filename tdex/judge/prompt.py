"""Judge request construction for the three Phase 1 input conditions.

  plain   : RGB frames only
  text    : RGB frames + per-finger taxel summary as text
  overlay : overlay-annotated frames + legend

Output is backend-neutral: a system string and a list of parts, each either
("text", str) or ("image", HxWx3 uint8). Backends turn parts into their wire format.
"""

from __future__ import annotations

import numpy as np

from tdex.render import legend_text
from tdex.types import BoundaryPacket

CONDITIONS = ("plain", "text", "overlay")

SYSTEM = (
    "You monitor a humanoid robot with two dexterous hands doing a tabletop task. "
    "You are called each time one hand finishes closing (a grasp attempt) or opening (a release). "
    "Decide whether that hand's action achieved what it was meant to.\n"
    "For a close: success = the hand is now holding the intended object securely. "
    "For an open: success = the object was released where the task needs it.\n"
    "outcome: 'success', 'fail', or 'unclear' (only if the evidence genuinely cannot decide).\n"
    "plan_still_valid: 'yes' if retrying the same step locally (open, back off, re-approach, re-close) "
    "could fix a failure; 'no' if the situation changed so the step should be replanned "
    "(e.g. object knocked over, fell out of reach, wrong object). For success answer 'yes'.\n"
    "reason: one or two sentences citing the visual or contact evidence.\n"
    'Reply with JSON only: {"outcome": ..., "plan_still_valid": ..., "reason": ...}'
)

Part = tuple[str, object]


def build_request(packet: BoundaryPacket, condition: str, views: tuple[str, ...] | None = None) -> tuple[str, list[Part]]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown judge condition {condition!r}")
    b = packet.boundary
    use_overlay = condition == "overlay"
    frames = packet.overlay_frames if use_overlay else packet.frames
    views = views or tuple(frames.keys())

    parts: list[Part] = [("text", f"Task instruction: {packet.task_instruction}")]
    if packet.expected_subtask:
        parts.append(("text", f"Planned current step: {packet.expected_subtask}"))
    parts.append(
        (
            "text",
            f"Event: the {b.hand} hand just finished {'closing' if b.boundary_type == 'close' else 'opening'}. "
            f"Local retries already spent on this hand: {packet.retry_count}.",
        )
    )
    if use_overlay:
        parts.append(("text", legend_text()))

    prev = packet.prev_overlay_frame if use_overlay else packet.prev_plain_frame
    if prev is not None and packet.prev_verdict is not None:
        parts.append(("text", f"Previous check (scene view) - verdict then: {packet.prev_verdict.outcome}: {packet.prev_verdict.reason}"))
        parts.append(("image", prev))

    for v in views:
        parts.append(("text", f"Current view: {v}"))
        parts.append(("image", frames[v]))

    if condition == "text":
        lines = [s.to_text() for s in packet.finger_summaries]
        parts.append(("text", "Fingertip/palm contact sensors for this hand:\n" + "\n".join(lines)))

    parts.append(("text", "Answer with the JSON verdict now."))
    return SYSTEM, parts


def images_in(parts: list[Part]) -> list[np.ndarray]:
    return [p for k, p in parts if k == "image"]
