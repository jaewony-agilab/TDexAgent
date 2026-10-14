"""Taxel overlay renderer.

Encoding (one marker per fingertip and palm, nothing finer):
  * fill color = finger (fixed, same on both hands)
  * outline    = hand (left: white, right: black)
  * arrow      = force on the link; length ∝ |F|, clipped at `force_clip_n`
                 (95th percentile measured on probe data)
  * dashed outline = anchor occluded in this view (depth test)
  * X          = contact lost since the previous boundary
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from tdex.types import FingerMark

FINGER_COLORS = {  # Okabe-Ito, distinguishable under common color-vision deficiencies
    "thumb": (213, 94, 0),
    "index": (230, 159, 0),
    "middle": (240, 228, 66),
    "ring": (0, 158, 115),
    "pinky": (86, 180, 233),
    "palm": (204, 121, 167),
}
HAND_OUTLINE = {"left": (255, 255, 255), "right": (0, 0, 0)}

LEGEND_TEXT = (
    "Overlay legend: each colored dot is one link of a robot hand "
    "(thumb=vermillion, index=orange, middle=yellow, ring=green, pinky=sky blue, palm=pink). "
    "White outline = left hand, black outline = right hand. "
    "An arrow from a dot is the contact force on that link; longer = stronger "
    "(full length = {clip:.0f} N or more); no arrow = no contact. "
    "Dashed outline = that link is hidden behind something in this view. "
    "A red X = that link had contact at the previous check and has lost it."
)


@dataclass
class RenderConfig:
    force_clip_n: float = 10.0  # set from probe data (95th percentile), see DECISIONS.md
    marker_radius_frac: float = 0.016  # of image width
    arrow_max_frac: float = 0.12  # of image width
    line_width_frac: float = 0.006


def _dashed_circle(draw: ImageDraw.ImageDraw, c, r, color, width, n_dash=8):
    for i in range(n_dash):
        a0 = 360.0 * i / n_dash
        draw.arc([c[0] - r, c[1] - r, c[0] + r, c[1] + r], a0, a0 + 360.0 / n_dash / 2, fill=color, width=width)


def _arrow(draw, start, end, color, width):
    draw.line([start, end], fill=color, width=width)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    head = max(3 * width, 4)
    for s in (+1, -1):
        a = ang + math.pi - s * math.pi / 7
        draw.line([end, (end[0] + head * math.cos(a), end[1] + head * math.sin(a))], fill=color, width=width)


def render_overlay(image: np.ndarray, marks: list[FingerMark], cfg: RenderConfig | None = None) -> np.ndarray:
    cfg = cfg or RenderConfig()
    img = Image.fromarray(np.ascontiguousarray(image)).convert("RGB")
    draw = ImageDraw.Draw(img)
    W = img.width
    r = max(2.0, cfg.marker_radius_frac * W)
    lw = max(1, round(cfg.line_width_frac * W))
    amax = cfg.arrow_max_frac * W

    # arrows first so markers stay legible on top
    for m in marks:
        if not m.in_view or m.direction is None:
            continue
        length = r + amax * min(m.magnitude / cfg.force_clip_n, 1.0)
        start = m.uv
        end = (start[0] + m.direction[0] * length, start[1] + m.direction[1] * length)
        _arrow(draw, start, end, HAND_OUTLINE[m.hand], lw + 2)  # halo for contrast
        _arrow(draw, start, end, FINGER_COLORS[m.finger], lw)

    for m in marks:
        if not m.in_view:
            continue
        x, y = m.uv
        box = [x - r, y - r, x + r, y + r]
        outline = HAND_OUTLINE[m.hand]
        if m.occluded:
            draw.ellipse(box, fill=None)
            _dashed_circle(draw, (x, y), r, FINGER_COLORS[m.finger], lw + 1)
            _dashed_circle(draw, (x, y), r + lw, outline, lw)
        else:
            draw.ellipse(box, fill=FINGER_COLORS[m.finger], outline=outline, width=lw)
        if m.lost:
            k = r * 0.9
            draw.line([(x - k, y - k), (x + k, y + k)], fill=(220, 0, 0), width=lw + 1)
            draw.line([(x - k, y + k), (x + k, y - k)], fill=(220, 0, 0), width=lw + 1)
    return np.asarray(img)


def legend_text(cfg: RenderConfig | None = None) -> str:
    return LEGEND_TEXT.format(clip=(cfg or RenderConfig()).force_clip_n)
