"""Pinhole projection for MuJoCo cameras, and anchors+forces -> overlay marks.

Conventions: MuJoCo cameras look along their local -z with +y up. Pixel
coordinates are (x right, y down) in the *flipped* image the env returns
(robocasa flips the OpenGL buffer upside down). Depth maps passed here must be
metric and in the same flipped orientation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tdex.types import FingerMark


@dataclass
class Camera:
    pos: np.ndarray  # (3,) world position
    xmat: np.ndarray  # (3,3) world-from-camera rotation (columns = camera axes)
    fovy_deg: float
    width: int
    height: int

    @property
    def focal(self) -> float:
        return (self.height / 2.0) / np.tan(np.deg2rad(self.fovy_deg) / 2.0)

    def to_camera(self, p_world: np.ndarray) -> np.ndarray:
        return (np.asarray(p_world, float) - self.pos) @ self.xmat  # = R^T (p - c)

    def project(self, p_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points (...,3) -> pixels (...,2) and depth along the view axis (...)."""
        pc = self.to_camera(p_world)
        depth = -pc[..., 2]
        safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
        f = self.focal
        u = self.width / 2.0 + f * pc[..., 0] / safe
        v = self.height / 2.0 - f * pc[..., 1] / safe
        return np.stack([u, v], axis=-1), depth


@dataclass
class ImageTransform:
    """Pixel map from the raw render to the image the overlay is drawn on (pad, then resize)."""

    pad_left: float = 0.0
    pad_top: float = 0.0
    scale: float = 1.0

    @classmethod
    def square_pad_resize(cls, width: int, height: int, out_size: int) -> "ImageTransform":
        """Matches robocasa GrootRoboCasaEnv.process_img (centered pad to square, resize)."""
        dim = max(width, height)
        return cls(pad_left=(dim - width) // 2, pad_top=(dim - height) // 2, scale=out_size / dim)

    def apply(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, float)
        return np.stack([(uv[..., 0] + self.pad_left), (uv[..., 1] + self.pad_top)], -1) * self.scale


def is_occluded(uv: np.ndarray, depth: float, depth_map: np.ndarray, tol: float = 0.01) -> bool:
    """Anchor is occluded if the depth buffer at its pixel is closer than the anchor by > tol [m]."""
    h, w = depth_map.shape[:2]
    x, y = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= x < w and 0 <= y < h):
        return False
    return bool(depth_map[y, x] < depth - tol)


def make_marks(
    anchors: dict[tuple[str, str], np.ndarray],
    forces: dict[tuple[str, str], np.ndarray],
    camera: Camera,
    transform: ImageTransform,
    out_size: tuple[int, int],
    depth_map: np.ndarray | None = None,
    lost: dict[tuple[str, str], bool] | None = None,
    min_force_n: float = 0.5,
    probe_len_m: float = 0.02,
) -> list[FingerMark]:
    """Project link anchors and force directions into the output image.

    The arrow direction is the image-space direction of anchor -> anchor + probe
    along the force; its length is set by the renderer from the magnitude.
    """
    marks = []
    W, H = out_size
    for key, anchor in anchors.items():
        hand, finger = key
        anchor = np.asarray(anchor, float)
        uv_raw, depth = camera.project(anchor)
        uv = transform.apply(uv_raw)
        in_view = depth > 0 and 0 <= uv[0] < W and 0 <= uv[1] < H
        occluded = bool(depth_map is not None and depth > 0 and is_occluded(uv_raw, depth, depth_map))
        f = np.asarray(forces.get(key, np.zeros(3)), float)
        mag = float(np.linalg.norm(f))
        direction = None
        if mag >= min_force_n and depth > 0:
            tip_raw, tip_depth = camera.project(anchor + probe_len_m * f / mag)
            if tip_depth > 0:
                d = transform.apply(tip_raw) - uv
                n = float(np.linalg.norm(d))
                if n > 1e-6:
                    direction = (float(d[0] / n), float(d[1] / n))
        marks.append(
            FingerMark(
                hand=hand,
                finger=finger,
                uv=(float(uv[0]), float(uv[1])),
                direction=direction,
                magnitude=mag,
                occluded=occluded,
                lost=bool((lost or {}).get(key, False)),
                in_view=bool(in_view),
            )
        )
    return marks
