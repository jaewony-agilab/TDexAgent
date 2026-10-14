"""Boundary frames: the policy's scene (ego) image and the boundary hand's wrist view,
each plain and with the taxel overlay. Wrist views are rendered only at boundaries;
the policy input is untouched."""

from __future__ import annotations

import numpy as np

from tdex.camera import ImageTransform, make_marks
from tdex.render import RenderConfig, render_overlay
from tdex.sim.gr1 import EGO_RAW, GR1Env
from tdex.sim.taxels import TaxelSensor


class FrameMaker:
    def __init__(self, env: GR1Env, sensor: TaxelSensor, render_cfg: RenderConfig, ego_camera: str = "egoview",
                 wrist_camera: str = "eye_in_{hand}_hand", wrist_size: int = 256, use_wrist: bool = True):
        self.env, self.sensor, self.render_cfg = env, sensor, render_cfg
        self.ego_camera = ego_camera
        self.wrist_camera = wrist_camera
        self.wrist_size = wrist_size
        self.use_wrist = use_wrist

    @staticmethod
    def ego_key(obs: dict) -> str:
        keys = [k for k in obs if k.startswith("video.") and "bg_crop" not in k]
        if len(keys) != 1:
            raise KeyError(f"expected one ego video key, got {keys}")
        return keys[0]

    def _view(self, cam_name, raw_wh, out_wh, transform, image, readings, lost):
        cam = self.env.camera(cam_name, *raw_wh)
        _, depth = self.env.render(self.env.camera_name(cam_name), *raw_wh, depth=True)
        marks = make_marks(self.sensor.anchors(), readings, cam, transform, out_wh, depth, lost)
        return render_overlay(image, marks, self.render_cfg)

    def __call__(self, b, obs, readings, lost):
        ego = np.asarray(obs[self.ego_key(obs)])[-1]
        frames = {"scene": ego}
        overlay = {"scene": self._view(self.ego_camera, EGO_RAW, ego.shape[1::-1], GR1Env.ego_transform(), ego, readings, lost)}
        if self.use_wrist:
            name = self.wrist_camera.format(hand=b.hand)
            s = self.wrist_size
            wrist, _ = self.env.render(self.env.camera_name(name), s, s, depth=False)
            frames["wrist"] = wrist
            overlay["wrist"] = self._view(name, (s, s), (s, s), ImageTransform(), wrist, readings, lost)
        return frames, overlay
