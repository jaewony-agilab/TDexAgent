"""GR-1 Tabletop env, built exactly as RLDX-1's rollout_policy.create_eval_env builds it,
except that the MultiStepWrapper executes one action per step so the harness can
inspect every step (the 16-step chunk is replayed by the harness instead).

Seeding mirrors RLDX-1's VideoRecordingWrapper: episode k of a run with base
seed s (env_idx 0) resets with seed (s * 100000 + k), so episode ids line up
with the published eval's init states.
"""

from __future__ import annotations

import os
import random

import numpy as np

from tdex.camera import Camera, ImageTransform

SEED_STRIDE = 100000
EGO_RAW = (1280, 800)  # robocasa GR1 key converter: egoview width, height
POLICY_IMG = 256


def batch_obs(obs: dict) -> dict:
    """Add the batch axis the way gymnasium's SyncVectorEnv(n=1) does."""
    return {k: (v,) if isinstance(v, str) else np.asarray(v)[None] for k, v in obs.items()}


class GR1Env:
    def __init__(self, task: str, video_delta_indices, state_delta_indices, seed: int = 42, env_idx: int = 0):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import gymnasium as gym
        import robocasa  # noqa: F401  (registers envs)
        import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
        from rldx.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper

        self.task = task
        self.env_seed = seed + env_idx
        random.seed(self.env_seed)
        np.random.seed(self.env_seed)
        self.base = gym.make(f"gr1_unified/{task}", enable_render=True, seed=self.env_seed)
        self.env = MultiStepWrapper(
            self.base,
            video_delta_indices=np.asarray(video_delta_indices),
            state_delta_indices=None if state_delta_indices is None else np.asarray(state_delta_indices),
            n_action_steps=1,
            max_episode_steps=None,
        )
        self.reset_hooks: list = []  # called after every reset (the MjModel may be rebuilt)

    # --- raw handles -------------------------------------------------------
    @property
    def rs(self):
        """The robosuite env."""
        return self.base.unwrapped.env

    @property
    def model(self):
        return self.rs.sim.model._model

    @property
    def data(self):
        return self.rs.sim.data._data

    # --- episode -----------------------------------------------------------
    def reset(self, episode_id: int) -> dict:
        ep_seed = self.env_seed * SEED_STRIDE + int(episode_id)
        random.seed(ep_seed)
        np.random.seed(ep_seed)
        self.rs.seed = ep_seed
        self.rs.rng = np.random.default_rng(ep_seed)
        obs, _ = self.env.reset(seed=ep_seed)
        for hook in self.reset_hooks:
            hook(self)
        return obs

    def step(self, action: dict[str, np.ndarray]):
        obs, reward, terminated, truncated, info = self.env.step({k: np.asarray(v)[None] for k, v in action.items()})
        success = bool(np.any(info.get("success", False)))
        return obs, float(reward), bool(terminated), bool(truncated), success, info

    def instruction(self, obs: dict) -> str:
        for k, v in obs.items():
            if k.startswith("annotation."):
                return v
        return ""

    # --- cameras -----------------------------------------------------------
    def camera_id(self, name: str) -> int:
        """Exact name, else the unique camera whose name ends with `_<name>` (robosuite prefixes)."""
        import mujoco

        m = self.model
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid >= 0:
            return cid
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) or "" for i in range(m.ncam)]
        hits = [i for i, n in enumerate(names) if n.endswith("_" + name)]
        if len(hits) != 1:
            raise KeyError(f"camera {name!r}: matches {[names[i] for i in hits]} among {names}")
        return hits[0]

    def camera_name(self, name: str) -> str:
        import mujoco

        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_id(name))

    def camera(self, name: str, width: int, height: int) -> Camera:

        cid = self.camera_id(name)
        return Camera(
            pos=np.array(self.data.cam_xpos[cid]),
            xmat=np.array(self.data.cam_xmat[cid]).reshape(3, 3),
            fovy_deg=float(self.model.cam_fovy[cid]),
            width=width,
            height=height,
        )

    def render(self, name: str, width: int, height: int, depth: bool = True):
        """RGB (and metric depth) in the env's flipped (row 0 = top) convention."""
        out = self.rs.sim.render(camera_name=name, width=width, height=height, depth=depth)
        if not depth:
            return np.ascontiguousarray(out[::-1]), None
        rgb, d = out
        m = self.model
        extent = m.stat.extent
        near, far = m.vis.map.znear * extent, m.vis.map.zfar * extent
        metric = near / (1.0 - d * (1.0 - near / far))
        return np.ascontiguousarray(rgb[::-1]), np.ascontiguousarray(metric[::-1])

    @staticmethod
    def ego_transform() -> ImageTransform:
        return ImageTransform.square_pad_resize(*EGO_RAW, POLICY_IMG)
