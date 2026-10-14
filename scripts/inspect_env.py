#!/usr/bin/env python
"""Phase 0 calibration probe. Needs an RLDX-1 server.

Dumps: obs/action keys and shapes, robot body / camera names, per-dim hand-command
percentiles from policy rollouts (-> closure.open_value / closed_value), taxel
magnitude percentiles (-> render.force_clip_n), and overlay frames at the most-closed
step so the projection convention can be checked by eye.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from tdex import config as C
from tdex.executor import RLDXExecutor
from tdex.runinfo import provenance, write_json
from tdex.sim.gr1 import GR1Env, batch_obs
from tdex.sim.frames import FrameMaker
from tdex.sim.taxels import TaxelSensor
from tdex.types import Boundary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=C.GR1_TASKS[0])
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--port", type=int, default=20100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(C.REPO / "configs/gr1.yaml"))
    args = ap.parse_args()
    cfg = C.load(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ex = RLDXExecutor("127.0.0.1", args.port, cfg["n_action_steps"])
    vdi, sdi = ex.modality_delta_indices()
    env = GR1Env(args.task, vdi, sdi, seed=cfg["seed"])
    obs = env.reset(0)
    m = env.model
    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
    report = {
        **provenance(args.config, vars(args)),
        "video_delta_indices": vdi,
        "state_delta_indices": sdi,
        "obs": {k: (v if isinstance(v, str) else list(np.shape(v))) for k, v in obs.items()},
        "action_space": {k: list(s.shape) for k, s in env.base.action_space.items()},
        "cameras": cams,
        "hand_bodies": [b for b in bodies if b and any(t in b.lower() for t in ("hand", "palm", "thumb", "index", "middle", "ring", "pinky"))],
        "body_prefixes": sorted({b.split("_")[0] + "_" for b in bodies if b and "_" in b}),
        "timestep": float(m.opt.timestep),
    }
    write_json(out / "inspect.json", report)  # write early: useful even if the rollout below fails

    tx = cfg["taxels"]
    sensor = TaxelSensor(env.model, env.data, tx["links"], tx["robot_prefixes"])
    env.reset_hooks.append(lambda e: sensor.bind(e.model, e.data))
    fm = FrameMaker(env, sensor, C.render_config(cfg))
    closure = C.closure_config(cfg)

    cmds = {"left": [], "right": []}
    states = {"left": [], "right": []}
    mags = []
    best = {"left": (-np.inf, None), "right": (-np.inf, None)}
    for ep in range(args.episodes):
        obs = env.reset(ep)
        ex.reset(f"inspect_{ep}")
        queue = []
        for t in range(args.steps):
            if not queue:
                queue = ex.act(batch_obs(obs))
            a = queue.pop(0)
            obs, _, term, trunc, success, _ = env.step(a)
            r = sensor.read()
            mags += [float(np.linalg.norm(f)) for f in r.values()]
            for h in ("left", "right"):
                c = np.asarray(a[f"action.{h}_hand"], float)
                cmds[h].append(c)
                states[h].append(np.asarray(obs[f"state.{h}_hand"])[-1])
                score = float(np.mean(c[list(closure.flexion_dims)]))
                if score > best[h][0]:
                    frames, overlay = fm(Boundary(h, "close", t), obs, r, {})
                    best[h] = (score, (frames, overlay))
            if success or term or trunc:
                break

    pct = [1, 5, 25, 50, 75, 95, 99]
    report["hand_command_percentiles"] = {h: {p: np.percentile(np.array(v), p, axis=0).round(4).tolist() for p in pct} for h, v in cmds.items()}
    report["hand_state_percentiles"] = {h: {p: np.percentile(np.array(v), p, axis=0).round(4).tolist() for p in pct} for h, v in states.items()}
    nz = np.array([x for x in mags if x > 1e-6]) if mags else np.zeros(1)
    report["taxel_force_percentiles_nonzero"] = {p: float(np.percentile(nz, p)) for p in pct}
    report["taxel_nonzero_fraction"] = float(len(nz) / max(len(mags), 1))
    write_json(out / "inspect.json", report)
    for h, (_, fo) in best.items():
        if fo is None:
            continue
        frames, overlay = fo
        for view in frames:
            Image.fromarray(frames[view]).save(out / f"mostclosed_{h}_{view}_plain.png")
            Image.fromarray(overlay[view]).save(out / f"mostclosed_{h}_{view}_overlay.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
