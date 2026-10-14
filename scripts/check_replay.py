#!/usr/bin/env python
"""Phase 0 step 2: can the released GR-1 demos be replayed from stored sim state?

Two inputs are supported because it is not yet known which form the demos come in:

  --lerobot DIR   LeRobot v2.1 dataset (GR00T-X-Embodiment-Sim style). Reports whether any
                  feature carries simulator state; without it, post-hoc replay is impossible.
  --hdf5 FILE     robomimic/robocasa hdf5 (data/demo_*/states, actions, attrs model_file).
                  For each demo: (a) per-step state restore + mj_forward -> are contact forces
                  available (sufficient for post-hoc taxels); (b) open-loop action replay from
                  the first state -> max state divergence (dynamic determinism).

Writes a yes/no verdict JSON to --out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tdex import config as C
from tdex.runinfo import provenance, write_json

STATE_HINTS = ("sim_state", "mujoco", "qpos", "qvel", "model_file", "states", "env_state")


def check_lerobot(root: Path) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text())
    feats = info.get("features", {})
    hits = [k for k in feats if any(h in k.lower() for h in STATE_HINTS)]
    extra = [p.name for p in root.rglob("*") if p.suffix in (".hdf5", ".h5", ".xml")][:20]
    return {"features": sorted(feats), "state_like_features": hits, "sim_files": extra,
            "replayable_from_state": bool(hits or extra)}


def check_hdf5(path: Path, n_demos: int, cfg: dict) -> dict:
    import h5py
    import robosuite
    import robocasa  # noqa: F401
    from tdex.sim.taxels import TaxelSensor

    f = h5py.File(path, "r")
    env_args = json.loads(f["data"].attrs["env_args"])
    kw = dict(env_args["env_kwargs"])
    kw.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    rs = robosuite.make(env_args["env_name"], **kw)
    tx = cfg["taxels"]
    out = []
    for name in sorted(f["data"].keys())[:n_demos]:
        g = f["data"][name]
        states, actions = g["states"][()], g["actions"][()]
        rs.reset()
        rs.reset_from_xml_string(rs.edit_model_xml(g.attrs["model_file"]))
        sim = rs.sim
        sensor = TaxelSensor(sim.model._model, sim.data._data, tx["links"], tx["robot_prefixes"])
        # (a) per-step restore -> contact forces
        nonzero = 0
        for s in states:
            sim.set_state_from_flattened(s)
            sim.forward()
            nonzero += any(np.linalg.norm(v) > 1e-3 for v in sensor.read().values())
        # (b) open-loop replay
        sim.set_state_from_flattened(states[0])
        sim.forward()
        errs = []
        for t, a in enumerate(actions[:-1]):
            rs.step(a)
            errs.append(float(np.max(np.abs(sim.get_state().flatten() - states[t + 1]))))
        out.append({"demo": name, "T": len(states), "steps_with_taxel_force": int(nonzero),
                    "openloop_max_err": max(errs) if errs else 0.0, "openloop_final_err": errs[-1] if errs else 0.0})
    return {
        "demos": out,
        "state_restore_gives_contacts": all(d["steps_with_taxel_force"] > 0 for d in out),
        "openloop_deterministic": all(d["openloop_max_err"] < 1e-6 for d in out),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot", type=Path)
    ap.add_argument("--hdf5", type=Path)
    ap.add_argument("--n-demos", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(C.REPO / "configs/gr1.yaml"))
    args = ap.parse_args()
    cfg = C.load(args.config)
    rep = provenance(args.config, vars(args))
    if args.lerobot:
        rep["lerobot"] = check_lerobot(args.lerobot)
    if args.hdf5:
        rep["hdf5"] = check_hdf5(args.hdf5, args.n_demos, cfg)
    rep["verdict_replay_from_stored_state"] = bool(
        rep.get("hdf5", {}).get("state_restore_gives_contacts") or rep.get("lerobot", {}).get("replayable_from_state")
    )
    write_json(Path(args.out) / "replay_check.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k.startswith("verdict")}))


if __name__ == "__main__":
    main()
