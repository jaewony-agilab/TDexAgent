#!/usr/bin/env python
"""Phase 0 step 5a: env step time with / without taxel reads (no policy; held actions).

The "does the policy still work" half (step 5b) is run_episodes.py --condition executor
with and without --no-taxels on the same init states; see slurm/phase0_sensorcost.sbatch.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tdex import config as C
from tdex.runinfo import provenance, write_json
from tdex.sim.gr1 import GR1Env
from tdex.sim.taxels import TaxelSensor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=C.GR1_TASKS[0])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(C.REPO / "configs/gr1.yaml"))
    args = ap.parse_args()
    cfg = C.load(args.config)
    env = GR1Env(args.task, np.array([0]), np.array([0]), seed=cfg["seed"])
    tx = cfg["taxels"]
    env.reset(0)
    sensor = TaxelSensor(env.model, env.data, tx["links"], tx["robot_prefixes"])
    env.reset_hooks.append(lambda e: sensor.bind(e.model, e.data))
    hold = {k: np.zeros(s.shape, s.dtype) for k, s in env.base.action_space.items()}

    timings = {}
    for mode in ("without", "with", "without", "with"):  # interleaved to average out drift
        env.reset(0)
        t0 = time.perf_counter()
        for _ in range(args.steps):
            env.step(hold)
            if mode == "with":
                sensor.read()
        timings.setdefault(mode, []).append((time.perf_counter() - t0) / args.steps)
    t_read = []
    for _ in range(200):
        t0 = time.perf_counter()
        sensor.read()
        t_read.append(time.perf_counter() - t0)
    rep = {
        **provenance(args.config, vars(args)),
        "step_s_without": float(np.mean(timings["without"])),
        "step_s_with": float(np.mean(timings["with"])),
        "read_only_s": float(np.median(t_read)),
    }
    rep["overhead_frac"] = rep["step_s_with"] / rep["step_s_without"] - 1
    write_json(Path(args.out) / "sensor_cost.json", rep)
    print(rep)


if __name__ == "__main__":
    main()
