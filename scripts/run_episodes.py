#!/usr/bin/env python
"""Run episodes of one task under one condition. Runs in the sim venv; needs an RLDX-1 server.

Conditions:
  executor  RLDX-1 alone (boundaries still detected and recorded; no judge/planner)
  plain     framework, judge on plain RGB          (c)
  text      framework, judge on RGB + taxel text   (Phase 1 / ablation)
  overlay   framework, judge on overlay            (t)
  oracle    framework, privileged judge            (o)
  touch     framework, touch-only gate             (ablation)

Writes a compact summary to --out (committed) and optional full packets to --packets-dir (scratch).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from tdex import config as C
from tdex.cost import CostCaps, CostLedger
from tdex.executor import RLDXExecutor
from tdex.gate import Gate
from tdex.harness import Harness, HarnessConfig
from tdex.packets import save_packet
from tdex.privileged.recorder import OracleJudge, Recorder
from tdex.reader import TactileReader
from tdex.runinfo import provenance, write_json
from tdex.sim.frames import FrameMaker
from tdex.sim.gr1 import GR1Env, batch_obs
from tdex.sim.taxels import TaxelSensor

FRAMEWORK = ("plain", "text", "overlay", "oracle", "touch")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--condition", required=True, choices=("executor",) + FRAMEWORK)
    ap.add_argument("--judge", default="claude", choices=("claude", "vlm"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--start-episode", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None, help="base seed (default from config)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(C.REPO / "configs/gr1.yaml"))
    ap.add_argument("--costcaps", default=os.environ.get("TDEX_COSTCAPS", str(C.REPO / "configs/costcaps.yaml")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("RLDX_PORT", 20100)))
    ap.add_argument("--vlm-url", default=os.environ.get("TDEX_VLM_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--no-taxels", action="store_true", help="sensor-cost check: do not read contacts")
    ap.add_argument("--no-planner", action="store_true")
    ap.add_argument("--no-recovery", action="store_true", help="ablation: fail-valid -> abort")
    ap.add_argument("--recovery-mode", default="primitive", choices=("primitive", "resume"))
    ap.add_argument("--no-wrist", action="store_true")
    ap.add_argument("--packets-dir", default=None, help="save every boundary packet here (scratch)")
    ap.add_argument("--n-sample-frames", type=int, default=6, help="annotated frames copied into --out")
    args = ap.parse_args()

    cfg = C.load(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prov = provenance(args.config, vars(args))
    run_id = f"{prov['slurm_job_id'] or 'local'}_{args.task}_{args.condition}_s{seed}"

    executor = RLDXExecutor(args.host, args.port, n_action_steps=cfg["n_action_steps"])
    vdi, sdi = executor.modality_delta_indices()
    env = GR1Env(args.task, vdi, sdi, seed=seed)
    env.reset(args.start_episode)  # build the sim once so model/data exist

    tx = cfg["taxels"]
    sensor = None
    if not args.no_taxels:
        sensor = TaxelSensor(env.model, env.data, tx["links"], tx["robot_prefixes"])
        env.reset_hooks.append(lambda e: sensor.bind(e.model, e.data))
    frame_fn = None
    if sensor is not None:
        frame_fn = FrameMaker(env, sensor, C.render_config(cfg), use_wrist=not args.no_wrist)

    recorder = Recorder(env)
    framework = args.condition in FRAMEWORK
    ledger = None
    judge = planner = None
    if framework:
        needs_api = (not args.no_planner) or (args.condition in ("plain", "text", "overlay") and args.judge == "claude")
        if needs_api:
            ledger = CostLedger(CostCaps.load(args.costcaps), run_id, out / "calls.jsonl")
        if args.condition == "oracle":
            judge = OracleJudge(recorder)
        elif args.condition == "touch":
            from tdex.judge.touch import TouchOnlyJudge

            judge = TouchOnlyJudge()
        elif args.judge == "claude":
            from tdex.judge.backends import ClaudeJudge

            judge = ClaudeJudge(args.condition, ledger, cfg["judge"]["claude_model"], cfg["judge"]["claude_effort"])
        else:
            from tdex.judge.backends import OpenVLMJudge

            judge = OpenVLMJudge(args.condition, args.vlm_url, cfg["judge"]["vlm_model"], call_log=str(out / "calls.jsonl"))
        if not args.no_planner:
            from tdex.planner import ClaudePlanner

            planner = ClaudePlanner(ledger, cfg["planner"]["model"], cfg["planner"]["effort"])
        if sensor is None and args.condition in ("text", "overlay", "touch"):
            raise SystemExit(f"condition {args.condition} needs taxels")

    sample_dir = out / "frames"
    n_saved = [0]

    def sink(packet, record):
        if args.packets_dir:
            save_packet(args.packets_dir, packet, record)
        if n_saved[0] < args.n_sample_frames and packet.overlay_frames:
            sample_dir.mkdir(exist_ok=True)
            b = packet.boundary
            for view, img in packet.overlay_frames.items():
                Image.fromarray(img).save(sample_dir / f"ep{packet.meta['episode']:03d}_s{b.step:04d}_{b.hand}_{b.boundary_type}_{view}.png")
            n_saved[0] += 1

    closure = C.closure_config(cfg)
    harness = Harness(
        env=env,
        executor=executor,
        batch_fn=batch_obs,
        closure={"left": closure, "right": closure},
        cfg=HarnessConfig(max_steps=cfg["max_episode_steps"], recovery_mode=args.recovery_mode),
        condition=args.condition,
        task=args.task,
        sensor=sensor.read if sensor else None,
        frame_fn=frame_fn,
        judge=judge,
        planner=planner,
        gate=Gate(max_retries=cfg["gate"]["max_retries"], recovery_enabled=not args.no_recovery),
        recorder=recorder.record,
        packet_sink=sink,
        reader=TactileReader(tx["contact_threshold_n"]),
    )

    results = []
    with open(out / "episodes.jsonl", "a") as f:
        for ep in range(args.start_episode, args.start_episode + args.episodes):
            r = harness.run_episode(ep)
            results.append(r)
            f.write(json.dumps(asdict(r)) + "\n")
            f.flush()
            print(f"[{args.task}] ep {ep}: success={r.success} steps={r.steps} boundaries={len(r.boundaries)} "
                  f"recov={r.n_recoveries} replans={r.n_replans} aborted={r.aborted!r} {r.wall_s:.0f}s", flush=True)

    succ = [r.success for r in results]
    summary = {
        **prov,
        "task": args.task,
        "condition": args.condition,
        "judge": None if not framework else (judge.name if hasattr(judge, "name") else str(judge)),
        "seed": seed,
        "episodes": len(results),
        "successes": int(sum(succ)),
        "success_rate": float(np.mean(succ)) if succ else None,
        "mean_steps": float(np.mean([r.steps for r in results])) if results else None,
        "mean_wall_s": float(np.mean([r.wall_s for r in results])) if results else None,
        "boundaries": int(sum(len(r.boundaries) for r in results)),
        "recoveries": int(sum(r.n_recoveries for r in results)),
        "replans": int(sum(r.n_replans for r in results)),
        "aborted": {k: sum(r.aborted == k for r in results) for k in ("gate", "planner", "budget")},
        "cost": ledger.summary() if ledger else None,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("task", "condition", "episodes", "success_rate")}))


if __name__ == "__main__":
    main()
