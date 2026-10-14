#!/usr/bin/env python
"""Phase 1 step 4: offline judge comparison on a saved boundary set.

Runs one (judge backend, input condition) over every saved packet, in episode order so
each call gets the previous boundary's frame and *its own* previous verdict. Truth is read
from privileged.json by this script only; the judge never sees it.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path

from tdex import config as C
from tdex.cost import CostCaps, CostLedger
from tdex.packets import load_packet, load_privileged
from tdex.runinfo import provenance, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True, type=Path)
    ap.add_argument("--condition", required=True, choices=("plain", "text", "overlay"))
    ap.add_argument("--judge", required=True, choices=("claude", "vlm"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--config", default=str(C.REPO / "configs/gr1.yaml"))
    ap.add_argument("--costcaps", default=os.environ.get("TDEX_COSTCAPS", str(C.REPO / "configs/costcaps.yaml")))
    ap.add_argument("--vlm-url", default=os.environ.get("TDEX_VLM_URL", "http://127.0.0.1:8000/v1"))
    args = ap.parse_args()
    cfg = C.load(args.config)
    prov = provenance(args.config, vars(args))
    args.out.mkdir(parents=True, exist_ok=True)

    ledger = None
    if args.judge == "claude":
        from tdex.judge.backends import ClaudeJudge

        run_id = f"{prov['slurm_job_id'] or 'local'}_offline_{args.condition}"
        ledger = CostLedger(CostCaps.load(args.costcaps), run_id, args.out / "calls.jsonl")
        judge = ClaudeJudge(args.condition, ledger, cfg["judge"]["claude_model"], cfg["judge"]["claude_effort"])
    else:
        from tdex.judge.backends import OpenVLMJudge

        judge = OpenVLMJudge(args.condition, args.vlm_url, cfg["judge"]["vlm_model"], call_log=str(args.out / "calls.jsonl"))

    dirs = sorted(p.parent for p in args.packets.rglob("packet.json"))  # names sort by task/ep/step
    if args.limit:
        dirs = dirs[: args.limit]
    prev: dict = {}
    rows = []
    with open(args.out / "verdicts.jsonl", "w") as f:
        for d in dirs:
            pk = load_packet(d)
            key = (pk.meta["task"], pk.meta["episode"], pk.boundary.hand)
            p = prev.get(key)
            if p:
                pk = replace(pk, prev_plain_frame=p[0], prev_overlay_frame=p[1], prev_verdict=p[2])
            v = judge(pk)
            prev[key] = (pk.frames.get("scene"), pk.overlay_frames.get("scene"), v)
            truth = load_privileged(d)
            row = {"packet": str(d.relative_to(args.packets)), **v.to_dict(), "truth": truth["truth"], "label": truth["label"],
                   "boundary_type": pk.boundary.boundary_type, "task": pk.meta["task"]}
            rows.append(row)
            f.write(json.dumps(row) + "\n")

    scored = [r for r in rows if r["truth"] != "unclear"]
    correct = sum(r["outcome"] == r["truth"] for r in scored)
    summary = {
        **prov,
        "judge": judge.name,
        "condition": args.condition,
        "n": len(rows),
        "n_scored": len(scored),
        "accuracy": correct / len(scored) if scored else None,
        "unclear_rate": sum(r["outcome"] == "unclear" for r in scored) / len(scored) if scored else None,
        "confusion": {f"{t}->{o}": n for (t, o), n in Counter((r["truth"], r["outcome"]) for r in scored).items()},
        "cost": ledger.summary() if ledger else None,
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("judge", "condition", "n_scored", "accuracy")}))


if __name__ == "__main__":
    main()
