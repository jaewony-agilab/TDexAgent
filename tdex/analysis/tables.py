"""Local analysis over committed results/ (never run on the cluster)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_summaries(root: str | Path) -> list[dict]:
    out = []
    for p in sorted(Path(root).rglob("summary.json")):
        d = json.loads(p.read_text())
        d["_path"] = str(p.parent)
        out.append(d)
    return out


def success_table(summaries: list[dict]) -> list[dict]:
    """Pool episodes per (task, condition, judge) across seeds/jobs."""
    agg: dict = defaultdict(lambda: {"k": 0, "n": 0, "jobs": 0, "usd": 0.0, "seeds": set()})
    for s in summaries:
        if "success_rate" not in s:
            continue
        key = (s["task"], s["condition"], s.get("judge"))
        a = agg[key]
        a["k"] += s["successes"]
        a["n"] += s["episodes"]
        a["jobs"] += 1
        a["seeds"].add(s.get("seed"))
        a["usd"] += (s.get("cost") or {}).get("run_usd", 0.0)
    rows = []
    for (task, cond, judge), a in sorted(agg.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        lo, hi = wilson(a["k"], a["n"])
        rows.append({"task": task, "condition": cond, "judge": judge, "successes": a["k"], "episodes": a["n"],
                     "sr": a["k"] / a["n"] if a["n"] else math.nan, "ci95": (lo, hi), "seeds": sorted(map(str, a["seeds"])),
                     "usd": a["usd"]})
    return rows


def to_markdown(rows: list[dict]) -> str:
    lines = ["| task | condition | judge | SR | 95% CI | n | seeds | $ |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['task'].split('_GR1')[0]} | {r['condition']} | {r['judge'] or '-'} | {r['sr']:.2f} | "
                     f"[{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}] | {r['episodes']} | {','.join(r['seeds'])} | {r['usd']:.2f} |")
    return "\n".join(lines)


def select_tasks(rows: list[dict], lo: float = 0.30, hi: float = 0.70) -> list[str]:
    """Phase 0 step 6 band. The 'at least one grasp' criterion holds for all 24 GR-1 tasks (all PnP)."""
    return [r["task"] for r in rows if r["condition"] == "executor" and lo <= r["sr"] <= hi]
