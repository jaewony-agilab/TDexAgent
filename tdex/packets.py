"""Save / load boundary packets (Phase 1 boundary set). Privileged records are stored
beside the packet and loaded separately so the judge path never receives them."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tdex.types import Boundary, BoundaryPacket, FingerSummary


def save_packet(root: str | Path, packet: BoundaryPacket, record: dict | None) -> Path:
    m = packet.meta
    d = Path(root) / f"{m['task']}" / f"ep{m['episode']:03d}_s{packet.boundary.step:04d}_{packet.boundary.hand}_{packet.boundary.boundary_type}"
    d.mkdir(parents=True, exist_ok=True)
    arrays = {f"plain__{k}": v for k, v in packet.frames.items()}
    arrays |= {f"overlay__{k}": v for k, v in packet.overlay_frames.items()}
    np.savez_compressed(d / "frames.npz", **arrays)
    (d / "packet.json").write_text(json.dumps({
        "boundary": asdict(packet.boundary),
        "finger_summaries": [asdict(s) for s in packet.finger_summaries],
        "task_instruction": packet.task_instruction,
        "retry_count": packet.retry_count,
        "meta": packet.meta,
    }))
    if record is not None:
        (d / "privileged.json").write_text(json.dumps(record))
    return d


def load_packet(d: str | Path) -> BoundaryPacket:
    d = Path(d)
    j = json.loads((d / "packet.json").read_text())
    z = np.load(d / "frames.npz")
    plain = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("plain__")}
    overlay = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("overlay__")}
    return BoundaryPacket(
        boundary=Boundary(**j["boundary"]),
        frames=plain,
        overlay_frames=overlay,
        finger_summaries=[FingerSummary(**{**s, "force_now": tuple(s["force_now"])}) for s in j["finger_summaries"]],
        task_instruction=j["task_instruction"],
        retry_count=j["retry_count"],
        meta=j["meta"],
    )


def load_privileged(d: str | Path) -> dict:
    return json.loads((Path(d) / "privileged.json").read_text())
