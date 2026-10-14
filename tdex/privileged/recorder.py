"""Ground-truth labels at boundaries, read from simulator object state.

Labels: close -> carried | empty ; open -> placed | dropped | carried (release failed).
Raw facts are stored too so labels can be recomputed offline.
"""

from __future__ import annotations

import numpy as np

from tdex.types import Boundary, Verdict


def boundary_facts(env, b: Boundary) -> dict:
    rs = env.rs
    import robocasa.utils.object_utils as OU

    facts: dict = {}
    obj = rs.objects.get("obj")
    if obj is None:
        return {"label": "n/a", "note": "env has no 'obj'"}
    facts["grasped"] = bool(rs._check_grasp(gripper=rs.robots[0].gripper[b.hand], object_geoms=obj))
    facts["grasped_other_hand"] = bool(
        rs._check_grasp(gripper=rs.robots[0].gripper["left" if b.hand == "right" else "right"], object_geoms=obj)
    )
    facts["obj_pos"] = np.asarray(rs.sim.data.body_xpos[rs.obj_body_id["obj"]]).round(4).tolist()
    if "container" in rs.objects:
        facts["in_receptacle"] = bool(OU.check_obj_in_receptacle(rs, "obj", "container"))
    facts["task_success_now"] = bool(rs._check_success())

    if b.boundary_type == "close":
        facts["label"] = "carried" if facts["grasped"] else "empty"
    elif facts["grasped"]:
        facts["label"] = "carried"
    else:
        placed = facts.get("in_receptacle", facts["task_success_now"])
        facts["label"] = "placed" if placed else "dropped"
    return facts


def truth_outcome(boundary_type: str, label: str) -> str:
    if label == "n/a":
        return "unclear"
    if boundary_type == "close":
        return "success" if label == "carried" else "fail"
    return "success" if label == "placed" else "fail"


class Recorder:
    def __init__(self, env):
        self.env = env
        self.records: list[dict] = []

    def record(self, b: Boundary, meta: dict) -> dict:
        facts = boundary_facts(self.env, b)
        rec = {**meta, "hand": b.hand, "boundary_type": b.boundary_type, "step": b.step, **facts,
               "truth": truth_outcome(b.boundary_type, facts["label"])}
        self.records.append(rec)
        return rec


class OracleJudge:
    """Upper bound: reads the recorder's label for the current boundary."""

    name = "oracle"

    def __init__(self, recorder: Recorder):
        self.recorder = recorder

    def __call__(self, packet) -> Verdict:
        rec = self.recorder.records[-1]
        assert rec["step"] == packet.boundary.step and rec["hand"] == packet.boundary.hand
        label, btype = rec["label"], rec["boundary_type"]
        outcome = truth_outcome(btype, label)
        # a failed grasp or a failed release can be retried locally; a drop needs a replan
        valid = "no" if (btype == "open" and label == "dropped") else "yes"
        return Verdict(outcome, valid, f"oracle label: {label}", self.name)
