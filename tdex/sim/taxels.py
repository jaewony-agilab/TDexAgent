"""Taxels on fingertip and palm links of both hands, read from MuJoCo contacts every step.

One taxel per link: the summed world-frame force that everything outside the
robot exerts on that link. Contact partner identity is never exposed. The
benchmark and RLDX-1 are not modified; this only reads mjData.
"""

from __future__ import annotations

import mujoco
import numpy as np


def resolve_body(model: "mujoco.MjModel", suffix: str) -> int:
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
    hits = [i for i, n in enumerate(names) if n == suffix or n.endswith("_" + suffix)]
    if len(hits) != 1:
        raise KeyError(f"taxel link {suffix!r}: expected 1 body match, got {[names[i] for i in hits]}")
    return hits[0]


class TaxelSensor:
    def __init__(self, model, data, links: dict[str, dict[str, str]], robot_prefixes: list[str]):
        """links: hand -> finger -> body-name suffix (see configs/gr1.yaml)."""
        self.links, self.robot_prefixes = links, robot_prefixes
        self.bind(model, data)

    def bind(self, model, data) -> None:
        """(Re)resolve ids. Call after every env reset: robocasa rebuilds the MjModel per episode."""
        links, robot_prefixes = self.links, self.robot_prefixes
        self.m, self.d = model, data
        self.body_to_key: dict[int, tuple[str, str]] = {}
        self.anchor_geom: dict[tuple[str, str], int] = {}
        for hand, fingers in links.items():
            for finger, suffix in fingers.items():
                b = resolve_body(model, suffix)
                self.body_to_key[b] = (hand, finger)
                geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b
                         and (model.geom_contype[g] or model.geom_conaffinity[g])]
                if not geoms:
                    raise KeyError(f"taxel link {suffix!r} has no collision geom")
                self.anchor_geom[(hand, finger)] = geoms[0]
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
        self.robot_body = np.array([any(n.startswith(p) for p in robot_prefixes) for n in names])
        missing = [names[b] for b in self.body_to_key if not self.robot_body[b]]
        if missing:
            raise ValueError(f"taxel links not covered by robot_prefixes {robot_prefixes}: {missing}")
        self.keys = list(self.anchor_geom)
        self._f6 = np.zeros(6)

    def read(self) -> dict[tuple[str, str], np.ndarray]:
        m, d = self.m, self.d
        out = {k: np.zeros(3) for k in self.keys}
        for i in range(d.ncon):
            c = d.contact[i]
            if c.efc_address < 0:
                continue
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            k1, k2 = self.body_to_key.get(b1), self.body_to_key.get(b2)
            if k1 is None and k2 is None:
                continue
            mujoco.mj_contactForce(m, d, i, self._f6)
            # contact frame rows: normal (geom1 -> geom2), tangents; force acts on geom2, reaction on geom1
            fw = c.frame.reshape(3, 3).T @ self._f6[:3]
            if k2 is not None and not self.robot_body[b1]:
                out[k2] += fw
            if k1 is not None and not self.robot_body[b2]:
                out[k1] -= fw
        return out

    def anchors(self) -> dict[tuple[str, str], np.ndarray]:
        """Fixed per-link anchor: the link's collision-geom center (forward kinematics of robot qpos)."""
        return {k: np.array(self.d.geom_xpos[g]) for k, g in self.anchor_geom.items()}
