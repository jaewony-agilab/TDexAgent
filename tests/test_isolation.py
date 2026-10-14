"""Privileged state is read only by the recorder and the oracle (tdex.privileged), which only
scripts/ may import. The judge path never imports simulator code."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "tdex"

# modules that must never see the simulator or privileged state
JUDGE_PATH = ["judge", "render.py", "reader.py", "planner.py", "gate.py", "boundary.py",
              "recovery.py", "camera.py", "types.py", "harness.py", "executor.py", "cost.py"]


def imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def files(rel):
    p = PKG / rel
    return sorted(p.rglob("*.py")) if p.is_dir() else [p]


def test_only_scripts_import_privileged():
    for f in PKG.rglob("*.py"):
        if "privileged" in f.relative_to(PKG).parts:
            continue
        bad = [m for m in imports(f) if m.startswith("tdex.privileged")]
        assert not bad, f"{f.relative_to(ROOT)} imports {bad}"


@pytest.mark.parametrize("rel", JUDGE_PATH)
def test_judge_path_has_no_sim_or_privileged_imports(rel):
    forbidden = ("tdex.sim", "tdex.privileged", "tdex.packets", "mujoco", "robosuite", "robocasa")
    for f in files(rel):
        bad = [m for m in imports(f) if m.startswith(forbidden)]
        assert not bad, f"{f.relative_to(ROOT)} imports {bad}"


def test_sim_package_does_not_read_object_state():
    """tdex.sim must not touch objects / success checks; that is the recorder's job."""
    banned = ("objects", "obj_body_id", "_check_success", "_check_grasp", "object_utils")
    for f in (PKG / "sim").rglob("*.py"):
        src = f.read_text()
        hits = [b for b in banned if b in src]
        assert not hits, f"{f.relative_to(ROOT)} references {hits}"


def test_packet_has_no_privileged_fields():
    from dataclasses import fields

    from tdex.types import BoundaryPacket

    names = {f.name for f in fields(BoundaryPacket)}
    assert not names & {"label", "truth", "privileged", "obj_pos", "grasped", "record"}
