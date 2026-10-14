"""Checks against real MuJoCo: projection convention and taxel force sign. Skipped without mujoco;
the rendering test is skipped without an offscreen GL backend."""

import os

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from tdex.camera import Camera  # noqa: E402
from tdex.sim.taxels import TaxelSensor  # noqa: E402

LINKS = {"right": {"index": "R_index_intermediate_link"}}

XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 .1" rgba=".2 .2 .2 1"/>
    <camera name="cam" pos="0.6 -1.2 1.5" xyaxes="1 0.3 0 -0.2 0.6 0.8" fovy="60"/>
    <body name="gripper0_right_R_index_intermediate_link" pos="0 0 0.05">
      <freejoint/>
      <geom type="box" size=".05 .05 .05" mass="1" rgba="0 0 1 1"/>
    </body>
    <body name="marker" pos="0.3 0.2 0.4" mocap="true">
      <geom type="sphere" size=".03" rgba="1 0 0 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def settle(m, d, n=500):
    for _ in range(n):
        mujoco.mj_step(m, d)


def test_taxel_force_sign_and_magnitude():
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    settle(m, d)
    s = TaxelSensor(m, d, LINKS, robot_prefixes=["gripper0_", "robot0_"])
    f = s.read()[("right", "index")]
    assert f[2] == pytest.approx(9.81, rel=0.05)  # floor pushes the link up with its weight
    assert abs(f[0]) < 0.2 and abs(f[1]) < 0.2
    assert np.allclose(s.anchors()[("right", "index")], d.xpos[1], atol=1e-6)


def test_taxel_ignores_robot_self_contact():
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    settle(m, d)
    # treat the floor's world body as robot: floor contact must then be ignored
    s = TaxelSensor(m, d, LINKS, robot_prefixes=["gripper0_", "world"])
    assert np.linalg.norm(s.read()[("right", "index")]) < 1e-9


def test_taxel_link_must_be_robot():
    m = mujoco.MjModel.from_xml_string(XML)
    with pytest.raises(ValueError):
        TaxelSensor(m, mujoco.MjData(m), LINKS, robot_prefixes=["robot0_"])


def test_projection_matches_mujoco_render():
    os.environ.setdefault("MUJOCO_GL", "egl")
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    W, H = 160, 120
    try:
        r = mujoco.Renderer(m, H, W)
    except Exception as e:  # no offscreen GL here
        pytest.skip(f"no renderer: {e}")
    try:
        r.update_scene(d, "cam")
        img = r.render()  # row 0 = top, same as the env's flipped frames
    finally:
        r.close()
    red = (img[..., 0] > 150) & (img[..., 1] < 80) & (img[..., 2] < 80)
    assert red.sum() > 5
    ys, xs = np.nonzero(red)
    cid = 0
    cam = Camera(pos=d.cam_xpos[cid].copy(), xmat=d.cam_xmat[cid].reshape(3, 3).copy(),
                 fovy_deg=float(m.cam_fovy[cid]), width=W, height=H)
    uv, depth = cam.project(np.array([0.3, 0.2, 0.4]))
    assert depth > 0
    assert np.allclose(uv, [xs.mean() + 0.5, ys.mean() + 0.5], atol=1.5)
