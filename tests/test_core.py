import numpy as np
import pytest

from tdex.boundary import HandBoundaryDetector
from tdex.camera import Camera, ImageTransform, is_occluded, make_marks
from tdex.gate import Action, Gate
from tdex.reader import TactileReader
from tdex.recovery import RecoveryPrimitive
from tdex.render import RenderConfig, render_overlay
from tdex.types import FingerMark, Verdict


# --- boundary --------------------------------------------------------------
def run_detector(det, closures, q_still_after=None):
    events = []
    q = np.zeros(6)
    for t, c in enumerate(closures):
        cmd = np.full(6, c)
        if q_still_after is None or t < q_still_after:
            q = q + 0.1  # fingers moving
        b = det.update(t, cmd, q)
        if b:
            events.append(b)
    return events


def test_close_then_open_with_settle(closure):
    det = HandBoundaryDetector("left", closure)
    traj = [0.0] * 5 + [0.8] * 15 + [0.1] * 15
    ev = run_detector(det, traj, q_still_after=None)
    # fingers never stop -> boundaries fire at max_settle_steps
    assert [e.boundary_type for e in ev] == ["close", "open"]
    assert ev[0].step == 5 + closure.max_settle_steps
    assert all(e.hand == "left" for e in ev)


def test_settles_early_when_fingers_stop(closure):
    det = HandBoundaryDetector("right", closure)
    ev = run_detector(det, [0.0] * 3 + [0.9] * 10, q_still_after=5)
    assert len(ev) == 1 and ev[0].boundary_type == "close"
    assert ev[0].step < 3 + closure.max_settle_steps


def test_hysteresis_no_chatter(closure):
    det = HandBoundaryDetector("right", closure)
    traj = [0.0, 0.7] + [0.5, 0.65, 0.45, 0.55] * 10  # stays between open and close thresholds
    ev = run_detector(det, traj)
    assert [e.boundary_type for e in ev] == ["close"]


def test_closure_uses_only_flexion_dims(closure):
    cmd = np.array([5.0, 1, 1, 1, 1, 5.0])
    assert closure.closure(cmd) == pytest.approx(1.0)


# --- reader ------------------------------------------------------------------
def test_reader_lost_and_peak():
    r = TactileReader(contact_threshold_n=0.5)
    r.update({("right", "thumb"): np.array([0, 0, 3.0])})
    s1 = {s.finger: s for s in r.summarize("right")}
    assert s1["thumb"].in_contact_now and not s1["thumb"].contact_lost
    r.update({("right", "thumb"): np.array([0, 0, 5.0])})
    r.update({("right", "thumb"): np.zeros(3)})
    s2 = {s.finger: s for s in r.summarize("right")}
    assert s2["thumb"].contact_lost and not s2["thumb"].in_contact_now
    assert s2["thumb"].peak_magnitude == pytest.approx(5.0)
    s3 = {s.finger: s for s in r.summarize("right")}
    assert not s3["thumb"].contact_lost and s3["thumb"].peak_magnitude == 0.0


def test_reader_hands_independent():
    r = TactileReader()
    r.update({("left", "index"): np.array([1.0, 0, 0])})
    r.summarize("right")
    r.update({("left", "index"): np.zeros(3)})
    left = {s.finger: s for s in r.summarize("left")}
    assert not left["index"].contact_lost  # left window never saw a boundary with contact


# --- camera ------------------------------------------------------------------
def cam(w=200, h=100):
    return Camera(pos=np.zeros(3), xmat=np.eye(3), fovy_deg=90.0, width=w, height=h)


def test_projection_conventions():
    c = cam()
    uv, d = c.project(np.array([0, 0, -2.0]))
    assert np.allclose(uv, [100, 50]) and d == pytest.approx(2.0)
    uv_r, _ = c.project(np.array([0.5, 0, -2.0]))
    uv_up, _ = c.project(np.array([0, 0.5, -2.0]))
    assert uv_r[0] > 100  # +x camera -> right
    assert uv_up[1] < 50  # +y camera (up) -> smaller row in the flipped image
    _, d_behind = c.project(np.array([0, 0, 1.0]))
    assert d_behind < 0


def test_square_pad_resize_matches_robocasa():
    t = ImageTransform.square_pad_resize(1280, 800, 256)
    assert t.pad_top == 240 and t.pad_left == 0
    assert np.allclose(t.apply(np.array([640.0, 400.0])), [128, 128])
    assert np.allclose(t.apply(np.array([0.0, 0.0])), [0, 48])


def test_occlusion():
    depth = np.full((100, 200), 1.0)
    assert is_occluded(np.array([100, 50]), 2.0, depth)
    assert not is_occluded(np.array([100, 50]), 0.995, depth)


def test_make_marks_direction_and_flags():
    c = cam()
    anchors = {("right", "index"): np.array([0, 0, -2.0]), ("right", "palm"): np.array([0, 0, 1.0])}
    forces = {("right", "index"): np.array([3.0, 0, 0])}
    marks = {m.finger: m for m in make_marks(anchors, forces, c, ImageTransform(), (200, 100),
                                             lost={("right", "index"): True})}
    assert marks["index"].direction[0] == pytest.approx(1.0, abs=1e-6)
    assert marks["index"].lost and marks["index"].magnitude == pytest.approx(3.0)
    assert not marks["palm"].in_view and marks["palm"].direction is None


# --- render ------------------------------------------------------------------
def test_render_draws_and_scales():
    img = np.full((256, 256, 3), 128, np.uint8)
    weak = render_overlay(img, [FingerMark("left", "thumb", (60, 128), (1.0, 0.0), 1.0, False, False)])
    strong = render_overlay(img, [FingerMark("left", "thumb", (60, 128), (1.0, 0.0), 50.0, False, False)])
    assert (weak != img).any()
    changed = lambda o: int(np.nonzero((o != img).any(-1).any(0))[0].max())  # rightmost changed column
    assert changed(strong) > changed(weak)
    clipped = render_overlay(img, [FingerMark("left", "thumb", (60, 128), (1.0, 0.0), 500.0, False, False)])
    assert changed(clipped) == changed(strong)  # clipped at force_clip_n


def test_render_skips_out_of_view_and_keeps_input():
    img = np.zeros((64, 64, 3), np.uint8)
    out = render_overlay(img, [FingerMark("right", "palm", (10, 10), None, 0.0, True, True, in_view=False)])
    assert (out == img).all()
    out2 = render_overlay(img, [FingerMark("right", "palm", (32, 32), None, 0.0, True, True)], RenderConfig())
    assert (out2 != img).any() and (img == 0).all()


# --- gate --------------------------------------------------------------------
V = lambda o, v="yes": Verdict(o, v, "r")


def test_gate_table():
    g = Gate(max_retries=2)
    assert g.decide("left", V("success"))[0] == Action.CONTINUE
    assert g.decide("left", V("fail"))[0] == Action.RECOVER
    assert g.decide("left", V("fail"))[0] == Action.RECOVER
    assert g.decide("left", V("fail"))[0] == Action.REPLAN  # cap hit
    assert g.decide("right", V("fail"))[0] == Action.RECOVER  # per-hand counter
    assert g.decide("right", V("fail", "no"))[0] == Action.REPLAN
    assert g.decide("right", V("unclear"))[0] == Action.CONTINUE
    assert g.decide("right", V("unclear"))[0] == Action.REPLAN
    assert g.decide("right", V("unclear"))[0] == Action.CONTINUE  # counter reset after replan


def test_gate_unclear_streak_broken_by_success():
    g = Gate()
    g.decide("left", V("unclear"))
    g.decide("left", V("success"))
    assert g.decide("left", V("unclear"))[0] == Action.CONTINUE


def test_gate_no_recovery_ablation():
    assert Gate(recovery_enabled=False).decide("left", V("fail"))[0] == Action.ABORT


# --- recovery ----------------------------------------------------------------
def test_recovery_sequence(closure):
    rec = RecoveryPrimitive(closure, retreat_steps=4, open_steps=3, close_steps=3)
    for t in range(6):
        rec.record({"action.right_arm": np.full(7, float(t)), "action.right_hand": np.full(6, 1.0),
                    "action.left_arm": np.zeros(7), "action.left_hand": np.zeros(6), "action.waist": np.zeros(3)})
    seq = rec.build("right")
    assert len(seq) == 3 + 4 + 4 + 3
    arms = [a["action.right_arm"][0] for a in seq]
    assert arms[3:7] == [5, 4, 3, 2] and arms[7:11] == [2, 3, 4, 5]
    assert seq[0]["action.right_hand"][1] == 1.0 and seq[2]["action.right_hand"][1] == 0.0
    assert seq[5]["action.right_hand"][1] == 0.0
    assert seq[-1]["action.right_hand"][1] == 1.0
    assert all((a["action.left_arm"] == 0).all() for a in seq)  # other parts held
    assert seq[0]["action.right_hand"][0] == 1.0 and seq[2]["action.right_hand"][0] == 1.0  # non-flexion dim kept


def test_recovery_empty_history(closure):
    assert RecoveryPrimitive(closure).build("left") == []
