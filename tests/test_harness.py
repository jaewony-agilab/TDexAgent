import numpy as np

from tdex.cost import BudgetExceeded
from tdex.gate import Gate
from tdex.harness import Harness, HarnessConfig
from tdex.packets import load_packet, load_privileged, save_packet
from tdex.planner import Plan, PlanStep, ReplanDecision
from tdex.types import Verdict


class FakeEnv:
    def __init__(self, success_at=None):
        self.success_at = success_at

    def _obs(self):
        return {"state.left_hand": self.q["left"][None].copy(), "state.right_hand": self.q["right"][None].copy(),
                "video.ego": np.zeros((1, 8, 8, 3), np.uint8), "annotation.human.task_description": "pick the can"}

    def reset(self, episode_id):
        self.t = 0
        self.q = {"left": np.zeros(6), "right": np.zeros(6)}
        return self._obs()

    def step(self, a):
        self.t += 1
        for h in ("left", "right"):
            self.q[h] = np.asarray(a[f"action.{h}_hand"], float)  # fingers track instantly
        succ = self.success_at is not None and self.t >= self.success_at
        return self._obs(), 0.0, False, False, succ, {}

    def instruction(self, obs):
        return obs["annotation.human.task_description"]


class FakeExecutor:
    """Right hand closes during policy steps [20, 60)."""

    def __init__(self):
        self.calls = 0

    def reset(self, session_id):
        self.k = 0

    def act(self, obs):
        self.calls += 1
        out = []
        for _ in range(16):
            c = 1.0 if 20 <= self.k < 60 else 0.0
            out.append({"action.right_hand": np.full(6, c), "action.left_hand": np.zeros(6),
                        "action.right_arm": np.full(7, self.k / 100), "action.left_arm": np.zeros(7),
                        "action.waist": np.zeros(3)})
            self.k += 1
        return out


class ScriptedJudge:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.packets = []

    def __call__(self, packet):
        self.packets.append(packet)
        o = self.outcomes.pop(0) if self.outcomes else ("success", "yes")
        return Verdict(o[0], o[1], "scripted", "fake")


def harness(closure, judge=None, planner=None, success_at=None, recorder=None, sink=None, max_steps=120, **kw):
    frames = lambda b, obs, readings, lost: ({"scene": np.zeros((8, 8, 3), np.uint8)}, {"scene": np.ones((8, 8, 3), np.uint8)})
    return Harness(env=FakeEnv(success_at), executor=FakeExecutor(), batch_fn=lambda o: o,
                   closure={"left": closure, "right": closure}, cfg=HarnessConfig(max_steps=max_steps, **kw),
                   condition="test", task="T", sensor=lambda: {("right", "thumb"): np.array([0, 0, 2.0])},
                   frame_fn=frames, judge=judge, planner=planner, gate=Gate(max_retries=2),
                   recorder=recorder, packet_sink=sink)


def test_executor_alone_records_boundaries(closure):
    recs = []
    h = harness(closure, recorder=lambda b, m: recs.append((b, m)) or {"label": "carried"})
    r = h.run_episode(0)
    assert [b["boundary_type"] for b in r.boundaries] == ["close", "open"]
    assert all(b["hand"] == "right" for b in r.boundaries)
    assert r.boundaries[0]["privileged"] == {"label": "carried"}
    assert r.steps == 120 and not r.success and r.policy_queries == 8
    assert len(recs) == 2 and recs[0][1]["episode"] == 0


def test_stop_on_success(closure):
    r = harness(closure, success_at=30).run_episode(1)
    assert r.success and r.steps == 30


def test_recovery_then_rejudge(closure):
    judge = ScriptedJudge([("fail", "yes"), ("success", "yes")])
    r = harness(closure, judge=judge, max_steps=200).run_episode(0)
    gates = [b.get("gate") for b in r.boundaries]
    assert gates[0] == "recover" and r.n_recoveries == 1
    # the recovery's own open is suppressed; the next judged event is the re-close
    assert r.boundaries[1]["boundary_type"] == "close" and gates[1] == "continue"
    assert judge.packets[1].retry_count == 1
    assert judge.packets[1].prev_verdict.outcome == "fail"
    assert judge.packets[1].prev_overlay_frame is not None
    assert "privileged" not in judge.packets[0].meta


def test_resume_mode_requeries_policy(closure):
    judge = ScriptedJudge([("fail", "yes")])
    h = harness(closure, judge=judge, recovery_mode="resume")
    r = h.run_episode(0)
    assert r.n_recoveries == 1 and r.boundaries[1]["boundary_type"] == "open"


def test_budget_abort(closure):
    def broke(packet):
        raise BudgetExceeded("cap")

    r = harness(closure, judge=broke).run_episode(0)
    assert r.aborted == "budget" and len(r.boundaries) == 1


class FakePlanner:
    def __init__(self, action="abort"):
        self.action = action
        self.replans = []

    def plan(self, instruction, meta):
        return Plan([PlanStep("right", "close", "grasp the can"), PlanStep("right", "open", "release in drawer")])

    def replan(self, instruction, reason, history, meta):
        self.replans.append((reason, list(history)))
        return ReplanDecision(self.action, "none", "x"), Plan()


def test_replan_abort_and_expected_subtask(closure):
    judge = ScriptedJudge([("fail", "no")])
    planner = FakePlanner("abort")
    r = harness(closure, judge=judge, planner=planner).run_episode(0)
    assert r.aborted == "planner" and r.n_replans == 1
    assert judge.packets[0].expected_subtask == "grasp the can"
    assert planner.replans[0][1][0].startswith("step")
    assert r.plan[0]["description"] == "grasp the can"


def test_plan_advances_on_success(closure):
    judge = ScriptedJudge([("success", "yes"), ("success", "yes")])
    harness(closure, judge=judge, planner=FakePlanner()).run_episode(0)
    assert judge.packets[1].expected_subtask == "release in drawer"


def test_packet_roundtrip(tmp_path, packet):
    d = save_packet(tmp_path, packet, {"label": "carried", "truth": "success"})
    p2 = load_packet(d)
    assert p2.boundary == packet.boundary and p2.task_instruction == packet.task_instruction
    assert (p2.overlay_frames["scene"] == packet.overlay_frames["scene"]).all()
    assert p2.finger_summaries[0] == packet.finger_summaries[0]
    assert load_privileged(d)["label"] == "carried"
