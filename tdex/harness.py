"""Episode loop: executor -> boundary detector -> tactile reader -> renderer -> judge -> gate
-> scripted recovery or planner.

Everything sim-specific is injected (env, sensor, frame maker, recorder), so the
loop runs against fakes in tests. The judge only ever receives a BoundaryPacket.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol

import numpy as np

from tdex.boundary import ClosureConfig, HandBoundaryDetector
from tdex.cost import BudgetExceeded
from tdex.gate import Action, Gate
from tdex.planner import Plan
from tdex.reader import TactileReader
from tdex.recovery import RecoveryPrimitive
from tdex.types import HANDS, Boundary, BoundaryPacket, Verdict

StepAction = dict[str, np.ndarray]
FrameFn = Callable[[Boundary, dict, dict, dict], tuple[dict, dict]]


class Env(Protocol):
    def reset(self, episode_id: int) -> dict: ...
    def step(self, action: StepAction) -> tuple: ...
    def instruction(self, obs: dict) -> str: ...


@dataclass
class HarnessConfig:
    max_steps: int = 720  # RLDX-1 GR-1 eval: --max_episode_steps 720
    stop_on_success: bool = True  # SR is "success at any step", so stopping early does not change it
    recovery_mode: str = "primitive"  # primitive | resume (ablation: re-query the policy instead)
    scene_view: str = "scene"


@dataclass
class EpisodeResult:
    task: str
    episode: int
    condition: str
    success: bool = False
    steps: int = 0
    policy_queries: int = 0
    aborted: str = ""
    n_recoveries: int = 0
    n_replans: int = 0
    boundaries: list[dict] = field(default_factory=list)
    wall_s: float = 0.0
    plan: list[dict] = field(default_factory=list)


class Harness:
    def __init__(
        self,
        env: Env,
        executor,
        batch_fn: Callable[[dict], dict],
        closure: dict[str, ClosureConfig],
        cfg: HarnessConfig,
        condition: str,
        task: str,
        sensor: Callable[[], dict] | None = None,
        frame_fn: FrameFn | None = None,
        judge: Callable[[BoundaryPacket], Verdict] | None = None,
        planner=None,
        gate: Gate | None = None,
        recorder: Callable[[Boundary, dict], dict] | None = None,
        packet_sink: Callable[[BoundaryPacket, dict | None], None] | None = None,
        reader: TactileReader | None = None,
    ):
        self.env, self.executor, self.batch_fn = env, executor, batch_fn
        self.cfg, self.condition, self.task = cfg, condition, task
        self.sensor, self.frame_fn, self.judge, self.planner = sensor, frame_fn, judge, planner
        self.gate = gate or Gate()
        self.recorder, self.packet_sink = recorder, packet_sink
        self.reader = reader or TactileReader()
        self.detectors = {h: HandBoundaryDetector(h, closure[h]) for h in HANDS}
        self.recovery = {h: RecoveryPrimitive(closure[h]) for h in HANDS}

    def run_episode(self, episode: int) -> EpisodeResult:
        t0 = time.time()
        res = EpisodeResult(task=self.task, episode=episode, condition=self.condition)
        obs = self.env.reset(episode)
        self.executor.reset(f"{self.task}_ep{episode}_{uuid.uuid4().hex[:8]}")
        for h in HANDS:
            self.detectors[h].reset()
            self.recovery[h].reset()
        self.reader.reset()
        self.gate.reset()
        instruction = self.env.instruction(obs)
        meta = {"task": self.task, "episode": episode, "condition": self.condition}

        plan = Plan()
        try:
            if self.planner is not None:
                plan = self.planner.plan(instruction, meta)
                res.plan = [asdict(s) for s in plan.steps]
        except BudgetExceeded:
            res.aborted = "budget"
            return res

        queue: list[StepAction] = []
        rqueue: list[StepAction] = []
        recovering = {h: False for h in HANDS}
        prev: dict[str, tuple] = {}
        history: list[str] = []

        for step in range(self.cfg.max_steps):
            if rqueue:
                action, src = rqueue.pop(0), "recovery"
            else:
                if not queue:
                    queue = self.executor.act(self.batch_fn(obs))
                    res.policy_queries += 1
                action, src = queue.pop(0), "policy"
            obs, _, terminated, truncated, success, _ = self.env.step(action)
            res.steps = step + 1
            if src == "policy":
                for h in HANDS:
                    self.recovery[h].record(action)
            res.success |= success
            if res.success and self.cfg.stop_on_success:
                break

            readings = self.sensor() if self.sensor else {}
            self.reader.update(readings)

            stop = False
            for hand in HANDS:
                b = self.detectors[hand].update(step, action[f"action.{hand}_hand"], obs[f"state.{hand}_hand"][-1])
                if b is None:
                    continue
                if recovering[hand]:
                    if b.boundary_type == "open":
                        continue  # the recovery's own release is not a task event
                    recovering[hand] = False
                entry, verdict, gate_action, reason = self._handle_boundary(b, obs, readings, instruction, plan, prev, meta)
                res.boundaries.append(entry)
                if verdict is None:
                    if gate_action == Action.ABORT:
                        res.aborted, stop = "budget", True
                    continue
                history.append(f"step {b.step} {hand} {b.boundary_type}: {verdict.outcome} ({verdict.reason})")
                if verdict.outcome == "success":
                    plan.advance(hand, b.boundary_type)

                if gate_action == Action.RECOVER:
                    res.n_recoveries += 1
                    queue = []
                    if self.cfg.recovery_mode == "primitive":
                        rqueue = self.recovery[hand].build(hand)
                        recovering[hand] = bool(rqueue)
                elif gate_action == Action.REPLAN:
                    res.n_replans += 1
                    queue = []
                    if self.planner is not None:
                        try:
                            decision, new_plan = self.planner.replan(instruction, reason, history, meta)
                        except BudgetExceeded:
                            res.aborted, stop = "budget", True
                            break
                        entry["replan"] = asdict(decision)
                        if new_plan.steps:
                            plan = new_plan
                        if decision.action == "abort":
                            res.aborted, stop = "planner", True
                        elif decision.action == "recover_then_resume" and decision.recover_hand in HANDS:
                            rqueue = self.recovery[decision.recover_hand].build(decision.recover_hand)
                            recovering[decision.recover_hand] = bool(rqueue)
                elif gate_action == Action.ABORT:
                    res.aborted, stop = "gate", True
            if stop or terminated or truncated:
                break

        res.wall_s = time.time() - t0
        return res

    def _handle_boundary(self, b, obs, readings, instruction, plan, prev, meta):
        summaries = self.reader.summarize(b.hand)
        lost = {(b.hand, s.finger): s.contact_lost for s in summaries}
        frames, overlay = self.frame_fn(b, obs, readings, lost) if self.frame_fn else ({}, {})
        bmeta = {**meta, "step": b.step, "hand": b.hand, "boundary_type": b.boundary_type}
        record = self.recorder(b, bmeta) if self.recorder else None
        entry = {
            "step": b.step,
            "hand": b.hand,
            "boundary_type": b.boundary_type,
            "fingers": [asdict(s) for s in summaries],
            "privileged": record,
        }
        p = prev.get(b.hand)
        packet = BoundaryPacket(
            boundary=b,
            frames=frames,
            overlay_frames=overlay,
            finger_summaries=summaries,
            task_instruction=instruction,
            retry_count=self.gate.state.retries[b.hand],
            expected_subtask=plan.expected(b.hand, b.boundary_type),
            prev_plain_frame=p[0] if p else None,
            prev_overlay_frame=p[1] if p else None,
            prev_verdict=p[2] if p else None,
            meta=bmeta,
        )
        if self.packet_sink:
            self.packet_sink(packet, record)
        if self.judge is None:
            return entry, None, Action.CONTINUE, ""
        try:
            verdict = self.judge(packet)
        except BudgetExceeded:
            entry["verdict"] = None
            return entry, None, Action.ABORT, "budget"
        gate_action, reason = self.gate.decide(b.hand, verdict)
        entry["verdict"] = verdict.to_dict()
        entry["gate"] = gate_action.value
        sv = self.cfg.scene_view
        prev[b.hand] = (frames.get(sv), overlay.get(sv), verdict)
        return entry, verdict, gate_action, reason
