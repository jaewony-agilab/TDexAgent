"""Scripted local recovery for one hand: open, retreat, re-approach, re-close.

Joint-space only, no IK and no planner call. The approach direction is the
arm's own recent command trajectory: retreat replays it backwards, re-approach
replays it forwards. Every other action key is held at its last command.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from tdex.boundary import ClosureConfig
from tdex.types import Hand

StepAction = dict[str, np.ndarray]  # "action.<part>" -> (dim,)


class RecoveryPrimitive:
    def __init__(self, closure: ClosureConfig, retreat_steps: int = 20, open_steps: int = 8, close_steps: int = 8):
        self.closure = closure
        self.retreat_steps = retreat_steps
        self.open_steps = open_steps
        self.close_steps = close_steps
        self.history: deque[StepAction] = deque(maxlen=retreat_steps)

    def reset(self) -> None:
        self.history.clear()

    def record(self, action: StepAction) -> None:
        self.history.append({k: np.array(v, copy=True) for k, v in action.items()})

    def _open_cmd(self, closed_cmd: np.ndarray) -> np.ndarray:
        cmd = np.array(closed_cmd, dtype=float, copy=True)
        cmd[list(self.closure.flexion_dims)] = self.closure.open_value
        return cmd

    def build(self, hand: Hand) -> list[StepAction]:
        if not self.history:
            return []
        hist = list(self.history)
        last = hist[-1]
        arm_key, hand_key = f"action.{hand}_arm", f"action.{hand}_hand"
        closed_cmd = np.asarray(last[hand_key], float)
        open_cmd = self._open_cmd(closed_cmd)

        def step(arm=None, hand_cmd=None) -> StepAction:
            a = {k: np.array(v, copy=True) for k, v in last.items()}
            if arm is not None:
                a[arm_key] = np.array(arm, copy=True)
            if hand_cmd is not None:
                a[hand_key] = np.asarray(hand_cmd, dtype=last[hand_key].dtype)
            return a

        seq: list[StepAction] = []
        for t in np.linspace(0, 1, self.open_steps):
            seq.append(step(hand_cmd=(1 - t) * closed_cmd + t * open_cmd))
        for a in reversed(hist):
            seq.append(step(arm=a[arm_key], hand_cmd=open_cmd))
        for a in hist:
            seq.append(step(arm=a[arm_key], hand_cmd=open_cmd))
        for t in np.linspace(0, 1, self.close_steps):
            seq.append(step(hand_cmd=(1 - t) * open_cmd + t * closed_cmd))
        return seq
