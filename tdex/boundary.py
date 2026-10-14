"""Subtask boundaries from the executor's hand-closure command.

A boundary is defined on the action space, not on contacts: the aggregate
finger-joint command crosses a closure threshold (with hysteresis), then a
settle window runs until the measured finger joints stop moving.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tdex.types import Boundary, Hand


@dataclass
class ClosureConfig:
    flexion_dims: tuple[int, ...]  # command dims that flex fingers (excludes thumb yaw)
    open_value: float  # command value of a fully open finger
    closed_value: float  # command value of a fully closed finger
    close_threshold: float = 0.6  # normalized closure to enter "closed"
    open_threshold: float = 0.3  # normalized closure to re-enter "open"
    settle_vel_eps: float = 1e-3  # max |Δq| per step to count as still [rad]
    settle_steps: int = 3  # consecutive still steps
    max_settle_steps: int = 30  # emit anyway after this many steps

    def closure(self, command: np.ndarray) -> float:
        c = np.asarray(command, dtype=float)[list(self.flexion_dims)]
        return float(np.mean((c - self.open_value) / (self.closed_value - self.open_value)))


class HandBoundaryDetector:
    def __init__(self, hand: Hand, cfg: ClosureConfig):
        self.hand = hand
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.closed = False
        self._pending: str | None = None
        self._pending_steps = 0
        self._still = 0
        self._prev_q: np.ndarray | None = None
        self.last_closure = 0.0

    def update(self, step: int, command: np.ndarray, joint_pos: np.ndarray) -> Boundary | None:
        cfg = self.cfg
        closure = cfg.closure(command)
        self.last_closure = closure
        q = np.asarray(joint_pos, dtype=float)
        vel = 0.0 if self._prev_q is None else float(np.max(np.abs(q - self._prev_q)))
        self._prev_q = q

        crossed = None
        if not self.closed and closure > cfg.close_threshold:
            self.closed, crossed = True, "close"
        elif self.closed and closure < cfg.open_threshold:
            self.closed, crossed = False, "open"
        if crossed is not None:  # a new crossing restarts the settle window
            self._pending, self._pending_steps, self._still = crossed, 0, 0
            return None

        if self._pending is None:
            return None
        self._pending_steps += 1
        self._still = self._still + 1 if vel < cfg.settle_vel_eps else 0
        if self._still >= cfg.settle_steps or self._pending_steps >= cfg.max_settle_steps:
            btype, self._pending = self._pending, None
            return Boundary(hand=self.hand, boundary_type=btype, step=step)
        return None
