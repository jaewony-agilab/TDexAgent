"""Gate rule (code, not prompt).

| judge                                   | gate                         |
|-----------------------------------------|------------------------------|
| success                                 | continue                     |
| fail, valid                             | scripted local recovery (≤N) |
| fail, invalid / unclear twice / cap hit | replan with the reason text  |
| budget cap                              | abort (raised as BudgetExceeded, handled by the harness) |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from tdex.types import Hand, Verdict


class Action(str, Enum):
    CONTINUE = "continue"
    RECOVER = "recover"
    REPLAN = "replan"
    ABORT = "abort"


@dataclass
class GateState:
    retries: dict[str, int] = field(default_factory=lambda: {"left": 0, "right": 0})
    consecutive_unclear: int = 0


class Gate:
    def __init__(self, max_retries: int = 2, recovery_enabled: bool = True):
        self.max_retries = max_retries
        self.recovery_enabled = recovery_enabled  # ablation: no recovery -> fail-valid aborts
        self.state = GateState()

    def reset(self) -> None:
        self.state = GateState()

    def decide(self, hand: Hand, verdict: Verdict) -> tuple[Action, str]:
        s = self.state
        if verdict.outcome == "success":
            s.consecutive_unclear = 0
            return Action.CONTINUE, ""
        if verdict.outcome == "unclear":
            s.consecutive_unclear += 1
            if s.consecutive_unclear >= 2:
                s.consecutive_unclear = 0
                return Action.REPLAN, f"judge unclear twice: {verdict.reason}"
            return Action.CONTINUE, ""
        s.consecutive_unclear = 0
        if verdict.plan_still_valid == "no":
            return Action.REPLAN, verdict.reason
        if not self.recovery_enabled:
            return Action.ABORT, f"fail (recovery disabled): {verdict.reason}"
        if s.retries[hand] >= self.max_retries:
            return Action.REPLAN, f"recovery cap {self.max_retries} hit on {hand} hand: {verdict.reason}"
        s.retries[hand] += 1
        return Action.RECOVER, verdict.reason
