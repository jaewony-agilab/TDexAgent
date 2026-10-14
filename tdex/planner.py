"""LLM planner (Claude Haiku), called once at episode start and on replan.

The executor is a frozen VLA fine-tuned on the benchmark's own instruction
strings, so the planner never rewrites the executor instruction (see
DECISIONS.md). It produces (a) the expected step list that the judge is told
about, and (b) on replan, one of a fixed set of actions the harness can execute.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from tdex.cost import CostLedger
from tdex.types import Hand

STEP = {
    "type": "object",
    "properties": {
        "hand": {"type": "string", "enum": ["left", "right", "either"]},
        "event": {"type": "string", "enum": ["close", "open"]},
        "description": {"type": "string"},
    },
    "required": ["hand", "event", "description"],
    "additionalProperties": False,
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {"steps": {"type": "array", "items": STEP}},
    "required": ["steps"],
    "additionalProperties": False,
}
REPLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["resume", "recover_then_resume", "abort"]},
        "recover_hand": {"type": "string", "enum": ["left", "right", "none"]},
        "reason": {"type": "string"},
        "steps": {"type": "array", "items": STEP},
    },
    "required": ["action", "recover_hand", "reason", "steps"],
    "additionalProperties": False,
}

PLAN_SYSTEM = (
    "You plan for a humanoid robot with two dexterous hands at a table. A learned policy executes the whole "
    "task from its instruction; you cannot change what it does, only describe the expected sequence of hand "
    "events so that a monitor can check each one. List the hand close (grasp) and open (release) events in order, "
    "each with the hand if it can be inferred (else 'either') and a short description. Reply with JSON only."
)
REPLAN_SYSTEM = (
    "You supervise a learned robot policy. A monitor flagged a problem that a local retry cannot fix, or that "
    "local retries failed to fix. Choose one action: 'resume' (let the policy continue from the current state; "
    "it often re-attempts on its own), 'recover_then_resume' (first open the given hand, back off and re-approach, "
    "then let the policy continue), or 'abort' (the episode cannot succeed, e.g. the object is out of reach or "
    "broken). Also return the remaining expected hand events. Reply with JSON only."
)


@dataclass
class PlanStep:
    hand: str
    event: str
    description: str


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)
    cursor: int = 0

    def expected(self, hand: Hand, event: str) -> str:
        for s in self.steps[self.cursor:]:
            if s.event == event and s.hand in (hand, "either"):
                return s.description
        return ""

    def advance(self, hand: Hand, event: str) -> None:
        for i in range(self.cursor, len(self.steps)):
            s = self.steps[i]
            if s.event == event and s.hand in (hand, "either"):
                self.cursor = i + 1
                return


@dataclass
class ReplanDecision:
    action: str
    recover_hand: str
    reason: str


class ClaudePlanner:
    def __init__(self, ledger: CostLedger, model: str = "claude-haiku-5-5", effort: str = "low", max_tokens: int = 4096):
        import anthropic

        self.client = anthropic.Anthropic()
        self.ledger = ledger
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def _call(self, purpose: str, system: str, user: str, schema: dict, meta: dict) -> dict:
        self.ledger.check()
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        self.ledger.record(purpose, self.model, resp.usage.input_tokens, resp.usage.output_tokens,
                           {"latency_s": time.time() - t0, "stop_reason": resp.stop_reason, **meta})
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def plan(self, instruction: str, meta: dict) -> Plan:
        d = self._call("plan", PLAN_SYSTEM, f"Task instruction: {instruction}", PLAN_SCHEMA, meta)
        return Plan([PlanStep(**s) for s in d.get("steps", [])])

    def replan(self, instruction: str, reason: str, history: list[str], meta: dict) -> tuple[ReplanDecision, Plan]:
        user = (
            f"Task instruction: {instruction}\nProblem reported by the monitor: {reason}\n"
            "Monitor history this episode (oldest first):\n" + ("\n".join(history) or "(none)")
        )
        d = self._call("replan", REPLAN_SYSTEM, user, REPLAN_SCHEMA, meta)
        if not d:
            return ReplanDecision("resume", "none", "planner output unparseable; resuming"), Plan()
        return (
            ReplanDecision(d["action"], d["recover_hand"], d["reason"]),
            Plan([PlanStep(**s) for s in d.get("steps", [])]),
        )
