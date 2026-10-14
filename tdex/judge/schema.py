from __future__ import annotations

import json
import re

from tdex.types import Verdict

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "enum": ["success", "fail", "unclear"]},
        "plan_still_valid": {"type": "string", "enum": ["yes", "no"]},
        "reason": {"type": "string"},
    },
    "required": ["outcome", "plan_still_valid", "reason"],
    "additionalProperties": False,
}


def parse_verdict(text: str, source: str = "") -> Verdict:
    """Parse a judge reply. Malformed output becomes `unclear` rather than an exception."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0) if m else text)
        outcome = d["outcome"]
        valid = d["plan_still_valid"]
        if outcome not in ("success", "fail", "unclear") or valid not in ("yes", "no"):
            raise ValueError
        return Verdict(outcome=outcome, plan_still_valid=valid, reason=str(d.get("reason", "")), source=source)
    except (ValueError, KeyError, TypeError, AttributeError):
        return Verdict(outcome="unclear", plan_still_valid="yes", reason=f"unparseable judge output: {text[:200]}", source=source)
