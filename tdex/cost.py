"""Dollar caps for API calls (planner and API judge), and a per-call log.

The cap file is required; adapters refuse to start without it. Prices live in
the cap file (not in code) so they can be updated without a commit.

Example (configs/costcaps.example.yaml):
    per_run_usd: 2.0
    global_usd: 200.0
    global_ledger: /path/on/cluster/tdex_global_ledger.jsonl
    prices_per_mtok:
      claude-haiku-5-5: {input: 0.10, output: 0.50}
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import yaml


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CostCaps:
    per_run_usd: float
    global_usd: float
    global_ledger: Path
    prices_per_mtok: dict[str, dict[str, float]]

    @classmethod
    def load(cls, path: str | os.PathLike | None) -> "CostCaps":
        if not path or not Path(path).is_file():
            raise FileNotFoundError(
                f"cost-cap file not found ({path!r}); refusing to call a paid API without caps"
            )
        d = yaml.safe_load(Path(path).read_text())
        return cls(
            per_run_usd=float(d["per_run_usd"]),
            global_usd=float(d["global_usd"]),
            global_ledger=Path(os.path.expandvars(d["global_ledger"])),
            prices_per_mtok=d["prices_per_mtok"],
        )


class CostLedger:
    """Tracks spend for one run and appends every call to the run log and global ledger."""

    def __init__(self, caps: CostCaps, run_id: str, call_log: str | os.PathLike | None = None):
        self.caps = caps
        self.run_id = run_id
        self.run_spent = 0.0
        self.call_log = Path(call_log) if call_log else None
        self.calls_by_purpose: dict[str, int] = {}
        self.usd_by_purpose: dict[str, float] = {}

    def price(self, model: str, input_tokens: int, output_tokens: int) -> float:
        if model not in self.caps.prices_per_mtok:
            raise KeyError(f"no price for model {model!r} in cost-cap file")
        p = self.caps.prices_per_mtok[model]
        return (input_tokens * p["input"] + output_tokens * p["output"]) / 1e6

    def global_spent(self) -> float:
        path = self.caps.global_ledger
        if not path.exists():
            return 0.0
        total = 0.0
        with open(path) as f:
            for line in f:
                if line.strip():
                    total += json.loads(line)["usd"]
        return total

    def check(self) -> None:
        if self.run_spent >= self.caps.per_run_usd:
            raise BudgetExceeded(f"per-run cap ${self.caps.per_run_usd} reached")
        if self.global_spent() >= self.caps.global_usd:
            raise BudgetExceeded(f"global cap ${self.caps.global_usd} reached")

    def record(self, purpose: str, model: str, input_tokens: int, output_tokens: int, extra: dict | None = None) -> float:
        usd = self.price(model, input_tokens, output_tokens)
        self.run_spent += usd
        self.calls_by_purpose[purpose] = self.calls_by_purpose.get(purpose, 0) + 1
        self.usd_by_purpose[purpose] = self.usd_by_purpose.get(purpose, 0.0) + usd
        entry = {
            "ts": time.time(),
            "run_id": self.run_id,
            "purpose": purpose,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "usd": usd,
            **(extra or {}),
        }
        line = json.dumps(entry) + "\n"
        self.caps.global_ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(self.caps.global_ledger, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line)
            fcntl.flock(f, fcntl.LOCK_UN)
        if self.call_log:
            self.call_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.call_log, "a") as f:
                f.write(line)
        return usd

    def summary(self) -> dict:
        return {
            "run_usd": self.run_spent,
            "calls_by_purpose": dict(self.calls_by_purpose),
            "usd_by_purpose": dict(self.usd_by_purpose),
        }
