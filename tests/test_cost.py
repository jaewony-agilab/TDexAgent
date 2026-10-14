import json

import pytest
import yaml

from tdex.cost import BudgetExceeded, CostCaps, CostLedger


def caps_file(tmp_path, per_run=1.0, glob=2.0):
    p = tmp_path / "caps.yaml"
    p.write_text(yaml.safe_dump({"per_run_usd": per_run, "global_usd": glob,
                                 "global_ledger": str(tmp_path / "ledger.jsonl"),
                                 "prices_per_mtok": {"m": {"input": 1.0, "output": 5.0}}}))
    return p


def test_missing_cap_file_refuses(tmp_path):
    with pytest.raises(FileNotFoundError):
        CostCaps.load(tmp_path / "nope.yaml")
    with pytest.raises(FileNotFoundError):
        CostCaps.load(None)


def test_per_run_cap_and_logging(tmp_path):
    led = CostLedger(CostCaps.load(caps_file(tmp_path)), "run1", tmp_path / "calls.jsonl")
    led.check()
    usd = led.record("judge", "m", 500_000, 100_000)  # 0.5 + 0.5
    assert usd == pytest.approx(1.0)
    with pytest.raises(BudgetExceeded):
        led.check()
    rows = [json.loads(l) for l in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert rows[0]["purpose"] == "judge" and rows[0]["run_id"] == "run1"
    assert led.summary()["calls_by_purpose"] == {"judge": 1}


def test_global_cap_spans_runs(tmp_path):
    caps = CostCaps.load(caps_file(tmp_path, per_run=10.0, glob=1.5))
    CostLedger(caps, "a").record("plan", "m", 1_000_000, 0)
    b = CostLedger(caps, "b")
    b.check()
    b.record("plan", "m", 600_000, 0)
    with pytest.raises(BudgetExceeded):
        b.check()


def test_unknown_model_price(tmp_path):
    led = CostLedger(CostCaps.load(caps_file(tmp_path)), "r")
    with pytest.raises(KeyError):
        led.record("judge", "other", 1, 1)
