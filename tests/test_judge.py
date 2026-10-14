from types import SimpleNamespace

import numpy as np
import pytest

from tdex.judge.backends import ClaudeJudge, OpenVLMJudge
from tdex.judge.prompt import build_request, images_in
from tdex.judge.schema import parse_verdict
from tdex.judge.touch import TouchOnlyJudge
from tdex.types import Verdict


def test_parse_verdict():
    v = parse_verdict('{"outcome": "fail", "plan_still_valid": "no", "reason": "slipped"}', "x")
    assert (v.outcome, v.plan_still_valid, v.reason, v.source) == ("fail", "no", "slipped", "x")
    assert parse_verdict('noise {"outcome":"success","plan_still_valid":"yes","reason":""} tail').outcome == "success"
    for bad in ("", "not json", '{"outcome": "maybe", "plan_still_valid": "yes"}', '{"outcome": "fail"}'):
        assert parse_verdict(bad).outcome == "unclear"


def texts(parts):
    return "\n".join(p for k, p in parts if k == "text")


def test_conditions_differ_only_in_touch_channel(packet):
    _, plain = build_request(packet, "plain")
    _, text = build_request(packet, "text")
    _, over = build_request(packet, "overlay")
    assert "contact sensors" not in texts(plain) and "legend" not in texts(plain).lower()
    assert "right thumb: contact" in texts(text)
    assert "legend" in texts(over).lower() and "right thumb" not in texts(over)
    assert all(int(i.max()) == 0 for i in images_in(plain))
    assert all(int(i.max()) == 1 for i in images_in(over))
    assert len(images_in(plain)) == len(images_in(over)) == 2


def test_previous_frame_included(packet):
    packet.prev_plain_frame = np.zeros((8, 8, 3), np.uint8)
    packet.prev_overlay_frame = np.ones((8, 8, 3), np.uint8)
    packet.prev_verdict = Verdict("fail", "yes", "thumb slipped")
    _, parts = build_request(packet, "overlay")
    assert "thumb slipped" in texts(parts)
    assert len(images_in(parts)) == 3 and images_in(parts)[0].shape == (8, 8, 3)


def test_unknown_condition(packet):
    with pytest.raises(ValueError):
        build_request(packet, "depth")


class FakeLedger:
    def __init__(self):
        self.calls = []
        self.checked = 0

    def check(self):
        self.checked += 1

    def record(self, purpose, model, i, o, extra=None):
        self.calls.append((purpose, model, i, o))


def test_claude_judge_request_shape(packet):
    captured = {}

    def create(**kw):
        captured.update(kw)
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=100, output_tokens=20), stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text='{"outcome":"success","plan_still_valid":"yes","reason":"held"}')])

    j = ClaudeJudge.__new__(ClaudeJudge)
    j.condition, j.ledger, j.model, j.effort, j.max_tokens, j.name = "overlay", FakeLedger(), "claude-haiku-5-5", "low", 4096, "c"
    j.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    v = j(packet)
    assert v.outcome == "success"
    assert captured["model"] == "claude-haiku-5-5"
    assert captured["output_config"]["format"]["type"] == "json_schema"
    content = captured["messages"][0]["content"]
    assert sum(c["type"] == "image" for c in content) == 2
    assert content[[c["type"] for c in content].index("image")]["source"]["media_type"] == "image/png"
    assert j.ledger.checked == 1 and j.ledger.calls[0][:2] == ("judge", "claude-haiku-5-5")


def test_claude_judge_refusal_is_unclear(packet):
    j = ClaudeJudge.__new__(ClaudeJudge)
    j.condition, j.ledger, j.model, j.effort, j.max_tokens, j.name = "plain", FakeLedger(), "m", "low", 10, "c"
    j.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        usage=SimpleNamespace(input_tokens=1, output_tokens=1), stop_reason="refusal", content=[])))
    assert j(packet).outcome == "unclear"


def test_vlm_content_format():
    parts = [("text", "hi"), ("image", np.zeros((4, 4, 3), np.uint8))]
    c = OpenVLMJudge.to_content(parts)
    assert c[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_touch_only(packet):
    assert TouchOnlyJudge()(packet).outcome == "success"  # thumb + index in contact on close
    for s in packet.finger_summaries:
        if s.finger == "thumb":
            s.in_contact_now = False
    assert TouchOnlyJudge()(packet).outcome == "fail"
