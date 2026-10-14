"""Judge backends: Claude (reference) and an open-weight VLM served by vLLM."""

from __future__ import annotations

import base64
import io
import json
import time

import numpy as np
from PIL import Image

from tdex.cost import CostLedger
from tdex.judge.prompt import Part, build_request
from tdex.judge.schema import VERDICT_SCHEMA, parse_verdict
from tdex.types import BoundaryPacket, Verdict


def png_b64(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


class Judge:
    name = "judge"

    def __init__(self, condition: str):
        self.condition = condition

    def __call__(self, packet: BoundaryPacket) -> Verdict:
        raise NotImplementedError


class ClaudeJudge(Judge):
    def __init__(self, condition: str, ledger: CostLedger, model: str = "claude-haiku-5-5", effort: str = "low", max_tokens: int = 4096):
        super().__init__(condition)
        import anthropic

        self.client = anthropic.Anthropic()  # credentials from the cluster environment
        self.ledger = ledger
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.name = f"claude:{model}"

    @staticmethod
    def to_content(parts: list[Part]) -> list[dict]:
        out = []
        for kind, val in parts:
            if kind == "text":
                out.append({"type": "text", "text": val})
            else:
                out.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64(val)}})
        return out

    def __call__(self, packet: BoundaryPacket) -> Verdict:
        self.ledger.check()
        system, parts = build_request(packet, self.condition)
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": self.to_content(parts)}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
        )
        self.ledger.record(
            "judge",
            self.model,
            resp.usage.input_tokens,
            resp.usage.output_tokens,
            {"condition": self.condition, "latency_s": time.time() - t0, "stop_reason": resp.stop_reason, **packet.meta},
        )
        if resp.stop_reason == "refusal":
            return Verdict("unclear", "yes", "judge refused", self.name)
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return parse_verdict(text, self.name)


class OpenVLMJudge(Judge):
    """Qwen-VL class model behind a vLLM OpenAI-compatible server (free per call)."""

    def __init__(self, condition: str, base_url: str, model: str, call_log: str | None = None, max_tokens: int = 512):
        super().__init__(condition)
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model
        self.max_tokens = max_tokens
        self.call_log = call_log
        self.name = f"vlm:{model}"

    @staticmethod
    def to_content(parts: list[Part]) -> list[dict]:
        out = []
        for kind, val in parts:
            if kind == "text":
                out.append({"type": "text", "text": val})
            else:
                out.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64(val)}"}})
        return out

    def __call__(self, packet: BoundaryPacket) -> Verdict:
        system, parts = build_request(packet, self.condition)
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": self.to_content(parts)}],
            response_format={"type": "json_schema", "json_schema": {"name": "verdict", "schema": VERDICT_SCHEMA}},
        )
        if self.call_log:
            u = resp.usage
            with open(self.call_log, "a") as f:
                f.write(json.dumps({"ts": time.time(), "purpose": "judge", "model": self.model, "usd": 0.0,
                                    "input_tokens": u.prompt_tokens if u else None,
                                    "output_tokens": u.completion_tokens if u else None,
                                    "latency_s": time.time() - t0, "condition": self.condition, **packet.meta}) + "\n")
        return parse_verdict(resp.choices[0].message.content or "", self.name)
