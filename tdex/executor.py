"""Executor adapter: frozen RLDX-1 behind its ZeroMQ policy server."""

from __future__ import annotations

import numpy as np

StepAction = dict[str, np.ndarray]


class RLDXExecutor:
    def __init__(self, host: str, port: int, n_action_steps: int = 16, timeout_ms: int = 120000):
        from rldx.policy.server_client import PolicyClient

        self.client = PolicyClient(host=host, port=port, timeout_ms=timeout_ms)
        self.n_action_steps = n_action_steps
        self.session_id = ""
        self._first = True

    def modality_delta_indices(self) -> tuple[np.ndarray, np.ndarray | None]:
        mc = self.client.get_modality_config()
        video = np.array(mc["video"].delta_indices)
        state = np.array(mc["state"].delta_indices) if "state" in mc else None
        return video, state

    def reset(self, session_id: str) -> None:
        self.session_id = session_id
        self._first = True

    def act(self, obs_batched: dict) -> list[StepAction]:
        """One policy query -> list of per-step actions (the chunk, truncated to n_action_steps)."""
        options = {"reset_memory": [self._first], "session_ids": [self.session_id]}
        actions, _ = self.client.get_action(obs_batched, options=options)
        self._first = False
        horizon = min(self.n_action_steps, min(np.asarray(v).shape[1] for v in actions.values()))
        return [{k: np.asarray(v)[0, t] for k, v in actions.items()} for t in range(horizon)]
