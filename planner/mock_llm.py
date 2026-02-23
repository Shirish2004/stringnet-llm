"""Deterministic base/mock LLM and oracle planners."""

from __future__ import annotations

import numpy as np


DEFAULT_VOCAB = [
    "tighten_net",
    "widen_net",
    "flank_left",
    "flank_right",
    "split",
    "focus_largest_cluster",
    "delay_enclose",
    "increase_speed",
    "reduce_speed",
    "move_leader_to",
]


class MockLLM:
    """Rule-based frozen model returning intent logits from scene tokens."""

    def __init__(self, vocab: list[str] | None = None) -> None:
        self.vocab = vocab or DEFAULT_VOCAB
        self.index = {k: i for i, k in enumerate(self.vocab)}

    def __call__(self, scene_tokens: dict) -> np.ndarray:
        logits = np.zeros(len(self.vocab), dtype=float)
        esc = scene_tokens["escape_prob_est"]
        spread = scene_tokens["sheep_spread"]
        def add(tok: str, value: float) -> None:
            if tok in self.index:
                logits[self.index[tok]] += value

        if esc > 0.45:
            add("tighten_net", 2.0)
            add("increase_speed", 1.0)
        if spread > 0.08:
            add("focus_largest_cluster", 1.8)
            add("split", 0.8)
        if scene_tokens["ACoM"][1] > 0.6:
            add("flank_right", 1.0)
        else:
            add("flank_left", 1.0)
        if np.allclose(logits, 0):
            add("widen_net", 0.5)
        return logits


class OracularPlanner(MockLLM):
    """Corrective oracle used for adapter supervision on failure scenes."""

    def corrective_intent(self, scene_tokens: dict) -> dict:
        if scene_tokens["escape_prob_est"] > 0.3:
            intent = "tighten_net"
            speed_scale = 1.1
        elif scene_tokens["sheep_spread"] > 0.09:
            intent = "focus_largest_cluster"
            speed_scale = 1.0
        else:
            intent = "flank_left"
            speed_scale = 0.95
        return {"intent_token": intent, "phase": "herd", "params": {"speed_scale": speed_scale}}
