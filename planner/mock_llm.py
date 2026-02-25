from __future__ import annotations
import numpy as np


DEFAULT_VOCAB = [
    "tighten_net", "widen_net", "flank_left", "flank_right",
    "split", "focus_largest_cluster", "delay_enclose",
    "increase_speed", "reduce_speed", "move_leader_to",
]

STROMBOM_VOCAB = [
    "tighten_collect", "loosen_collect", "push_harder", "back_off",
    "spread_formation", "compress_formation", "speed_up", "slow_down",
    "flank_left", "flank_right",
]


class MockLLM:
    """Rule-based frozen model returning StringNet intent logits from scene tokens."""

    def __init__(self, vocab: list[str] | None = None) -> None:
        self.vocab = vocab or DEFAULT_VOCAB
        self.index = {k: i for i, k in enumerate(self.vocab)}

    def __call__(self, scene_tokens: dict) -> np.ndarray:
        logits = np.zeros(len(self.vocab), dtype=float)
        esc = scene_tokens["escape_prob_est"]
        spread = scene_tokens["sheep_spread"]

        def add(tok: str, val: float) -> None:
            if tok in self.index:
                logits[self.index[tok]] += val

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
    """Corrective oracle used for StringNet adapter supervision on failure scenes."""

    def corrective_intent(self, scene_tokens: dict) -> dict:
        if scene_tokens["escape_prob_est"] > 0.3:
            tok, speed = "tighten_net", 1.1
        elif scene_tokens["sheep_spread"] > 0.09:
            tok, speed = "focus_largest_cluster", 1.0
        else:
            tok, speed = "flank_left", 0.95
        return {"intent_token": tok, "phase": "herd", "params": {"speed_scale": speed}}


class StrombomMockLLM:
    """Rule-based frozen model returning Strömbom parameter-adjustment intent logits."""

    def __init__(self, vocab: list[str] | None = None) -> None:
        self.vocab = vocab or STROMBOM_VOCAB
        self.index = {k: i for i, k in enumerate(self.vocab)}

    def __call__(self, scene_tokens: dict) -> np.ndarray:
        logits = np.zeros(len(self.vocab), dtype=float)
        esc = scene_tokens["escape_prob_est"]
        spread = scene_tokens["sheep_spread"]
        acom_y = scene_tokens["ACoM"][1]

        def add(tok: str, val: float) -> None:
            if tok in self.index:
                logits[self.index[tok]] += val

        if esc > 0.5:
            add("tighten_collect", 2.0)
            add("compress_formation", 1.5)
            add("speed_up", 1.2)
        elif esc > 0.3:
            add("push_harder", 1.8)
            add("tighten_collect", 0.8)
        if spread > 0.12:
            add("spread_formation", 1.5)
            add("loosen_collect", 1.0)
        elif spread < 0.04:
            add("compress_formation", 0.8)
            add("push_harder", 1.0)
        if acom_y > 0.6:
            add("flank_right", 1.0)
        else:
            add("flank_left", 1.0)
        if np.allclose(logits, 0):
            add("back_off", 0.5)
        return logits


class StrombomOracle(StrombomMockLLM):
    """Corrective oracle for Strömbom adapter supervision on failure scenes."""

    def corrective_intent(self, scene_tokens: dict) -> dict:
        esc = scene_tokens["escape_prob_est"]
        spread = scene_tokens["sheep_spread"]
        if esc > 0.45:
            tok, speed = "tighten_collect", 1.2
        elif spread > 0.10:
            tok, speed = "spread_formation", 1.05
        else:
            tok, speed = "push_harder", 1.1
        return {"intent_token": tok, "params": {"speed_scale": speed}}