from __future__ import annotations

import json
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .adapter_train import (
    AdapterMLP,
    ReplayBuffer,
    DEFAULT_VOCAB,
    STROMBOM_VOCAB,
    SCENE_DIM,
    intent_to_idx,
    scene_to_tensor,
    train_adapter,
)
from .mock_llm import MockLLM, OracularPlanner, StrombomMockLLM, StrombomOracle


_STRINGNET_SYSTEM = (
    "You are a high-level planner for StringNet herding robots. "
    "Defenders form a semi-circular net around sheep and herd them to a goal region. "
    "Respond with ONLY a JSON object — no markdown, no explanation. "
    "Required keys: intent_token (string), phase (string), radius_scale (float), speed_scale (float). "
    "intent_token must be one of: tighten_net, widen_net, flank_left, flank_right, "
    "split, focus_largest_cluster, delay_enclose, increase_speed, reduce_speed, move_leader_to. "
    "phase must be one of: seek, enclose, herd. "
    "radius_scale range: 0.8 to 1.3. speed_scale range: 0.8 to 1.3."
)

_STROMBOM_SYSTEM = (
    "You are a high-level planner for Strömbom herding dogs. "
    "Dogs alternate between collecting stray sheep and driving the herd toward a goal. "
    "Respond with ONLY a JSON object — no markdown, no explanation. "
    "Required keys: intent_token (string), collect_radius_scale (float), "
    "drive_offset_scale (float), formation_radius_scale (float), speed_scale (float). "
    "intent_token must be one of: tighten_collect, loosen_collect, push_harder, back_off, "
    "spread_formation, compress_formation, speed_up, slow_down, flank_left, flank_right. "
    "All scale values range: 0.7 to 1.5."
)

_STRINGNET_USER_TEMPLATE = (
    "Scene state:\n"
    "  ACoM (normalized): ({acom_x:.4f}, {acom_y:.4f})\n"
    "  sheep_spread: {spread:.4f}\n"
    "  largest_cluster_dist: {cluster_dist:.4f}\n"
    "  escape_prob_est: {escape_prob:.4f}\n"
    "  obstacle_density_nearby: {obstacle:.4f}\n"
    "  current_phase: {phase}\n"
    "Select the optimal StringNet intent."
)

_STROMBOM_USER_TEMPLATE = (
    "Scene state:\n"
    "  ACoM (normalized): ({acom_x:.4f}, {acom_y:.4f})\n"
    "  sheep_spread: {spread:.4f}\n"
    "  largest_cluster_dist: {cluster_dist:.4f}\n"
    "  escape_prob_est: {escape_prob:.4f}\n"
    "  obstacle_density_nearby: {obstacle:.4f}\n"
    "  current_strombom_phase: {phase}\n"
    "Select the optimal Strömbom parameter-adjustment intent."
)


class QwenPlanner:
    """Generic HuggingFace Qwen planner; system prompt and vocab injected at construction."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        system_prompt: str = _STRINGNET_SYSTEM,
        user_template: str = _STRINGNET_USER_TEMPLATE,
        vocab: list[str] | None = None,
        device: Optional[str] = None,
        max_new_tokens: int = 128,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.vocab = vocab or DEFAULT_VOCAB
        self.max_new_tokens = max_new_tokens
        self.available = False

        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if resolved == "cuda" else None,
                trust_remote_code=True,
            )
            if resolved != "cuda":
                self.model = self.model.to(self.device)
            self.model.eval()
            self.available = True
        except Exception as exc:
            print(f"[QwenPlanner] Failed to load {model_name}: {exc}")
            self.tokenizer = None
            self.model = None

    def _build_messages(self, scene_tokens: dict, phase: str) -> list[dict]:
        user_msg = self.user_template.format(
            acom_x=scene_tokens["ACoM"][0],
            acom_y=scene_tokens["ACoM"][1],
            spread=scene_tokens["sheep_spread"],
            cluster_dist=scene_tokens["largest_cluster_dist"],
            escape_prob=scene_tokens["escape_prob_est"],
            obstacle=scene_tokens["obstacle_density_nearby"],
            phase=phase,
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    def _parse_response(self, raw: str) -> Optional[dict]:
        raw = raw.strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        try:
            data = json.loads(raw[start:end])
            tok = data.get("intent_token", "")
            if tok not in self.vocab:
                return None
            return data | {"intent_token": tok}
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    def generate(self, scene_tokens: dict, phase: str = "seek") -> Optional[dict]:
        """Run Qwen inference and return parsed intent dict or None on failure."""
        if not self.available or self.model is None or self.tokenizer is None:
            return None
        messages = self._build_messages(scene_tokens, phase)
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            gen = out_ids[0][inputs["input_ids"].shape[-1]:]
            return self._parse_response(self.tokenizer.decode(gen, skip_special_tokens=True))
        except Exception as exc:
            print(f"[QwenPlanner] Inference error: {exc}")
            return None

    def to_logits(self, scene_tokens: dict, phase: str = "seek") -> Optional[np.ndarray]:
        """Convert Qwen intent to a one-hot logit spike over the vocabulary."""
        result = self.generate(scene_tokens, phase)
        if result is None:
            return None
        logits = np.zeros(len(self.vocab), dtype=np.float32)
        logits[self.vocab.index(result["intent_token"])] = 5.0
        return logits


class LLMPlanner:
    """Combines Qwen, rule-based fallback, and online PyTorch adapter for StringNet intent."""

    def __init__(
        self,
        vocab: Optional[list[str]] = None,
        adapter_dim: int = 64,
        qwen_model: str = "Qwen/Qwen3-0.6B",
        qwen_device: Optional[str] = None,
        buffer_capacity: int = 256,
        device: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.vocab = vocab or DEFAULT_VOCAB
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.qwen = QwenPlanner(
            model_name=qwen_model,
            system_prompt=_STRINGNET_SYSTEM,
            user_template=_STRINGNET_USER_TEMPLATE,
            vocab=self.vocab,
            device=qwen_device,
        )
        self.base_llm = MockLLM(self.vocab)
        self.adapter = AdapterMLP(SCENE_DIM, adapter_dim, len(self.vocab))
        self.adapter.to(self.device).eval()
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.update_logs: list[dict] = []
        self.snapshots: deque = deque(maxlen=10)

    def _adapter_logits(self, scene_tokens: dict) -> np.ndarray:
        x = scene_to_tensor(scene_tokens).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.adapter(x)
        return out.squeeze(0).cpu().numpy()

    def _decode(self, logits: np.ndarray, source: str) -> dict:
        idx = int(np.argmax(logits))
        tok = self.vocab[idx]
        phase = "seek"
        if tok in {"tighten_net", "focus_largest_cluster", "increase_speed"}:
            phase = "herd"
        elif tok in {"split", "delay_enclose"}:
            phase = "enclose"
        return {
            "formation_type": "semi-ellipse",
            "params": {
                "radius_scale": 0.9 if tok == "tighten_net" else 1.1 if tok == "widen_net" else 1.0,
                "speed_scale": 1.1 if tok == "increase_speed" else 0.9 if tok == "reduce_speed" else 1.0,
            },
            "assignments": {"leader": 0},
            "phase": phase,
            "intent_token": tok,
            "logits": logits.tolist(),
            "source": source,
        }

    def plan(self, scene_tokens: dict, current_phase: str = "seek") -> dict:
        """Compute herding intent from Qwen + rule base + adapter delta."""
        base = self.base_llm(scene_tokens)
        adap = self._adapter_logits(scene_tokens)
        qwen_out = self.qwen.to_logits(scene_tokens, phase=current_phase)
        if qwen_out is not None:
            logits = 0.4 * base + 0.4 * qwen_out + 0.2 * adap
            source = "qwen+adapter"
        else:
            logits = base + adap
            source = "rule+adapter"
        return self._decode(logits, source)

    def push_failure(self, scene_tokens: dict, oracle_intent: dict) -> None:
        """Add one corrective example to the replay buffer."""
        self.buffer.push(scene_to_tensor(scene_tokens), intent_to_idx(oracle_intent, self.vocab))

    def update_adapter(self, lr: float = 5e-4, epochs: int = 3, batch_size: int = 16) -> list[float]:
        """Run one round of adapter training on the current replay buffer."""
        return train_adapter(self.adapter, self.buffer, lr=lr, epochs=epochs,
                             batch_size=batch_size, device=self.device)

    def save_snapshot(self, directory: str = "shepherd_codex/checkpoints") -> Path:
        """Serialise adapter weights to disk as a timestamped .pt file."""
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        path = p / f"adapter_{ts}.pt"
        torch.save(self.adapter.state_dict(), path)
        self.snapshots.append(path)
        return path

    def load_snapshot(self, path: Path) -> None:
        """Restore adapter weights from a .pt checkpoint."""
        self.adapter.load_state_dict(torch.load(path, map_location=self.device))
        self.adapter.eval()

    def logged_update(
        self,
        failing_scene: dict,
        oracle_intent: dict,
        lr: float = 5e-4,
        epochs: int = 3,
        **kwargs,
    ) -> dict:
        """Push failure, retrain adapter, log pre/post intent change and return log dict."""
        pre = self.plan(failing_scene)
        snap = self.save_snapshot()
        self.push_failure(failing_scene, oracle_intent)
        losses = self.update_adapter(lr=lr, epochs=epochs)
        post = self.plan(failing_scene)
        log = {
            "timestamp": datetime.utcnow().isoformat(),
            "failing_scene_tokens": deepcopy(failing_scene),
            "pre_plan": pre["intent_token"],
            "oracle_intent": oracle_intent["intent_token"],
            "loss_history": losses,
            "post_plan": post["intent_token"],
            "buffer_size": len(self.buffer),
            "snapshot": str(snap),
            "qwen_available": self.qwen.available,
        }
        self.update_logs.append(log)
        Path("shepherd_codex/checkpoints").mkdir(parents=True, exist_ok=True)
        with open("shepherd_codex/checkpoints/update_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
        return log


_STROMBOM_TOKEN_PARAMS: dict[str, dict] = {
    "tighten_collect":   {"collect_radius_scale": 0.70, "drive_offset_scale": 1.00, "formation_radius_scale": 1.00, "speed_scale": 1.05, "flank_bias": 0.0},
    "loosen_collect":    {"collect_radius_scale": 1.35, "drive_offset_scale": 1.00, "formation_radius_scale": 1.10, "speed_scale": 0.95, "flank_bias": 0.0},
    "push_harder":       {"collect_radius_scale": 1.00, "drive_offset_scale": 0.70, "formation_radius_scale": 1.00, "speed_scale": 1.10, "flank_bias": 0.0},
    "back_off":          {"collect_radius_scale": 1.00, "drive_offset_scale": 1.35, "formation_radius_scale": 1.10, "speed_scale": 0.90, "flank_bias": 0.0},
    "spread_formation":  {"collect_radius_scale": 1.00, "drive_offset_scale": 1.00, "formation_radius_scale": 1.30, "speed_scale": 1.00, "flank_bias": 0.0},
    "compress_formation":{"collect_radius_scale": 1.00, "drive_offset_scale": 1.00, "formation_radius_scale": 0.75, "speed_scale": 1.05, "flank_bias": 0.0},
    "speed_up":          {"collect_radius_scale": 1.00, "drive_offset_scale": 1.00, "formation_radius_scale": 1.00, "speed_scale": 1.25, "flank_bias": 0.0},
    "slow_down":         {"collect_radius_scale": 1.00, "drive_offset_scale": 1.00, "formation_radius_scale": 1.00, "speed_scale": 0.80, "flank_bias": 0.0},
    "flank_left":        {"collect_radius_scale": 1.00, "drive_offset_scale": 1.00, "formation_radius_scale": 1.00, "speed_scale": 1.00, "flank_bias": -1.2},
    "flank_right":       {"collect_radius_scale": 1.00, "drive_offset_scale": 1.00, "formation_radius_scale": 1.00, "speed_scale": 1.00, "flank_bias":  1.2},
}


class StrombomLLMPlanner:
    """Combines Qwen, Strömbom rule base, and online adapter for Strömbom parameter intent."""

    def __init__(
        self,
        vocab: Optional[list[str]] = None,
        adapter_dim: int = 64,
        qwen_model: str = "Qwen/Qwen3-0.6B",
        qwen_device: Optional[str] = None,
        buffer_capacity: int = 256,
        device: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.vocab = vocab or STROMBOM_VOCAB
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.qwen = QwenPlanner(
            model_name=qwen_model,
            system_prompt=_STROMBOM_SYSTEM,
            user_template=_STROMBOM_USER_TEMPLATE,
            vocab=self.vocab,
            device=qwen_device,
        )
        self.base_llm = StrombomMockLLM(self.vocab)
        self.adapter = AdapterMLP(SCENE_DIM, adapter_dim, len(self.vocab))
        self.adapter.to(self.device).eval()
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.update_logs: list[dict] = []
        self.snapshots: deque = deque(maxlen=10)

    def _adapter_logits(self, scene_tokens: dict) -> np.ndarray:
        x = scene_to_tensor(scene_tokens).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.adapter(x)
        return out.squeeze(0).cpu().numpy()

    def _decode(self, logits: np.ndarray, source: str) -> dict:
        idx = int(np.argmax(logits))
        tok = self.vocab[idx]
        p = _STROMBOM_TOKEN_PARAMS.get(tok, _STROMBOM_TOKEN_PARAMS["push_harder"])
        return {
            "intent_token": tok,
            "source": source,
            "logits": logits.tolist(),
            "params": deepcopy(p),
        }

    def plan(self, scene_tokens: dict, current_phase: str = "drive") -> dict:
        """Compute Strömbom parameter intent from Qwen + rule base + adapter delta."""
        base = self.base_llm(scene_tokens)
        adap = self._adapter_logits(scene_tokens)
        qwen_out = self.qwen.to_logits(scene_tokens, phase=current_phase)
        if qwen_out is not None:
            logits = 0.4 * base + 0.4 * qwen_out + 0.2 * adap
            source = "qwen+adapter"
        else:
            logits = base + adap
            source = "rule+adapter"
        return self._decode(logits, source)

    def push_failure(self, scene_tokens: dict, oracle_intent: dict) -> None:
        """Add one corrective example to the replay buffer."""
        self.buffer.push(scene_to_tensor(scene_tokens), intent_to_idx(oracle_intent, self.vocab))

    def update_adapter(self, lr: float = 5e-4, epochs: int = 3, batch_size: int = 16) -> list[float]:
        """Run one round of adapter training on the current replay buffer."""
        return train_adapter(self.adapter, self.buffer, lr=lr, epochs=epochs,
                             batch_size=batch_size, device=self.device)

    def save_snapshot(self, directory: str = "shepherd_codex/checkpoints") -> Path:
        """Serialise adapter weights to disk as a timestamped .pt file."""
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        path = p / f"strombom_adapter_{ts}.pt"
        torch.save(self.adapter.state_dict(), path)
        self.snapshots.append(path)
        return path

    def logged_update(
        self,
        failing_scene: dict,
        oracle_intent: dict,
        lr: float = 5e-4,
        epochs: int = 3,
        **kwargs,
    ) -> dict:
        """Push failure, retrain adapter, log pre/post intent change and return log dict."""
        pre = self.plan(failing_scene)
        snap = self.save_snapshot()
        self.push_failure(failing_scene, oracle_intent)
        losses = self.update_adapter(lr=lr, epochs=epochs)
        post = self.plan(failing_scene)
        log = {
            "timestamp": datetime.utcnow().isoformat(),
            "pre_plan": pre["intent_token"],
            "oracle_intent": oracle_intent["intent_token"],
            "loss_history": losses,
            "post_plan": post["intent_token"],
            "buffer_size": len(self.buffer),
            "snapshot": str(snap),
            "qwen_available": self.qwen.available,
        }
        self.update_logs.append(log)
        Path("shepherd_codex/checkpoints").mkdir(parents=True, exist_ok=True)
        with open("shepherd_codex/checkpoints/strombom_update_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
        return log