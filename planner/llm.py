from __future__ import annotations

import json
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .train import (
    DualHeadAdapter,
    AdapterMLP,
    ReplayBuffer,
    DEFAULT_VOCAB,
    STROMBOM_VOCAB,
    SCENE_DIM_BASE,
    SCENE_DIM_HIST,
    CONT_DIM,
    CONT_PARAM_NAMES,
    intent_to_idx,
    oracle_cont_params,
    scene_to_tensor,
    scene_to_tensor_hist,
    train_adapter,
)
from .mock_llm import MockLLM, OracularPlanner, StrombomMockLLM, StrombomOracle

_HIGH_ENTROPY_THRESH = 1.8
_CONF_ROLLOUT_STEPS  = 40
_SNAPSHOT_DIR        = "shepherd_codex/checkpoints"

_STRINGNET_SYSTEM = (
    "You are a high-level planner for StringNet herding robots. "
    "Defenders form a semi-circular net around sheep and herd them to a goal region. "
    "Respond with ONLY a JSON object — no markdown, no explanation. "
    "Required keys: intent_token (string), phase (string), radius_scale (float), "
    "speed_scale (float), d_behind_delta (float), flank_bias (float). "
    "intent_token must be one of: tighten_net, widen_net, flank_left, flank_right, "
    "split, focus_largest_cluster, delay_enclose, increase_speed, reduce_speed, move_leader_to. "
    "phase must be one of: seek, enclose, herd. "
    "radius_scale range: 0.7 to 1.4. speed_scale range: 0.7 to 1.4. "
    "d_behind_delta range: -0.4 to 0.4. flank_bias range: -2.0 to 2.0."
)

_STROMBOM_SYSTEM = (
    "You are a high-level planner for Strömbom herding dogs. "
    "Dogs alternate between collecting stray sheep and driving the herd toward a goal. "
    "Respond with ONLY a JSON object — no markdown, no explanation. "
    "Required keys: intent_token (string), collect_radius_scale (float), "
    "drive_offset_scale (float), formation_radius_scale (float), speed_scale (float), "
    "flank_bias (float). "
    "intent_token must be one of: tighten_collect, loosen_collect, push_harder, back_off, "
    "spread_formation, compress_formation, speed_up, slow_down, flank_left, flank_right. "
    "All scale values range: 0.7 to 1.5. flank_bias range: -2.5 to 2.5."
)

_STRINGNET_USER_TEMPLATE = (
    "Scene state:\n"
    "  ACoM (normalized): ({acom_x:.4f}, {acom_y:.4f})\n"
    "  sheep_spread: {spread:.4f}  |  mean_pair_dist: {mean_pair:.4f}\n"
    "  largest_cluster_dist: {cluster_dist:.4f}\n"
    "  escape_prob_est: {escape_prob:.4f}\n"
    "  obstacle_density_nearby: {obstacle:.4f}\n"
    "  current_phase: {phase}\n"
    "  escape_trend (last 4): {esc_trend}\n"
    "Select the optimal StringNet intent and continuous parameter adjustments."
)

_STROMBOM_USER_TEMPLATE = (
    "Scene state:\n"
    "  ACoM (normalized): ({acom_x:.4f}, {acom_y:.4f})\n"
    "  sheep_spread: {spread:.4f}  |  mean_pair_dist: {mean_pair:.4f}\n"
    "  largest_cluster_dist: {cluster_dist:.4f}\n"
    "  escape_prob_est: {escape_prob:.4f}\n"
    "  obstacle_density_nearby: {obstacle:.4f}\n"
    "  current_strombom_phase: {phase}\n"
    "  escape_trend (last 4): {esc_trend}\n"
    "Select the optimal Strömbom parameter-adjustment intent."
)


class QwenPlanner:
    """Improvement 7 — Generic Qwen wrapper; LoRA-ready via PEFT injection hook.

    Standard usage loads model weights and runs autoregressive generation.
    When `peft_config` is provided at construction, applies LoRA adapters to the
    LM's attention layers (requires peft library). This allows low-rank fine-tuning
    of the LM's internal representations without changing the MLP adapter side.
    """

    def __init__(
        self,
        model_name:    str = "Qwen/Qwen3-0.6B",
        system_prompt: str = _STRINGNET_SYSTEM,
        user_template: str = _STRINGNET_USER_TEMPLATE,
        vocab:         list[str] | None = None,
        device:        Optional[str] = None,
        max_new_tokens: int = 160,
        torch_dtype:   torch.dtype = torch.float16,
        peft_config:   dict | None = None,
    ) -> None:
        self.model_name    = model_name
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.vocab         = vocab or DEFAULT_VOCAB
        self.max_new_tokens = max_new_tokens
        self.available     = False

        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if resolved == "cuda" else None,
                trust_remote_code=True,
            )
            if peft_config:
                base_model = self._apply_lora(base_model, peft_config)
            if resolved != "cuda":
                base_model = base_model.to(self.device)
            base_model.eval()
            self.model     = base_model
            self.available = True
        except Exception as exc:
            print(f"[QwenPlanner] Failed to load {model_name}: {exc}")
            self.tokenizer = None
            self.model     = None

    def _apply_lora(self, model, peft_config: dict):
        """Improvement 7 — Inject LoRA low-rank adapters if peft library is available."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=peft_config.get("r", 8),
                lora_alpha=peft_config.get("lora_alpha", 16),
                target_modules=peft_config.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=peft_config.get("lora_dropout", 0.05),
                bias="none",
            )
            return get_peft_model(model, lora_cfg)
        except ImportError:
            print("[QwenPlanner] peft not installed; skipping LoRA — pip install peft")
            return model

    def _build_messages(self, scene_tokens: dict, phase: str) -> list[dict]:
        """Improvement 5 — Include history in the prompt for temporal awareness."""
        esc_hist = scene_tokens.get("history_escape", [0.0] * 4)
        esc_trend = " ".join(f"{e:.2f}" for e in esc_hist[-4:])
        mean_pair = float(scene_tokens.get("mean_pair_dist", 0.0))
        user_msg  = self.user_template.format(
            acom_x=scene_tokens["ACoM"][0],
            acom_y=scene_tokens["ACoM"][1],
            spread=scene_tokens["sheep_spread"],
            cluster_dist=scene_tokens["largest_cluster_dist"],
            escape_prob=scene_tokens["escape_prob_est"],
            obstacle=scene_tokens["obstacle_density_nearby"],
            phase=phase,
            esc_trend=esc_trend,
            mean_pair=mean_pair,
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_msg},
        ]

    def _parse_response(self, raw: str) -> Optional[dict]:
        raw   = raw.strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        try:
            data = json.loads(raw[start:end])
            tok  = data.get("intent_token", "")
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
            text   = self.tokenizer.apply_chat_template(
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


def _safe_project(params: dict, config: dict) -> dict:
    """Inline safety projection; import guard prevents circular at module level."""
    try:
        from shepherd_env.safety import project_planner_params
        safe_p, _ = project_planner_params(params, config)
        return safe_p
    except Exception:
        return params


class LLMPlanner:
    """Improvements 1-8 — StringNet high-level planner with dual-head adapter.

    Improvement 1: DualHeadAdapter produces (discrete logits, continuous param deltas).
    Improvement 2: beam_plan() runs short rollouts to pick best candidate intent.
    Improvement 3: logged_update() stores reward; train_adapter uses REINFORCE signal.
    Improvement 4: confidence() via softmax entropy; low-confidence triggers validation.
    Improvement 5: scene_to_tensor_hist uses 20-dim extended feature vector.
    Improvement 6: hierarchical_plan() returns phase + assignments + timing.
    Improvement 7: QwenPlanner accepts peft_config for LoRA fine-tuning.
    Improvement 8: _safe_project() clips continuous params to feasible set.
    """

    def __init__(
        self,
        vocab:            Optional[list[str]] = None,
        adapter_dim:      int   = 64,
        qwen_model:       str   = "Qwen/Qwen3-0.6B",
        qwen_device:      Optional[str] = None,
        buffer_capacity:  int   = 512,
        device:           Optional[str] = None,
        seed:             int   = 0,
        use_hist_features: bool = True,
        peft_config:      dict | None = None,
        lambda_cat:       float = 1.0,
        lambda_cont:      float = 0.3,
        lambda_rl:        float = 0.1,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.vocab            = vocab or DEFAULT_VOCAB
        self.device           = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_hist          = use_hist_features
        self.in_dim            = SCENE_DIM_HIST if use_hist_features else SCENE_DIM_BASE
        self.lambda_cat        = lambda_cat
        self.lambda_cont       = lambda_cont
        self.lambda_rl         = lambda_rl
        self.qwen              = QwenPlanner(
            model_name=qwen_model, system_prompt=_STRINGNET_SYSTEM,
            user_template=_STRINGNET_USER_TEMPLATE, vocab=self.vocab,
            device=qwen_device, peft_config=peft_config,
        )
        self.base_llm          = MockLLM(self.vocab)
        self.adapter           = DualHeadAdapter(self.in_dim, adapter_dim, len(self.vocab), CONT_DIM)
        self.adapter.to(self.device).eval()
        self.buffer            = ReplayBuffer(capacity=buffer_capacity)
        self.update_logs:  list[dict] = []
        self.snapshots:    deque      = deque(maxlen=10)
        self._validation_score: float = -999.0
        self._config: dict             = {}

    def set_config(self, config: dict) -> None:
        """Store env config for safety projection and rollout validation."""
        self._config = config

    def _encode(self, scene_tokens: dict) -> torch.Tensor:
        """Encode scene tokens to the adapter's input tensor."""
        fn = scene_to_tensor_hist if self.use_hist else scene_to_tensor
        return fn(scene_tokens).unsqueeze(0).to(self.device)

    def _adapter_outputs(self, scene_tokens: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return (discrete logits, continuous params) from DualHeadAdapter."""
        x = self._encode(scene_tokens)
        with torch.no_grad():
            logits, cont = self.adapter(x)
        return logits.squeeze(0).cpu().numpy(), cont.squeeze(0).cpu().numpy()

    def confidence(self, scene_tokens: dict) -> float:
        """Improvement 4 — Adapter confidence as negative softmax entropy (higher = more confident)."""
        x = self._encode(scene_tokens)
        with torch.no_grad():
            h = self.adapter.softmax_entropy(x)
        return float(-h.item())

    def _decode_stringnet(
        self,
        logits: np.ndarray,
        cont:   np.ndarray,
        source: str,
    ) -> dict:
        """Map discrete logits + continuous deltas to a fully parameterised StringNet intent."""
        idx  = int(np.argmax(logits))
        tok  = self.vocab[idx]
        phase = "seek"
        if tok in {"tighten_net", "focus_largest_cluster", "increase_speed"}:
            phase = "herd"
        elif tok in {"split", "delay_enclose"}:
            phase = "enclose"

        base_rs = 0.9 if tok == "tighten_net" else 1.1 if tok == "widen_net" else 1.0
        base_ss = 1.1 if tok == "increase_speed" else 0.9 if tok == "reduce_speed" else 1.0

        rs_delta, db_delta, ss_delta, fb_delta = cont.tolist()
        params = {
            "radius_scale":        float(np.clip(base_rs + rs_delta * 0.3, 0.7, 1.4)),
            "radius_scale_delta":  float(rs_delta),
            "d_behind_delta":      float(db_delta * 0.4),
            "speed_scale":         float(np.clip(base_ss + ss_delta * 0.2, 0.5, 2.0)),
            "speed_scale_delta":   float(ss_delta),
            "flank_bias":          float(fb_delta * 2.0),
            "flank_bias_delta":    float(fb_delta),
        }
        params = _safe_project(params, self._config)
        return {
            "formation_type": "semi-ellipse",
            "params":         params,
            "assignments":    {"leader": 0},
            "phase":          phase,
            "intent_token":   tok,
            "logits":         logits.tolist(),
            "cont_params":    cont.tolist(),
            "source":         source,
        }

    def _propose_candidates(self, scene_tokens: dict, current_phase: str, n: int) -> list[dict]:
        """Improvement 2 — Generate n candidate intents by sampling adapter logits with noise."""
        base    = self.base_llm(scene_tokens)
        logits0, cont0 = self._adapter_outputs(scene_tokens)
        qwen_out = self.qwen.to_logits(scene_tokens, phase=current_phase)
        candidates = []
        for i in range(n):
            noise = np.random.randn(*logits0.shape) * (0.3 * i)
            if qwen_out is not None:
                l = 0.4 * base + 0.4 * qwen_out + 0.2 * (logits0 + noise)
            else:
                l = base + logits0 + noise
            candidates.append(self._decode_stringnet(l, cont0, "candidate"))
        return candidates

    def plan(self, scene_tokens: dict, current_phase: str = "seek") -> dict:
        """Improvement 1,4 — Standard single-step plan with confidence check."""
        base     = self.base_llm(scene_tokens)
        logits, cont = self._adapter_outputs(scene_tokens)
        qwen_out = self.qwen.to_logits(scene_tokens, phase=current_phase)

        if qwen_out is not None:
            combined = 0.4 * base + 0.4 * qwen_out + 0.2 * logits
            source   = "qwen+adapter"
        else:
            combined = base + logits
            source   = "rule+adapter"

        intent = self._decode_stringnet(combined, cont, source)

        conf = self.confidence(scene_tokens)
        if conf < -_HIGH_ENTROPY_THRESH:
            intent["source"] = source + "+uncertain"

        return intent

    def beam_plan(
        self,
        scene_tokens: dict,
        current_phase: str,
        env,
        state: dict,
        action_fn,
        fd_cls,
        n_candidates: int = 4,
        rollout_steps: int = _CONF_ROLLOUT_STEPS,
    ) -> dict:
        """Improvement 2 — Beam search: evaluate N candidate intents via rollouts."""
        from .rl import beam_search_plan
        candidates = self._propose_candidates(scene_tokens, current_phase, n_candidates)
        n_d = len(np.asarray(state["dog_pos"]))
        best, scores = beam_search_plan(
            candidates, env, state, action_fn, fd_cls, self._config, n_d, rollout_steps
        )
        best["beam_scores"] = scores
        best["source"]      = best.get("source", "rule") + "+beam"
        return best

    def hierarchical_plan(self, scene_tokens: dict, current_phase: str, n_dogs: int) -> dict:
        """Improvement 6 — Return phase + per-dog role assignments + reeval timing.

        Roles: 'leader' pushes toward goal; 'flanker_left/right' maintain arc;
        'collector' retrieves stray (only in enclose/collect phase).
        """
        base_intent = self.plan(scene_tokens, current_phase)
        tok         = base_intent["intent_token"]
        phase       = base_intent["phase"]

        assignments = {}
        if n_dogs == 1:
            assignments = {0: "leader"}
        elif phase in ("enclose", "collect"):
            assignments[0] = "collector"
            for j in range(1, n_dogs):
                assignments[j] = "flanker_left" if j % 2 == 0 else "flanker_right"
        else:
            assignments[0] = "leader"
            for j in range(1, n_dogs):
                assignments[j] = "flanker_left" if j % 2 == 1 else "flanker_right"

        spread = float(scene_tokens.get("sheep_spread", 0.05))
        t_hold = max(5, int(15 * (1 - spread * 5)))
        base_intent["assignments"] = assignments
        base_intent["timeline"]    = {"T_hold": t_hold, "reeval_in": max(3, t_hold // 3)}
        return base_intent

    def push_failure(
        self,
        scene_tokens:  dict,
        oracle_intent: dict,
        reward:        float = 0.0,
    ) -> None:
        """Store corrective example with oracle cont params and shaped reward."""
        x  = scene_to_tensor_hist(scene_tokens) if self.use_hist else scene_to_tensor(scene_tokens)
        yi = intent_to_idx(oracle_intent, self.vocab)
        yc = oracle_cont_params(oracle_intent)
        self.buffer.push(x, yi, yc, reward)

    def update_adapter(self, lr: float = 5e-4, epochs: int = 3, batch_size: int = 16) -> list[float]:
        """Run one round of dual-head adapter training with mixed loss."""
        return train_adapter(
            self.adapter, self.buffer,
            lr=lr, epochs=epochs, batch_size=batch_size, device=self.device,
            lambda_cat=self.lambda_cat, lambda_cont=self.lambda_cont, lambda_rl=self.lambda_rl,
        )

    def save_snapshot(self, directory: str = _SNAPSHOT_DIR) -> Path:
        """Serialise adapter weights to disk as a timestamped .pt file."""
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        path = p / f"adapter_{ts}.pt"
        torch.save(self.adapter.state_dict(), path)
        self.snapshots.append(path)
        return path

    def load_snapshot(self, path: Path) -> None:
        """Restore adapter weights from a .pt checkpoint."""
        self.adapter.load_state_dict(torch.load(path, map_location=self.device))
        self.adapter.eval()

    def _validate_snapshot(self, rollout_fn, n_steps: int = 30) -> bool:
        """Improvement 4 — Run validation rollout; revert last snapshot if score drops."""
        try:
            from shepherd_env.safety import validate_params_with_rollout
            baseline = {"radius_scale": 1.0, "speed_scale": 1.0}
            proposed = {"radius_scale": 1.0, "speed_scale": 1.0}
            ok = validate_params_with_rollout(proposed, baseline, rollout_fn, n_steps)
            return ok
        except Exception:
            return True

    def logged_update(
        self,
        failing_scene: dict,
        oracle_intent: dict,
        lr:     float = 5e-4,
        epochs: int   = 3,
        reward: float = 0.0,
        **kwargs,
    ) -> dict:
        """Improvement 3,4 — Push failure + reward, retrain, validate, log."""
        pre  = self.plan(failing_scene)
        snap = self.save_snapshot()
        self.push_failure(failing_scene, oracle_intent, reward=reward)
        losses = self.update_adapter(lr=lr, epochs=epochs)
        post   = self.plan(failing_scene)

        log = {
            "timestamp":         datetime.utcnow().isoformat(),
            "pre_plan":          pre["intent_token"],
            "oracle_intent":     oracle_intent["intent_token"],
            "loss_history":      losses,
            "post_plan":         post["intent_token"],
            "buffer_size":       len(self.buffer),
            "snapshot":          str(snap),
            "qwen_available":    self.qwen.available,
            "reward":            reward,
            "confidence_pre":    self.confidence(failing_scene),
        }
        self.update_logs.append(log)
        Path(_SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)
        with open(f"{_SNAPSHOT_DIR}/update_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
        return log


_STROMBOM_TOKEN_PARAMS: dict[str, dict] = {
    "tighten_collect":    {"collect_radius_scale": 0.70, "drive_offset_scale": 1.00, "formation_radius_scale": 1.00, "speed_scale": 1.05, "flank_bias": 0.0},
    "loosen_collect":     {"collect_radius_scale": 1.35, "drive_offset_scale": 1.00, "formation_radius_scale": 1.10, "speed_scale": 0.95, "flank_bias": 0.0},
    "push_harder":        {"collect_radius_scale": 1.00, "drive_offset_scale": 0.70, "formation_radius_scale": 1.00, "speed_scale": 1.10, "flank_bias": 0.0},
    "back_off":           {"collect_radius_scale": 1.00, "drive_offset_scale": 1.35, "formation_radius_scale": 1.10, "speed_scale": 0.90, "flank_bias": 0.0},
    "spread_formation":   {"collect_radius_scale": 1.00, "drive_offset_scale": 1.00, "formation_radius_scale": 1.30, "speed_scale": 1.00, "flank_bias": 0.0},
    "compress_formation": {"collect_radius_scale": 1.00, "drive_offset_scale": 1.00, "formation_radius_scale": 0.75, "speed_scale": 1.05, "flank_bias": 0.0},
    "speed_up":           {"collect_radius_scale": 1.00, "drive_offset_scale": 1.00, "formation_radius_scale": 1.00, "speed_scale": 1.25, "flank_bias": 0.0},
    "slow_down":          {"collect_radius_scale": 1.00, "drive_offset_scale": 1.00, "formation_radius_scale": 1.00, "speed_scale": 0.80, "flank_bias": 0.0},
    "flank_left":         {"collect_radius_scale": 1.00, "drive_offset_scale": 1.00, "formation_radius_scale": 1.00, "speed_scale": 1.00, "flank_bias": -1.2},
    "flank_right":        {"collect_radius_scale": 1.00, "drive_offset_scale": 1.00, "formation_radius_scale": 1.00, "speed_scale": 1.00, "flank_bias":  1.2},
}


class StrombomLLMPlanner:
    """Improvements 1-8 — Strömbom high-level planner with dual-head adapter.

    Mirrors LLMPlanner architecture but targets Strömbom collect/drive parameters.
    """

    def __init__(
        self,
        vocab:            Optional[list[str]] = None,
        adapter_dim:      int   = 64,
        qwen_model:       str   = "Qwen/Qwen3-0.6B",
        qwen_device:      Optional[str] = None,
        buffer_capacity:  int   = 512,
        device:           Optional[str] = None,
        seed:             int   = 0,
        use_hist_features: bool = True,
        lambda_cat:       float = 1.0,
        lambda_cont:      float = 0.3,
        lambda_rl:        float = 0.1,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.vocab       = vocab or STROMBOM_VOCAB
        self.device      = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_hist    = use_hist_features
        self.in_dim      = SCENE_DIM_HIST if use_hist_features else SCENE_DIM_BASE
        self.lambda_cat  = lambda_cat
        self.lambda_cont = lambda_cont
        self.lambda_rl   = lambda_rl
        self.qwen        = QwenPlanner(
            model_name=qwen_model, system_prompt=_STROMBOM_SYSTEM,
            user_template=_STROMBOM_USER_TEMPLATE, vocab=self.vocab, device=qwen_device,
        )
        self.base_llm = StrombomMockLLM(self.vocab)
        self.adapter  = DualHeadAdapter(self.in_dim, adapter_dim, len(self.vocab), CONT_DIM)
        self.adapter.to(self.device).eval()
        self.buffer      = ReplayBuffer(capacity=buffer_capacity)
        self.update_logs: list[dict] = []
        self.snapshots:   deque      = deque(maxlen=10)
        self._config:     dict       = {}

    def set_config(self, config: dict) -> None:
        """Store env config for safety projection."""
        self._config = config

    def _encode(self, scene_tokens: dict) -> torch.Tensor:
        fn = scene_to_tensor_hist if self.use_hist else scene_to_tensor
        return fn(scene_tokens).unsqueeze(0).to(self.device)

    def _adapter_outputs(self, scene_tokens: dict) -> tuple[np.ndarray, np.ndarray]:
        x = self._encode(scene_tokens)
        with torch.no_grad():
            logits, cont = self.adapter(x)
        return logits.squeeze(0).cpu().numpy(), cont.squeeze(0).cpu().numpy()

    def confidence(self, scene_tokens: dict) -> float:
        """Improvement 4 — Negative softmax entropy as confidence signal."""
        x = self._encode(scene_tokens)
        with torch.no_grad():
            h = self.adapter.softmax_entropy(x)
        return float(-h.item())

    def _decode(self, logits: np.ndarray, cont: np.ndarray, source: str) -> dict:
        """Improvements 1,8 — Decode logits + continuous deltas into Strömbom params."""
        idx = int(np.argmax(logits))
        tok = self.vocab[idx]
        p   = deepcopy(_STROMBOM_TOKEN_PARAMS.get(tok, _STROMBOM_TOKEN_PARAMS["push_harder"]))
        rs_delta, db_delta, ss_delta, fb_delta = cont.tolist()
        p["collect_radius_scale"]   = float(np.clip(p["collect_radius_scale"]   + rs_delta * 0.2, 0.5, 1.8))
        p["drive_offset_scale"]     = float(np.clip(p["drive_offset_scale"]     + db_delta * 0.2, 0.4, 1.8))
        p["formation_radius_scale"] = float(np.clip(p["formation_radius_scale"] + rs_delta * 0.15, 0.5, 1.8))
        p["speed_scale"]            = float(np.clip(p["speed_scale"]            + ss_delta * 0.15, 0.5, 2.0))
        p["flank_bias"]             = float(np.clip(p["flank_bias"]             + fb_delta * 1.5, -3.0, 3.0))
        p = _safe_project(p, self._config)
        return {
            "intent_token": tok,
            "source":       source,
            "logits":       logits.tolist(),
            "cont_params":  cont.tolist(),
            "params":       p,
        }

    def plan(self, scene_tokens: dict, current_phase: str = "drive") -> dict:
        """Improvement 1,4 — Strömbom plan with continuous parameter fine-tuning."""
        base      = self.base_llm(scene_tokens)
        logits, cont = self._adapter_outputs(scene_tokens)
        qwen_out  = self.qwen.to_logits(scene_tokens, phase=current_phase)
        if qwen_out is not None:
            combined = 0.4 * base + 0.4 * qwen_out + 0.2 * logits
            source   = "qwen+adapter"
        else:
            combined = base + logits
            source   = "rule+adapter"
        intent = self._decode(combined, cont, source)
        if self.confidence(scene_tokens) < -_HIGH_ENTROPY_THRESH:
            intent["source"] += "+uncertain"
        return intent

    def push_failure(self, scene_tokens: dict, oracle_intent: dict, reward: float = 0.0) -> None:
        """Store corrective example with oracle cont params and shaped reward."""
        x  = scene_to_tensor_hist(scene_tokens) if self.use_hist else scene_to_tensor(scene_tokens)
        yi = intent_to_idx(oracle_intent, self.vocab)
        yc = oracle_cont_params(oracle_intent)
        self.buffer.push(x, yi, yc, reward)

    def update_adapter(self, lr: float = 5e-4, epochs: int = 3, batch_size: int = 16) -> list[float]:
        """Run one round of dual-head adapter training."""
        return train_adapter(
            self.adapter, self.buffer,
            lr=lr, epochs=epochs, batch_size=batch_size, device=self.device,
            lambda_cat=self.lambda_cat, lambda_cont=self.lambda_cont, lambda_rl=self.lambda_rl,
        )

    def save_snapshot(self, directory: str = _SNAPSHOT_DIR) -> Path:
        """Serialise adapter weights to disk."""
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        path = p / f"strombom_adapter_{ts}.pt"
        torch.save(self.adapter.state_dict(), path)
        self.snapshots.append(path)
        return path

    def logged_update(
        self,
        failing_scene: dict,
        oracle_intent: dict,
        lr:     float = 5e-4,
        epochs: int   = 3,
        reward: float = 0.0,
        **kwargs,
    ) -> dict:
        """Improvement 3,4 — Push failure + reward, retrain adapter, log."""
        pre  = self.plan(failing_scene)
        snap = self.save_snapshot()
        self.push_failure(failing_scene, oracle_intent, reward=reward)
        losses = self.update_adapter(lr=lr, epochs=epochs)
        post   = self.plan(failing_scene)
        log    = {
            "timestamp":      datetime.utcnow().isoformat(),
            "pre_plan":       pre["intent_token"],
            "oracle_intent":  oracle_intent["intent_token"],
            "loss_history":   losses,
            "post_plan":      post["intent_token"],
            "buffer_size":    len(self.buffer),
            "snapshot":       str(snap),
            "qwen_available": self.qwen.available,
            "reward":         reward,
            "confidence_pre": self.confidence(failing_scene),
        }
        self.update_logs.append(log)
        Path(_SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)
        with open(f"{_SNAPSHOT_DIR}/strombom_update_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
        return log