"""Failure detection metrics for StringNet herding episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class FailureDetector:
    margin_thresh: float = 0.0
    err_thresh: float = 0.4
    t_fail: int = 8
    bad_steps: int = 0
    history: list[dict] = field(default_factory=list)

    def _containment_margin(self, sheep_pos: np.ndarray, center: np.ndarray, radius: float) -> float:
        d = np.linalg.norm(sheep_pos - center[None, :], axis=1)
        return float(np.min(radius - d))

    def step(self, state: dict, xi_des: np.ndarray, scene_tokens: dict) -> dict:
        sheep_pos = np.asarray(state["sheep_pos"])
        dog_pos = np.asarray(state["dog_pos"])
        center = np.mean(xi_des, axis=0)
        radius = float(np.mean(np.linalg.norm(xi_des - center, axis=1)))
        containment_margin = self._containment_margin(sheep_pos, center, radius)
        formation_error = float(np.mean(np.linalg.norm(dog_pos - xi_des, axis=1)))
        escape_count = int(np.sum(np.linalg.norm(sheep_pos - center[None, :], axis=1) > radius))
        vel_var = float(np.var(np.linalg.norm(state["sheep_vel"], axis=1)))
        escape_prob_est = float(np.clip(scene_tokens["escape_prob_est"] + 0.2 * vel_var - 0.1 * containment_margin, 0, 1))
        cond = (containment_margin < self.margin_thresh) or (escape_prob_est > 0.3) or (formation_error > self.err_thresh)
        self.bad_steps = self.bad_steps + 1 if cond else 0
        failure = self.bad_steps > self.t_fail
        out = {
            "containment_margin": containment_margin,
            "formation_error": formation_error,
            "escape_count": escape_count,
            "escape_prob_est": escape_prob_est,
            "failure": failure,
        }
        self.history.append(out)
        return out
