from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FailureDetector:
    """Failure detection + RL reward shaping (Improvement 3).

    Tracks containment, formation error, and escape risk; additionally emits
    a scalar reward signal suitable for REINFORCE / PPO fine-tuning of the adapter.
    """
    margin_thresh:  float = 0.0
    err_thresh:     float = 0.4
    t_fail:         int   = 8
    bad_steps:      int   = 0
    history:        list[dict] = field(default_factory=list)
    episode_return: float = 0.0
    _gamma:         float = 0.97

    def _containment_margin(
        self, sheep_pos: np.ndarray, center: np.ndarray, radius: float
    ) -> float:
        """Signed distance of the tightest sheep from the formation boundary."""
        d = np.linalg.norm(sheep_pos - center[None, :], axis=1)
        return float(np.min(radius - d))

    def _shape_reward(
        self,
        containment_margin: float,
        formation_error: float,
        escape_prob: float,
        escape_count: int,
        in_goal: int,
        n_sheep: int,
    ) -> float:
        """Dense reward combining containment, formation quality, and goal progress.

        Improvement 3 — reward used by REINFORCE and value-baseline training:
          +goal_fraction * 2.0 : progress toward herding sheep into goal
          +containment_margin * 0.5 : reward keeping formation tight
          -escape_prob * 1.0  : penalise high escape risk
          -escape_count * 0.3 : penalise individual escapes
          -formation_error * 0.2 : penalise dogs not in formation
        """
        goal_fraction = in_goal / max(n_sheep, 1)
        r = (
            goal_fraction * 2.0
            + containment_margin * 0.5
            - escape_prob * 1.0
            - escape_count * 0.3
            - formation_error * 0.2
        )
        return float(np.clip(r, -3.0, 3.0))

    def step(self, state: dict, xi_des: np.ndarray, scene_tokens: dict) -> dict:
        """Advance one step; returns metrics dict including shaped RL reward."""
        sheep_pos = np.asarray(state["sheep_pos"])
        dog_pos   = np.asarray(state["dog_pos"])
        n_sheep   = len(sheep_pos)
        center    = np.mean(xi_des, axis=0)
        radius    = float(np.mean(np.linalg.norm(xi_des - center, axis=1)))

        containment_margin = self._containment_margin(sheep_pos, center, radius)
        formation_error    = float(np.mean(np.linalg.norm(dog_pos - xi_des, axis=1)))
        escape_count       = int(np.sum(np.linalg.norm(sheep_pos - center[None, :], axis=1) > radius))
        vel_var            = float(np.var(np.linalg.norm(state["sheep_vel"], axis=1)))
        escape_prob_est    = float(np.clip(
            scene_tokens["escape_prob_est"] + 0.2 * vel_var - 0.1 * containment_margin, 0, 1
        ))

        goal_region = state.get("goal_region")
        in_goal = 0
        if goal_region is not None:
            gr = np.asarray(goal_region)
            in_mask = (
                (sheep_pos[:, 0] >= gr[0, 0]) & (sheep_pos[:, 0] <= gr[1, 0])
                & (sheep_pos[:, 1] >= gr[0, 1]) & (sheep_pos[:, 1] <= gr[1, 1])
            )
            in_goal = int(np.sum(in_mask))

        reward = self._shape_reward(
            containment_margin, formation_error, escape_prob_est, escape_count, in_goal, n_sheep
        )
        self.episode_return = self.episode_return * self._gamma + reward

        cond = (
            (containment_margin < self.margin_thresh)
            or (escape_prob_est > 0.3)
            or (formation_error > self.err_thresh)
        )
        self.bad_steps = self.bad_steps + 1 if cond else 0
        failure = self.bad_steps > self.t_fail

        out = {
            "containment_margin": containment_margin,
            "formation_error":    formation_error,
            "escape_count":       escape_count,
            "escape_prob_est":    escape_prob_est,
            "failure":            failure,
            "reward":             reward,
            "episode_return":     self.episode_return,
            "in_goal":            in_goal,
        }
        self.history.append(out)
        return out

    def reset_episode(self) -> None:
        """Reset episodic accumulators between episodes."""
        self.episode_return = 0.0
        self.bad_steps = 0

    def mean_reward(self, last_n: int = 50) -> float:
        """Moving average of reward over the last n steps for baseline estimation."""
        rewards = [h["reward"] for h in self.history[-last_n:]]
        return float(np.mean(rewards)) if rewards else 0.0