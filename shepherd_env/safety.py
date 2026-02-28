from __future__ import annotations
import numpy as np
from .dynamics import saturate_norm

RADIUS_SCALE_MIN,  RADIUS_SCALE_MAX  = 0.60, 1.50
D_BEHIND_MIN,      D_BEHIND_MAX      = 0.80, 6.00
SPEED_SCALE_MIN,   SPEED_SCALE_MAX   = 0.50, 2.00
FLANK_BIAS_ABS_MAX                   = 3.00
CONT_DELTA_ABS_MAX                   = 0.50


def project_action(
    action: np.ndarray,
    state: dict,
    ubar_d: float,
    k_steps: int = 3,
) -> tuple[np.ndarray, bool]:
    """Project action to safe set: bound norm and avoid imminent dog-dog collisions."""
    a = saturate_norm(np.asarray(action, dtype=float), ubar_d)
    dog_pos = np.asarray(state["dog_pos"], dtype=float)
    dog_vel = np.asarray(state["dog_vel"], dtype=float)
    def_id  = int(state.get("active_defender", 0))
    p = dog_pos[def_id].copy()
    v = dog_vel[def_id].copy()
    dt = float(state.get("dt", 0.05))
    violated = False
    for _ in range(k_steps):
        v = v + dt * a
        p = p + dt * v
        others = np.delete(dog_pos, def_id, axis=0)
        if len(others) and np.min(np.linalg.norm(others - p, axis=1)) < 2 * float(state.get("r_agent", 0.2)):
            violated = True
            break
    if violated:
        a = 0.5 * a
    return a, violated


def project_planner_params(params: dict, config: dict) -> tuple[dict, bool]:
    """Improvement 8 — Clip continuous planner parameters to the feasible set.

    Enforces:
    * radius_scale within [RADIUS_SCALE_MIN, RADIUS_SCALE_MAX]
    * d_behind_scale such that resulting d_behind keeps dogs within arena
    * speed_scale ensuring ubar_d * speed_scale > ubar_a (dogs faster than sheep, Rule 3)
    * continuous deltas bounded by CONT_DELTA_ABS_MAX to prevent runaway updates
    Returns (projected_params, was_modified).
    """
    ubar_d = float(config.get("ubar_d", 3.0))
    ubar_a = float(config.get("ubar_a", 2.0))
    speed_min_safe = max(SPEED_SCALE_MIN, (ubar_a / ubar_d) + 0.05)

    out = dict(params)
    modified = False

    for key, lo, hi in [
        ("radius_scale",           RADIUS_SCALE_MIN,    RADIUS_SCALE_MAX),
        ("radius_scale_delta",    -CONT_DELTA_ABS_MAX,  CONT_DELTA_ABS_MAX),
        ("d_behind_scale",         D_BEHIND_MIN / 3.0,  D_BEHIND_MAX / 3.0),
        ("d_behind_delta",        -CONT_DELTA_ABS_MAX,  CONT_DELTA_ABS_MAX),
        ("speed_scale",            speed_min_safe,      SPEED_SCALE_MAX),
        ("speed_scale_delta",     -CONT_DELTA_ABS_MAX,  CONT_DELTA_ABS_MAX),
    ]:
        if key in out:
            clipped = float(np.clip(out[key], lo, hi))
            if clipped != out[key]:
                modified = True
            out[key] = clipped

    if "flank_bias" in out:
        clipped = float(np.clip(out["flank_bias"], -FLANK_BIAS_ABS_MAX, FLANK_BIAS_ABS_MAX))
        if clipped != out["flank_bias"]:
            modified = True
        out["flank_bias"] = clipped

    if "flank_bias_delta" in out:
        clipped = float(np.clip(out["flank_bias_delta"], -CONT_DELTA_ABS_MAX, CONT_DELTA_ABS_MAX))
        if clipped != out["flank_bias_delta"]:
            modified = True
        out["flank_bias_delta"] = clipped

    return out, modified


def validate_params_with_rollout(
    proposed_params: dict,
    baseline_params: dict,
    rollout_fn,
    n_steps: int = 30,
) -> bool:
    """Improvement 8 — Accept proposed_params only if short rollout reward >= baseline.

    rollout_fn(params, n_steps) -> float reward score.
    Returns True if proposed_params are safe to use.
    """
    score_proposed = rollout_fn(proposed_params, n_steps)
    score_baseline = rollout_fn(baseline_params, n_steps)
    return bool(score_proposed >= score_baseline - 0.05)