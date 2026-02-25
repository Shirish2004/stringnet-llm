from __future__ import annotations
import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    """Return unit vector of v; zero vector when norm is negligible."""
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else np.zeros_like(v)


def _herd_center(sheep_pos: np.ndarray) -> np.ndarray:
    """Rule 2 — Herd center C = (1/N) * sum(x_i) over all N sheep."""
    return sheep_pos.mean(axis=0)


def _furthest_sheep(sheep_pos: np.ndarray, C: np.ndarray) -> tuple[int, np.ndarray, float]:
    """Rule 2 — s_f = argmax_i ||x_i - C||; returns (index, position, distance)."""
    dists = np.linalg.norm(sheep_pos - C, axis=1)
    idx = int(np.argmax(dists))
    return idx, sheep_pos[idx], float(dists[idx])


def _collect_target(stray_pos: np.ndarray, C: np.ndarray, d_collect: float) -> np.ndarray:
    """Rule 3 — P_collect = x_f - v̂(C - x_f) * d_collect.

    Dog placed behind stray relative to herd center; repulsion steers stray back toward C.
    """
    v_hat = _unit(C - stray_pos)
    return stray_pos - v_hat * d_collect


def _drive_targets(
    C: np.ndarray,
    goal: np.ndarray,
    n_dogs: int,
    d_behind: float,
    formation_radius: float,
    flank_bias: float = 0.0,
) -> np.ndarray:
    """Rule 4 — g = v̂(Goal - C); P_drive = C - g * d_behind; dogs arc around P_drive.

    Semicircular arc (span = 0.9π) centred at P_drive, perpendicular to drive direction,
    satisfying the multi-dog extension of Rule 6.
    """
    g = _unit(goal - C)
    perp = np.array([-g[1], g[0]])
    p_drive = C - g * d_behind + perp * flank_bias
    phi_back = np.arctan2(-g[1], -g[0])
    span = np.pi * 0.90
    targets = np.zeros((n_dogs, 2), dtype=float)
    for i in range(n_dogs):
        theta = phi_back - span / 2.0 + (i + 0.5) * (span / max(n_dogs, 1))
        targets[i] = p_drive + formation_radius * np.array([np.cos(theta), np.sin(theta)])
    return targets


def _assign_collectors(dog_pos: np.ndarray, stray_pos: np.ndarray, n_collectors: int) -> list[int]:
    """Rule 6 — Assign the n_collectors closest dogs to retrieve the stray."""
    dists = np.linalg.norm(dog_pos - stray_pos, axis=1)
    return list(np.argsort(dists)[:n_collectors])


def compute_strombom_targets(
    sheep_pos: np.ndarray,
    dog_pos: np.ndarray,
    goal: np.ndarray,
    collect_radius: float = 2.5,
    d_collect: float = 1.5,
    d_behind: float = 3.0,
    formation_radius: float = 2.5,
    n_collectors: int = 1,
    flank_bias: float = 0.0,
    goal_radius: float = 1.5,
) -> tuple[np.ndarray, str]:
    """Exact Strömbom multi-dog rules from the paper; returns (per-dog targets, phase string).

    Rule 2: C = (1/N)Σx_i; s_f = argmax_i ||x_i - C||.
    Rule 3 (Collect): ||x_f - C|| > R_collect → P_collect = x_f - v̂(C-x_f)*d_collect.
    Rule 4 (Drive):   ||x_f - C|| <= R_collect → g = v̂(Goal-C); arc at P_drive = C - g*d_behind.
    Rule 6 (Multi-dog): nearest dog(s) go to P_collect; rest form semicircular arc.
    Rule 7 (Termination): if ||C - Goal|| <= R_goal return phase 'done'.
    """
    sheep_pos = np.asarray(sheep_pos, dtype=float)
    dog_pos   = np.asarray(dog_pos,   dtype=float)
    n_d = len(dog_pos)

    C = _herd_center(sheep_pos)

    if float(np.linalg.norm(C - goal)) <= goal_radius:
        return np.tile(C, (n_d, 1)), "done"

    _, stray_pos, stray_dist = _furthest_sheep(sheep_pos, C)

    if stray_dist > collect_radius:
        phase      = "collect"
        collectors = _assign_collectors(dog_pos, stray_pos, min(n_collectors, n_d))
        p_collect  = _collect_target(stray_pos, C, d_collect)

        g          = _unit(goal - C)
        perp       = np.array([-g[1], g[0]])
        guard_base = C - g * d_behind * 0.6
        non_cols   = [i for i in range(n_d) if i not in collectors]
        n_guards   = max(len(non_cols), 1)

        targets = np.zeros((n_d, 2), dtype=float)
        for ci in collectors:
            targets[ci] = p_collect
        for rank, gi in enumerate(non_cols):
            offset = (rank - (n_guards - 1) / 2.0) * (formation_radius * 0.75)
            targets[gi] = guard_base + perp * (offset + flank_bias * 0.5)
    else:
        phase   = "drive"
        targets = _drive_targets(C, goal, n_d, d_behind, formation_radius, flank_bias)

    return np.clip(targets, 0.2, 19.8), phase


def strombom_action(
    j: int,
    targets: np.ndarray,
    state: dict,
    config: dict,
    speed_scale: float = 1.0,
) -> np.ndarray:
    """PD locomotion law for dog j toward its Strömbom target; saturates to ubar_d * speed_scale.

    Rule 3: dogs must move faster than sheep — speed_scale enforces this margin.
    Rule 4: action is pure positional locomotion; no direct contact with sheep.
    """
    dog_pos = np.asarray(state["dog_pos"], dtype=float)
    dog_vel = np.asarray(state["dog_vel"], dtype=float)
    k_p = float(config.get("k1", 2.0)) * 2.2
    k_d = float(config.get("k2", 1.5))
    err = targets[j] - dog_pos[j]
    u   = k_p * err - k_d * dog_vel[j]
    ubar = float(config["ubar_d"]) * speed_scale
    n   = float(np.linalg.norm(u))
    return (u / n * ubar) if n > ubar else u


def check_termination(sheep_pos: np.ndarray, goal: np.ndarray, goal_radius: float = 1.5) -> bool:
    """Rule 7 — True when ||C - Goal|| <= R_goal; herd center has entered goal region."""
    return bool(float(np.linalg.norm(_herd_center(sheep_pos) - goal)) <= goal_radius)