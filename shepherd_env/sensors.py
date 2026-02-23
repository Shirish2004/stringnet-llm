"""Perception helpers: simple LiDAR and symbolic scene-token extraction."""

from __future__ import annotations

import numpy as np

MAP_SIZE = np.array([20.0, 20.0])
MAP_DIAG = float(np.linalg.norm(MAP_SIZE))


def lidar_scan(
    origin: np.ndarray,
    heading: float,
    sheep_positions: np.ndarray,
    obstacles: list[np.ndarray] | None = None,
    n_rays: int = 16,
    max_range: float = 8.0,
) -> np.ndarray:
    """Return [distance, class_id] per ray for circular sheep obstacles."""
    obstacles = obstacles or []
    rays = np.linspace(-np.pi / 2, np.pi / 2, n_rays) + heading
    out = np.full((n_rays, 2), [max_range, 0.0], dtype=float)
    radius = 0.2
    for i, ang in enumerate(rays):
        d = np.array([np.cos(ang), np.sin(ang)])
        best = max_range
        for sp in sheep_positions:
            rel = sp - origin
            t = float(np.dot(rel, d))
            if t <= 0:
                continue
            perp2 = float(np.dot(rel, rel) - t * t)
            if perp2 > radius * radius:
                continue
            hit = t - np.sqrt(max(radius * radius - perp2, 0.0))
            if 0 <= hit < best:
                best = hit
                out[i, 1] = 1.0
        for obs in obstacles:
            # axis-aligned rectangle obstacle: [[xmin,ymin],[xmax,ymax]]
            mn, mx = obs
            for t in np.linspace(0.0, max_range, 64):
                p = origin + t * d
                if mn[0] <= p[0] <= mx[0] and mn[1] <= p[1] <= mx[1] and t < best:
                    best = t
                    out[i, 1] = 2.0
                    break
        out[i, 0] = best
    return out


def feature_extractor(scene_state: dict) -> dict:
    """Deterministic symbolic tokens for planning from scene state."""
    sheep_pos = np.asarray(scene_state["sheep_pos"], dtype=float)
    sheep_vel = np.asarray(scene_state["sheep_vel"], dtype=float)
    goal = np.asarray(scene_state["goal"], dtype=float)
    acom = sheep_pos.mean(axis=0)
    spread = float(np.mean(np.linalg.norm(sheep_pos - acom, axis=1)))
    dists = np.linalg.norm(sheep_pos - goal, axis=1)
    largest_cluster_dist = float(np.max(dists))
    vel_var = float(np.var(np.linalg.norm(sheep_vel, axis=1)))
    escape_prob_est = float(np.clip(0.15 + 0.8 * vel_var + 0.1 * (spread / 3.0), 0.0, 1.0))
    obstacle_density_nearby = float(scene_state.get("obstacle_density_nearby", 0.0))
    return {
        "ACoM": (acom / MAP_SIZE).tolist(),
        "sheep_spread": spread / MAP_DIAG,
        "largest_cluster_dist": largest_cluster_dist / MAP_DIAG,
        "escape_prob_est": escape_prob_est,
        "obstacle_density_nearby": obstacle_density_nearby,
    }
