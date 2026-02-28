from __future__ import annotations
from collections import deque
import numpy as np

MAP_SIZE = np.array([20.0, 20.0])
MAP_DIAG = float(np.linalg.norm(MAP_SIZE))

HIST_LEN = 4
SCENE_DIM_BASE = 6
SCENE_DIM_HIST  = SCENE_DIM_BASE + HIST_LEN * 3 + 2


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
            mn, mx = obs
            for t in np.linspace(0.0, max_range, 64):
                p = origin + t * d
                if mn[0] <= p[0] <= mx[0] and mn[1] <= p[1] <= mx[1] and t < best:
                    best = t
                    out[i, 1] = 2.0
                    break
        out[i, 0] = best
    return out


def _pairwise_sheep_stats(sheep_pos: np.ndarray) -> tuple[float, float]:
    """Compute mean and minimum pairwise inter-sheep distance for spacing awareness."""
    n = len(sheep_pos)
    if n < 2:
        return 0.0, 0.0
    diffs = sheep_pos[:, None, :] - sheep_pos[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = dists[mask]
    return float(np.mean(vals)) / MAP_DIAG, float(np.min(vals)) / MAP_DIAG


def feature_extractor(scene_state: dict) -> dict:
    """Deterministic symbolic tokens for planning from scene state (6 base features)."""
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


class SceneHistory:
    """Rolling buffer of the last HIST_LEN scene snapshots for temporal context.

    Improvement 5: provides trajectory history (ACoM trend, escape trend) and
    pairwise sheep spacing so the adapter can distinguish transient vs persistent spread.
    """

    def __init__(self, hist_len: int = HIST_LEN) -> None:
        self._hist_len = hist_len
        self._acom_buf: deque[np.ndarray] = deque(maxlen=hist_len)
        self._esc_buf:  deque[float]      = deque(maxlen=hist_len)

    def update(self, tokens: dict) -> None:
        """Ingest one step's tokens into the history buffers."""
        self._acom_buf.append(np.asarray(tokens["ACoM"], dtype=float))
        self._esc_buf.append(float(tokens["escape_prob_est"]))

    def _padded(self) -> tuple[np.ndarray, np.ndarray]:
        """Return history arrays zero-padded to hist_len if buffer not yet full."""
        acom_arr = np.zeros((self._hist_len, 2), dtype=float)
        esc_arr  = np.zeros(self._hist_len, dtype=float)
        for i, (a, e) in enumerate(zip(self._acom_buf, self._esc_buf)):
            acom_arr[i] = a
            esc_arr[i]  = e
        return acom_arr, esc_arr

    def feature_extractor_hist(self, scene_state: dict) -> dict:
        """Return base tokens + history_acom, history_escape, pairwise_stats fields."""
        tokens = feature_extractor(scene_state)
        self.update(tokens)
        acom_hist, esc_hist = self._padded()
        sheep_pos = np.asarray(scene_state["sheep_pos"], dtype=float)
        mean_pair, min_pair = _pairwise_sheep_stats(sheep_pos)
        tokens["history_acom"]   = acom_hist.tolist()
        tokens["history_escape"] = esc_hist.tolist()
        tokens["mean_pair_dist"] = mean_pair
        tokens["min_pair_dist"]  = min_pair
        return tokens