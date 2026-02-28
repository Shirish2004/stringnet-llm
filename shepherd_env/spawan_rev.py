"""Spawn utilities for sheep/dogs with left-quadrant constraints."""

from __future__ import annotations

import numpy as np

LEFT_X = (0.0, 8.0)
LEFT_Y = (5.0, 15.0)


def sample_poisson_disc_in_circle(
    n: int,
    center: np.ndarray,
    radius: float,
    d_min: float,
    rng: np.random.Generator,
    max_tries: int = 20_000,
) -> np.ndarray:
    """Rejection-sample points in a circle with min-separation ``d_min``."""
    pts: list[np.ndarray] = []
    tries = 0
    while len(pts) < n and tries < max_tries:
        tries += 1
        r = radius * np.sqrt(rng.uniform())
        th = rng.uniform(0, 2 * np.pi)
        p = center + np.array([r * np.cos(th), r * np.sin(th)])
        if not (LEFT_X[0] <= p[0] <= LEFT_X[1] and LEFT_Y[0] <= p[1] <= LEFT_Y[1]):
            continue
        if all(np.linalg.norm(p - q) >= d_min for q in pts):
            pts.append(p)
    if len(pts) != n:
        raise RuntimeError("Could not sample requested number of sheep positions.")
    return np.vstack(pts)


def spawn_sheep(
    n_sheep: int,
    rng: np.random.Generator,
    rho_ac_range: tuple[float, float] = (0.6, 1.2),
    d_min: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Spawn sheep clustered in the left quadrant around ACoM center ``r_ac``."""
    center = np.array([rng.uniform(2.0, 6.0), rng.uniform(7.0, 13.0)], dtype=float)
    radius = rng.uniform(*rho_ac_range)
    pos = sample_poisson_disc_in_circle(n_sheep, center, radius, d_min, rng)
    return pos, center, radius


def spawn_dogs(n_dogs: int, rng: np.random.Generator) -> np.ndarray:
    """Spawn dogs on left periphery near top/bottom edges."""
    nodes = [
        np.array([0.5, 1.0]),
        np.array([0.5, 19.0]),
        np.array([1.0, 3.0]),
        np.array([1.0, 17.0]),
        np.array([0.8, 10.0]),
    ]
    idx = rng.choice(len(nodes), size=n_dogs, replace=False)
    jitter = rng.normal(0.0, 0.05, size=(n_dogs, 2))
    return np.vstack([nodes[i] for i in idx]) + jitter
