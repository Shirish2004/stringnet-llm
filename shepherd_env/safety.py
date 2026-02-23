"""Safety shield for bounded controls and short-horizon collision checks."""

from __future__ import annotations

import numpy as np

from .dynamics import saturate_norm


def project_action(action: np.ndarray, state: dict, ubar_d: float, k_steps: int = 3) -> tuple[np.ndarray, bool]:
    """Project action to safe set: bound norm and avoid imminent dog-dog collisions."""
    a = saturate_norm(np.asarray(action, dtype=float), ubar_d)
    dog_pos = np.asarray(state["dog_pos"], dtype=float)
    dog_vel = np.asarray(state["dog_vel"], dtype=float)
    def_id = int(state.get("active_defender", 0))
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
