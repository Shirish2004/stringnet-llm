"""Damped double-integrator dynamics and integration utilities."""

from __future__ import annotations

import numpy as np


def saturate_norm(u: np.ndarray, ubar: float) -> np.ndarray:
    """Project each 2D control vector in ``u`` to the closed ball ||u||<=ubar."""
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        n = np.linalg.norm(u)
        return u if n <= ubar or n == 0 else (ubar / n) * u
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    scale = np.ones_like(norms)
    mask = norms > ubar
    scale[mask] = ubar / np.maximum(norms[mask], 1e-9)
    return u * scale


def semi_implicit_euler_step(
    pos: np.ndarray,
    vel: np.ndarray,
    ctrl: np.ndarray,
    c_d: float,
    ubar: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One step for ``r_dot=v``, ``v_dot=u-C_D|v|v`` with semi-implicit Euler."""
    ctrl_sat = saturate_norm(ctrl, ubar)
    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    vdot = ctrl_sat - c_d * speed * vel
    vel_next = vel + dt * vdot
    pos_next = pos + dt * vel_next
    return pos_next, vel_next
