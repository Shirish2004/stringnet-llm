import numpy as np

from shepherd_codex.shepherd_env.controllers import (
    apply_enclosing_controller,
    apply_herding_controller,
    apply_seeking_controller,
)


def _state():
    return {
        "dog_pos": np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]),
        "dog_vel": np.zeros((3, 2)),
    }


def test_controller_outputs_are_bounded() -> None:
    config = {"ubar_d": 3.0, "k1": 2.0, "k2": 1.5, "alpha1": 0.7, "alpha2": 0.8, "C_D": 0.1}
    form = {"center": np.array([4.0, 4.0]), "phi": 0.2, "radius": 1.2}
    for fn in [apply_seeking_controller, apply_enclosing_controller, apply_herding_controller]:
        u = fn(0, form, _state(), config)
        assert u.shape == (2,)
        assert np.linalg.norm(u) <= config["ubar_d"] + 1e-6
