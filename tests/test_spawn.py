import numpy as np

from shepherd_codex.shepherd_env.spawn import spawn_sheep


def test_spawn_inside_left_quadrant_and_dmin() -> None:
    rng = np.random.default_rng(0)
    sheep, _, _ = spawn_sheep(8, rng, (0.6, 1.2), 0.15)
    assert np.all((sheep[:, 0] >= 0.0) & (sheep[:, 0] <= 8.0))
    assert np.all((sheep[:, 1] >= 5.0) & (sheep[:, 1] <= 15.0))
    for i in range(len(sheep)):
        for j in range(i + 1, len(sheep)):
            assert np.linalg.norm(sheep[i] - sheep[j]) >= 0.15
