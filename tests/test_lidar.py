import numpy as np

from shepherd_codex.shepherd_env.sensors import lidar_scan


def test_lidar_occlusion_prefers_nearest() -> None:
    sheep = np.array([[2.0, 0.0], [3.0, 0.0]])
    scan = lidar_scan(np.array([0.0, 0.0]), 0.0, sheep, n_rays=9, max_range=10.0)
    mid = scan[scan.shape[0] // 2]
    assert mid[1] == 1.0
    assert mid[0] < 2.0
