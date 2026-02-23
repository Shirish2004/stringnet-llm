"""Real-time PyQt viewer for shepherding with planner prompt/intent visibility."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import yaml

from metrics.failure_detector import FailureDetector
from planner.llm_planner import LLMPlanner
from planner.mock_llm import OracularPlanner
from shepherd_env.controllers import apply_enclosing_controller, apply_herding_controller, apply_seeking_controller
from shepherd_env.env import ShepherdEnv
from shepherd_env.sensors import feature_extractor

try:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtGui import QColor, QPainter, QPen
    from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPlainTextEdit, QVBoxLayout, QWidget
except Exception as exc:  # pragma: no cover - import availability is environment-dependent
    raise SystemExit(f"PyQt5 is required for live_viewer.py: {exc}")


class ViewerWindow(QMainWindow):
    """Simple viewer: left canvas for agents, right panel for LLM outputs and fine-tune logs."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.env = ShepherdEnv("./configs/default.yaml")
        self.state = self.env.reset(seed=cfg["seed"])
        self.planner = LLMPlanner(adapter_dim=cfg["adapter_dim"], seed=cfg["seed"])
        self.oracle = OracularPlanner(self.planner.vocab)
        self.detector = FailureDetector(cfg["containment_margin_thresh"], cfg["formation_error_thresh"], cfg["T_fail"])
        self.updates = 0
        self.t = 0
        self.intent = None

        self.setWindowTitle("Shepherd Codex Live Viewer")
        self.resize(1100, 760)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self.status = QLabel("Running...")
        self.prompt_box = QPlainTextEdit()
        self.prompt_box.setReadOnly(True)
        self.prompt_box.setPlaceholderText("LLM prompt and planner output")

        layout.addWidget(self.status)
        layout.addWidget(self.prompt_box)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(int(cfg["dt"] * 1000))

    def _draw_scene(self) -> None:
        pix = self.grab()
        painter = QPainter(pix)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(245, 245, 245))

        # map area
        map_w = int(w * 0.62)
        map_h = int(h * 0.92)
        ox, oy = 20, 40
        painter.setPen(QPen(QColor(40, 40, 40), 2))
        painter.drawRect(ox, oy, map_w, map_h)

        # goal region
        gx0, gy0 = self.state["goal_region"][0]
        gx1, gy1 = self.state["goal_region"][1]
        qx0 = ox + int((gx0 / 20.0) * map_w)
        qx1 = ox + int((gx1 / 20.0) * map_w)
        qy0 = oy + int((gy0 / 20.0) * map_h)
        qy1 = oy + int((gy1 / 20.0) * map_h)
        painter.fillRect(qx0, qy0, qx1 - qx0, qy1 - qy0, QColor(204, 255, 204))

        # sheep/dogs
        for p in self.state["sheep_pos"]:
            x = ox + int((p[0] / 20.0) * map_w)
            y = oy + int((p[1] / 20.0) * map_h)
            painter.setBrush(QColor(40, 120, 255))
            painter.drawEllipse(x - 5, y - 5, 10, 10)
        for p in self.state["dog_pos"]:
            x = ox + int((p[0] / 20.0) * map_w)
            y = oy + int((p[1] / 20.0) * map_h)
            painter.setBrush(QColor(200, 40, 40))
            painter.drawEllipse(x - 7, y - 7, 14, 14)

        painter.end()
        self.setMask(pix.mask())

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        w, h = self.width(), self.height()
        map_w = int(w * 0.62)
        map_h = int(h * 0.92)
        ox, oy = 20, 40
        painter.fillRect(ox, oy, map_w, map_h, QColor(250, 250, 250))
        painter.setPen(QPen(QColor(40, 40, 40), 2))
        painter.drawRect(ox, oy, map_w, map_h)

        gx0, gy0 = self.state["goal_region"][0]
        gx1, gy1 = self.state["goal_region"][1]
        qx0 = ox + int((gx0 / 20.0) * map_w)
        qx1 = ox + int((gx1 / 20.0) * map_w)
        qy0 = oy + int((gy0 / 20.0) * map_h)
        qy1 = oy + int((gy1 / 20.0) * map_h)
        painter.fillRect(qx0, qy0, qx1 - qx0, qy1 - qy0, QColor(204, 255, 204))

        for p in self.state["sheep_pos"]:
            x = ox + int((p[0] / 20.0) * map_w)
            y = oy + int((p[1] / 20.0) * map_h)
            painter.setBrush(QColor(40, 120, 255))
            painter.drawEllipse(x - 5, y - 5, 10, 10)

        for p in self.state["dog_pos"]:
            x = ox + int((p[0] / 20.0) * map_w)
            y = oy + int((p[1] / 20.0) * map_h)
            painter.setBrush(QColor(200, 40, 40))
            painter.drawEllipse(x - 7, y - 7, 14, 14)

    def tick(self) -> None:
        if self.t >= self.cfg["T_max"]:
            self.timer.stop()
            self.status.setText(f"Finished at T_max, updates={self.updates}")
            return

        tokens = feature_extractor(self.state)
        self.intent = self.planner.plan(tokens)
        formation = {"center": self.state["sheep_pos"].mean(axis=0), "phi": 0.0, "radius": 1.3 * self.intent["params"]["radius_scale"]}

        acts = {}
        for j in range(self.state["dog_pos"].shape[0]):
            if self.intent["phase"] == "seek":
                acts[j] = apply_seeking_controller(j, formation, self.state, self.cfg)
            elif self.intent["phase"] == "enclose":
                acts[j] = apply_enclosing_controller(j, formation, self.state, self.cfg)
            else:
                acts[j] = apply_herding_controller(j, formation, self.state, self.cfg)

        xi_des = np.tile(formation["center"][None, :], (self.state["dog_pos"].shape[0], 1))
        self.state, _, done, info = self.env.step(acts)
        metrics = self.detector.step(self.state, xi_des, tokens)

        if metrics["failure"] and self.updates < self.cfg["max_adapter_updates_per_episode"]:
            oracle_plan = self.oracle.corrective_intent(tokens)
            log = self.planner.logged_update(tokens, oracle_plan, lr=self.cfg["adapter_lr"], epochs=self.cfg["adapter_epochs"], seed=self.cfg["seed"])
            self.updates += 1
            update_text = (
                f"\n\n[ADAPTER UPDATE]\npre={log['pre_plan']['intent_token']} post={log['post_plan']['intent_token']} "
                f"oracle={oracle_plan['intent_token']}\nloss={log['loss_history']}"
            )
        else:
            update_text = ""

        self.status.setText(
            f"t={self.t} intent={self.intent['intent_token']} phase={self.intent['phase']} "
            f"escape_prob={tokens['escape_prob_est']:.3f} failures={self.updates}"
        )
        self.prompt_box.setPlainText(
            f"Prompt to LLM:\n{self.intent['llm_prompt']}\n\n"
            f"Planner output:\n{self.intent}\n"
            f"Metrics:\n{metrics}{update_text}"
        )

        self.t += 1
        self.update()
        if done:
            self.timer.stop()
            self.status.setText(f"Episode ended at t={self.t}, success={info['success']}, updates={self.updates}")


def main() -> None:
    with open("./configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    app = QApplication(sys.argv)
    win = ViewerWindow(cfg)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
