"""Real-time matplotlib visualizer for StringNet herding simulation."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from collections import deque


PHASE_COLORS = {"seek": "#f39c12", "enclose": "#e74c3c", "herd": "#2ecc71"}
SHEEP_COLOR = "#3498db"
DOG_COLOR = "#e74c3c"
STRINGNET_COLOR = "#e74c3c"
GOAL_COLOR = "#2ecc71"
HISTORY_LEN = 120


class HerdingVisualizer:
    """Live matplotlib figure with arena, StringNet overlay, and metrics panel."""

    def __init__(self, mode: str = "stringnet", figsize: tuple = (14, 7)) -> None:
        self.mode = mode
        self.fig = plt.figure(figsize=figsize, facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title("StringNet Herding — Real-Time Simulation")
        gs = self.fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], hspace=0.35, wspace=0.3,
                                   left=0.06, right=0.97, top=0.90, bottom=0.12)

        self.ax_arena = self.fig.add_subplot(gs[:, 0])
        self.ax_escape = self.fig.add_subplot(gs[0, 1])
        self.ax_form = self.fig.add_subplot(gs[1, 1])
        self.ax_intent = self.fig.add_subplot(gs[0, 2])
        self.ax_margin = self.fig.add_subplot(gs[1, 2])

        self._setup_arena()
        self._setup_metric_axes()

        self._history_escape: deque[float] = deque(maxlen=HISTORY_LEN)
        self._history_form: deque[float] = deque(maxlen=HISTORY_LEN)
        self._history_margin: deque[float] = deque(maxlen=HISTORY_LEN)
        self._intent_counts: dict[str, int] = {}

        self._sheep_sc = None
        self._dog_sc = None
        self._dog_arrows: list = []
        self._stringnet_lines: list = []
        self._trail_lines: list = []
        self._sheep_trails: list[deque] = []
        self._dog_trails: list[deque] = []
        # Text artists (created in _setup_arena / _add_mode_indicator)
        # don't overwrite them here — _setup_arena() sets these up.
        self._initialised = False

        self._add_mode_indicator()
        plt.ion()

    def _setup_arena(self) -> None:
        """Configure the main 20x20 arena axis."""
        ax = self.ax_arena
        ax.set_facecolor("#0d0d1a")
        ax.set_xlim(-0.5, 20.5)
        ax.set_ylim(-0.5, 20.5)
        ax.set_aspect("equal")
        ax.set_title("Arena", color="white", fontsize=11, pad=4)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.tick_params(colors="#888899", labelsize=7)

        goal = patches.FancyBboxPatch(
            (12.0, 8.0), 8.0, 4.0,
            boxstyle="round,pad=0.1",
            linewidth=1.5, edgecolor=GOAL_COLOR, facecolor="#1a3a1a", alpha=0.6, zorder=1,
        )
        ax.add_patch(goal)
        ax.text(16.0, 10.0, "GOAL", ha="center", va="center",
                color=GOAL_COLOR, fontsize=9, fontweight="bold", zorder=2)

        self._phase_text = ax.text(
            0.02, 0.97, "Phase: seek", transform=ax.transAxes,
            color=PHASE_COLORS["seek"], fontsize=9, va="top", fontweight="bold",
        )
        self._step_text = ax.text(
            0.98, 0.97, "Step: 0", transform=ax.transAxes,
            color="white", fontsize=8, va="top", ha="right",
        )
        self._success_text = ax.text(
            0.5, 0.5, "", transform=ax.transAxes,
            color="#f1c40f", fontsize=18, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", alpha=0.85),
            zorder=20,
        )

    def _setup_metric_axes(self) -> None:
        """Style the four small metric sub-axes."""
        for ax, title in [
            (self.ax_escape, "Escape Prob"),
            (self.ax_form, "Formation Error"),
            (self.ax_intent, "Intent Tokens"),
            (self.ax_margin, "Containment Margin"),
        ]:
            ax.set_facecolor("#0d0d1a")
            ax.set_title(title, color="white", fontsize=8, pad=3)
            ax.tick_params(colors="#888899", labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")

    def _add_mode_indicator(self) -> None:
        """Add mode badge text at top of figure."""
        label = "🐕 StringNet" if self.mode == "stringnet" else "🧠 StringNet + LLM Adapter"
        color = "#27ae60" if self.mode == "stringnet" else "#8e44ad"
        self._mode_text = self.fig.text(
            0.5, 0.96, label, ha="center", va="top",
            fontsize=12, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor=color, alpha=0.9),
        )

    def _init_agents(self, n_sheep: int, n_dogs: int) -> None:
        """Lazily create scatter and arrow artists on first call."""
        ax = self.ax_arena
        self._sheep_sc = ax.scatter(
            [], [], s=80, c=SHEEP_COLOR, marker="o", zorder=5,
            edgecolors="white", linewidths=0.5, label="Sheep",
        )
        self._dog_sc = ax.scatter(
            [], [], s=130, c=DOG_COLOR, marker="^", zorder=6,
            edgecolors="white", linewidths=0.8, label="Dogs",
        )
        self._sheep_trails = [deque(maxlen=30) for _ in range(n_sheep)]
        self._dog_trails = [deque(maxlen=30) for _ in range(n_dogs)]
        self._trail_lines = [
            ax.plot([], [], lw=0.6, alpha=0.3, color=SHEEP_COLOR, zorder=3)[0]
            for _ in range(n_sheep)
        ] + [
            ax.plot([], [], lw=0.8, alpha=0.4, color=DOG_COLOR, zorder=3)[0]
            for _ in range(n_dogs)
        ]
        self._dog_arrows = [
            ax.annotate("", xy=(0, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=DOG_COLOR, lw=1.2), zorder=7)
            for _ in range(n_dogs)
        ]
        self._stringnet_lines = [
            ax.plot([], [], lw=1.5, color=STRINGNET_COLOR, alpha=0.7, zorder=4, linestyle="--")[0]
            for _ in range(n_dogs)
        ]
        ax.legend(loc="lower right", facecolor="#1a1a2e", edgecolor="#444466",
                  labelcolor="white", fontsize=7, markerscale=0.8)
        self._initialised = True

    def update(self, state: dict, metrics: dict, intent: dict, step: int, done: bool = False, success: bool = False) -> None:
        """Redraw all artists for the current timestep."""
        sheep_pos = np.asarray(state["sheep_pos"])
        dog_pos = np.asarray(state["dog_pos"])
        dog_vel = np.asarray(state["dog_vel"])
        n_sheep, n_dogs = len(sheep_pos), len(dog_pos)

        if not self._initialised:
            self._init_agents(n_sheep, n_dogs)

        self._sheep_sc.set_offsets(sheep_pos)
        self._dog_sc.set_offsets(dog_pos)

        for i, pos in enumerate(sheep_pos):
            self._sheep_trails[i].append(pos.copy())
            trail = np.array(self._sheep_trails[i])
            self._trail_lines[i].set_data(trail[:, 0], trail[:, 1])

        for j, (pos, vel) in enumerate(zip(dog_pos, dog_vel)):
            self._dog_trails[j].append(pos.copy())
            trail = np.array(self._dog_trails[j])
            self._trail_lines[n_sheep + j].set_data(trail[:, 0], trail[:, 1])
            speed = np.linalg.norm(vel)
            if speed > 0.05:
                tip = pos + 0.6 * vel / speed
                self._dog_arrows[j].set_position(pos)
                self._dog_arrows[j].xy = tip
                self._dog_arrows[j].xytext = pos.tolist()

        for j in range(n_dogs):
            nxt = (j + 1) % n_dogs
            xs = [dog_pos[j, 0], dog_pos[nxt, 0]]
            ys = [dog_pos[j, 1], dog_pos[nxt, 1]]
            self._stringnet_lines[j].set_data(xs, ys)

        phase = state.get("phase", "seek")
        self._phase_text.set_text(f"Phase: {phase}")
        self._phase_text.set_color(PHASE_COLORS.get(phase, "white"))
        self._step_text.set_text(f"Step: {step}")

        if done:
            msg = "✓ SUCCESS!" if success else "✗ TIMEOUT"
            color = "#2ecc71" if success else "#e74c3c"
            self._success_text.set_text(msg)
            self._success_text.get_bbox_patch().set_edgecolor(color)
            self._success_text.set_color(color)
        else:
            self._success_text.set_text("")

        esc = metrics.get("escape_prob_est", 0)
        form = metrics.get("formation_error", 0)
        margin = metrics.get("containment_margin", 0)
        tok = intent.get("intent_token", "?")

        self._history_escape.append(esc)
        self._history_form.append(form)
        self._history_margin.append(margin)
        self._intent_counts[tok] = self._intent_counts.get(tok, 0) + 1

        for ax, hist, color, ylim in [
            (self.ax_escape, self._history_escape, "#e74c3c", (0, 1)),
            (self.ax_form, self._history_form, "#f39c12", (0, 2)),
            (self.ax_margin, self._history_margin, "#2ecc71", None),
        ]:
            ax.cla()
            ax.set_facecolor("#0d0d1a")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")
            ax.tick_params(colors="#888899", labelsize=6)
            ax.plot(list(hist), color=color, lw=1.2)
            if ylim:
                ax.set_ylim(*ylim)
            ax.axhline(y=list(hist)[-1], color=color, lw=0.5, alpha=0.4, linestyle=":")

        labels = list(self._intent_counts.keys())
        values = [self._intent_counts[k] for k in labels]
        self.ax_intent.cla()
        self.ax_intent.set_facecolor("#0d0d1a")
        for spine in self.ax_intent.spines.values():
            spine.set_edgecolor("#444466")
        self.ax_intent.tick_params(colors="#888899", labelsize=5)
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        self.ax_intent.barh(labels, values, color=colors)
        self.ax_intent.set_title("Intent Tokens", color="white", fontsize=8, pad=3)

        self.ax_escape.set_title(f"Escape Prob  {esc:.2f}", color="white", fontsize=8, pad=3)
        self.ax_form.set_title(f"Formation Err  {form:.2f}", color="white", fontsize=8, pad=3)
        self.ax_margin.set_title(f"Containment  {margin:.2f}", color="white", fontsize=8, pad=3)

        source = intent.get("source", "rule")
        src_color = "#8e44ad" if "qwen" in source else "#27ae60"
        self.ax_arena.set_xlabel(
            f"Intent: {tok}  |  Source: {source}",
            color=src_color, fontsize=8,
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        """Close the figure."""
        plt.ioff()
        plt.close(self.fig)