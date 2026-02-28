from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from collections import deque


PHASE_COLORS = {
    "seek": "#f39c12", "enclose": "#e74c3c", "herd": "#2ecc71",
    "collect": "#e67e22", "drive": "#1abc9c",
}
SHEEP_COLOR = "#3498db"
DOG_COLOR = "#e74c3c"
STRINGNET_COLOR = "#e74c3c"
TARGET_COLOR = "#f1c40f"
GOAL_COLOR = "#2ecc71"
HISTORY_LEN = 120

MODE_META = {
    "strombom":     {"label": "Strömbom (Rule-Based)",         "color": "#1a5276"},
    "strombom_llm": {"label": "Strömbom + Qwen3 + Adapter",    "color": "#7d3c98"},
    "stringnet":    {"label": "StringNet (Rule-Based)",         "color": "#1a5c34"},
    "stringnet_llm":{"label": "StringNet + Qwen3 + Adapter",   "color": "#8e44ad"},
}


class HerdingVisualizer:
    """Live matplotlib figure with arena, formation overlay, and metrics panel for all 4 modes."""

    def __init__(self, mode: str = "stringnet", figsize: tuple = (15, 7), arena_size: float = 20.0) -> None:
        self.mode = mode
        self.arena_size = float(arena_size)
        self.fig = plt.figure(figsize=figsize, facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title(f"Herding Simulation — {MODE_META.get(mode, {}).get('label', mode)}")
        gs = self.fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], hspace=0.38, wspace=0.32,
                                   left=0.06, right=0.97, top=0.90, bottom=0.12)
        self.ax_arena = self.fig.add_subplot(gs[:, 0])
        self.ax_escape = self.fig.add_subplot(gs[0, 1])
        self.ax_form = self.fig.add_subplot(gs[1, 1])
        self.ax_intent = self.fig.add_subplot(gs[0, 2])
        self.ax_margin = self.fig.add_subplot(gs[1, 2])

        self._goal_patch: patches.FancyBboxPatch | None = None
        self._goal_text = None
        self._setup_arena()
        self._setup_metric_axes()

        self._history_escape: deque[float] = deque(maxlen=HISTORY_LEN)
        self._history_form: deque[float] = deque(maxlen=HISTORY_LEN)
        self._history_margin: deque[float] = deque(maxlen=HISTORY_LEN)
        self._intent_counts: dict[str, int] = {}

        self._sheep_sc = None
        self._dog_sc = None
        self._dog_arrows: list = []
        self._net_lines: list = []
        self._target_sc = None
        self._trail_lines: list = []
        self._sheep_trails: list[deque] = []
        self._dog_trails: list[deque] = []
        self._initialised = False

        self._add_mode_indicator()
        plt.ion()

    def _setup_arena(self) -> None:
        """Configure the main 20x20 arena axis with default goal patch."""
        ax = self.ax_arena
        ax.set_facecolor("#0d0d1a")
        pad = 0.5
        ax.set_xlim(-pad, self.arena_size + pad)
        ax.set_ylim(-pad, self.arena_size + pad)
        ax.set_aspect("equal")
        ax.set_title("Arena", color="white", fontsize=11, pad=4)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")
        ax.tick_params(colors="#888899", labelsize=7)

        goal_w, goal_h = 0.4 * self.arena_size, 0.2 * self.arena_size
        goal_x = 0.7 * self.arena_size - goal_w / 2
        goal_y = 0.1 * self.arena_size - goal_h / 2
        self._goal_patch = patches.FancyBboxPatch(
            (goal_x, goal_y), goal_w, goal_h,
            boxstyle="round,pad=0.1", linewidth=1.5,
            edgecolor=GOAL_COLOR, facecolor="#1a3a1a", alpha=0.6, zorder=1,
        )
        ax.add_patch(self._goal_patch)
        self._goal_text = ax.text(goal_x + goal_w / 2, goal_y + goal_h / 2, "GOAL",
                                  ha="center", va="center",
                                  color=GOAL_COLOR, fontsize=9, fontweight="bold", zorder=2)

        self._phase_text = ax.text(0.02, 0.97, "Phase: —", transform=ax.transAxes,
                                   color="#f39c12", fontsize=9, va="top", fontweight="bold")
        self._step_text = ax.text(0.98, 0.97, "Step: 0", transform=ax.transAxes,
                                  color="white", fontsize=8, va="top", ha="right")
        self._success_text = ax.text(0.5, 0.5, "", transform=ax.transAxes,
                                     color="#f1c40f", fontsize=18, fontweight="bold",
                                     ha="center", va="center",
                                     bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", alpha=0.85),
                                     zorder=20)

    def _update_goal(self, state: dict) -> None:
        """Reposition goal patch using goal_region from state dict."""
        if "goal_region" not in state:
            return
        gr = np.asarray(state["goal_region"])
        x0, y0 = float(gr[0, 0]), float(gr[0, 1])
        w, h = float(gr[1, 0] - gr[0, 0]), float(gr[1, 1] - gr[0, 1])
        self._goal_patch.set_bounds(x0, y0, w, h)
        gx, gy = x0 + w / 2, y0 + h / 2
        self._goal_text.set_position((gx, gy))

    def _setup_metric_axes(self) -> None:
        """Style the four small metric sub-axes."""
        for ax, title in [
            (self.ax_escape, "Escape Prob"),
            (self.ax_form,   "Formation Error"),
            (self.ax_intent, "Intent Tokens"),
            (self.ax_margin, "Containment Margin"),
        ]:
            ax.set_facecolor("#0d0d1a")
            ax.set_title(title, color="white", fontsize=8, pad=3)
            ax.tick_params(colors="#888899", labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor("#444466")

    def _add_mode_indicator(self) -> None:
        """Add mode badge text at top of figure."""
        meta = MODE_META.get(self.mode, {"label": self.mode, "color": "#888888"})
        self._mode_text = self.fig.text(
            0.5, 0.96, meta["label"], ha="center", va="top",
            fontsize=12, fontweight="bold", color=meta["color"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                      edgecolor=meta["color"], alpha=0.9),
        )

    def _init_agents(self, n_sheep: int, n_dogs: int) -> None:
        """Lazily create scatter, trail, arrow, and net artists on first frame."""
        ax = self.ax_arena
        self._sheep_sc = ax.scatter([], [], s=80, c=SHEEP_COLOR, marker="o", zorder=5,
                                    edgecolors="white", linewidths=0.5, label="Sheep")
        self._dog_sc = ax.scatter([], [], s=130, c=DOG_COLOR, marker="^", zorder=6,
                                  edgecolors="white", linewidths=0.8, label="Dogs")
        self._target_sc = ax.scatter([], [], s=60, c=TARGET_COLOR, marker="x", zorder=4,
                                     linewidths=1.5, label="Targets")
        self._sheep_trails = [deque(maxlen=30) for _ in range(n_sheep)]
        self._dog_trails   = [deque(maxlen=30) for _ in range(n_dogs)]
        self._trail_lines = [
            ax.plot([], [], lw=0.6, alpha=0.28, color=SHEEP_COLOR, zorder=3)[0]
            for _ in range(n_sheep)
        ] + [
            ax.plot([], [], lw=0.8, alpha=0.38, color=DOG_COLOR, zorder=3)[0]
            for _ in range(n_dogs)
        ]
        self._dog_arrows = [
            ax.annotate("", xy=(0, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=DOG_COLOR, lw=1.2), zorder=7)
            for _ in range(n_dogs)
        ]
        is_stringnet = self.mode in ("stringnet", "stringnet_llm")
        self._net_lines = [
            ax.plot([], [], lw=1.5, color=STRINGNET_COLOR, alpha=0.7,
                    zorder=4, linestyle="--")[0]
            for _ in range(n_dogs)
        ] if is_stringnet else []
        ax.legend(loc="lower right", facecolor="#1a1a2e", edgecolor="#444466",
                  labelcolor="white", fontsize=7, markerscale=0.8)
        self._initialised = True

    def update(
        self,
        state: dict,
        metrics: dict,
        intent: dict,
        step: int,
        done: bool = False,
        success: bool = False,
        strombom_targets: np.ndarray | None = None,
    ) -> None:
        """Redraw all artists for the current timestep."""
        sheep_pos = np.asarray(state["sheep_pos"])
        dog_pos   = np.asarray(state["dog_pos"])
        dog_vel   = np.asarray(state["dog_vel"])
        n_sheep, n_dogs = len(sheep_pos), len(dog_pos)

        if not self._initialised:
            self._init_agents(n_sheep, n_dogs)

        self._update_goal(state)
        self._sheep_sc.set_offsets(sheep_pos)
        self._dog_sc.set_offsets(dog_pos)

        for i, pos in enumerate(sheep_pos):
            self._sheep_trails[i].append(pos.copy())
            t = np.array(self._sheep_trails[i])
            self._trail_lines[i].set_data(t[:, 0], t[:, 1])

        for j, (pos, vel) in enumerate(zip(dog_pos, dog_vel)):
            self._dog_trails[j].append(pos.copy())
            t = np.array(self._dog_trails[j])
            self._trail_lines[n_sheep + j].set_data(t[:, 0], t[:, 1])
            spd = float(np.linalg.norm(vel))
            if spd > 0.05:
                tip = pos + 0.6 * vel / spd
                self._dog_arrows[j].xy = tip
                self._dog_arrows[j].xytext = pos.tolist()

        if self._net_lines:
            for j in range(n_dogs):
                nxt = (j + 1) % n_dogs
                self._net_lines[j].set_data(
                    [dog_pos[j, 0], dog_pos[nxt, 0]],
                    [dog_pos[j, 1], dog_pos[nxt, 1]],
                )

        if self._target_sc is not None:
            if strombom_targets is not None:
                self._target_sc.set_offsets(strombom_targets)
            else:
                self._target_sc.set_offsets(np.empty((0, 2)))

        phase = state.get("phase", intent.get("phase", "—"))
        self._phase_text.set_text(f"Phase: {phase}")
        self._phase_text.set_color(PHASE_COLORS.get(phase, "white"))
        self._step_text.set_text(f"Step: {step}")

        if done:
            msg = "✓ SUCCESS!" if success else "✗ TIMEOUT"
            clr = "#2ecc71" if success else "#e74c3c"
            self._success_text.set_text(msg)
            self._success_text.get_bbox_patch().set_edgecolor(clr)
            self._success_text.set_color(clr)
        else:
            self._success_text.set_text("")

        esc  = metrics.get("escape_prob_est", 0)
        form = metrics.get("formation_error", 0)
        margin = metrics.get("containment_margin", 0)
        tok = intent.get("intent_token", "?")

        self._history_escape.append(esc)
        self._history_form.append(form)
        self._history_margin.append(margin)
        self._intent_counts[tok] = self._intent_counts.get(tok, 0) + 1

        for ax, hist, color, ylim in [
            (self.ax_escape, self._history_escape, "#e74c3c", (0, 1)),
            (self.ax_form,   self._history_form,   "#f39c12", (0, 2.5)),
            (self.ax_margin, self._history_margin, "#2ecc71", None),
        ]:
            ax.cla()
            ax.set_facecolor("#0d0d1a")
            for sp in ax.spines.values():
                sp.set_edgecolor("#444466")
            ax.tick_params(colors="#888899", labelsize=6)
            ax.plot(list(hist), color=color, lw=1.2)
            if ylim:
                ax.set_ylim(*ylim)
            if hist:
                ax.axhline(y=list(hist)[-1], color=color, lw=0.5, alpha=0.4, linestyle=":")

        labels = list(self._intent_counts.keys())
        vals   = [self._intent_counts[k] for k in labels]
        self.ax_intent.cla()
        self.ax_intent.set_facecolor("#0d0d1a")
        for sp in self.ax_intent.spines.values():
            sp.set_edgecolor("#444466")
        self.ax_intent.tick_params(colors="#888899", labelsize=5)
        colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 1)))
        self.ax_intent.barh(labels, vals, color=colors)
        self.ax_intent.set_title("Intent Tokens", color="white", fontsize=8, pad=3)

        self.ax_escape.set_title(f"Escape Prob  {esc:.2f}",    color="white", fontsize=8, pad=3)
        self.ax_form.set_title(f"Formation Err  {form:.2f}",   color="white", fontsize=8, pad=3)
        self.ax_margin.set_title(f"Containment  {margin:.2f}", color="white", fontsize=8, pad=3)

        src = intent.get("source", "rule")
        src_color = "#8e44ad" if "qwen" in src else "#27ae60"
        self.ax_arena.set_xlabel(f"Intent: {tok}  |  Source: {src}", color=src_color, fontsize=8)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        """Close the figure and disable interactive mode."""
        plt.ioff()
        plt.close(self.fig)