from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg" if sys.platform != "linux" else "Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shepherd_env.env import ShepherdEnv
from shepherd_env.controllers import (
    apply_seeking_controller,
    apply_enclosing_controller,
    apply_herding_controller,
)
from shepherd_env.sensors import feature_extractor
from planner.mock_llm import OracularPlanner
from planner.llm_planner import LLMPlanner
from metrics.failure_detector import FailureDetector
from visualizer import HerdingVisualizer


def _desired_formation(state: dict, radius_scale: float = 1.0, d_behind: float = 1.2) -> dict:
    """Compute semi-circular StringNet formation parameters from state."""
    acom = state["sheep_pos"].mean(axis=0)
    goal = state["goal"]
    u = goal - acom
    u = u / max(np.linalg.norm(u), 1e-9)
    return {
        "center": acom - d_behind * u,
        "phi": float(np.arctan2(u[1], u[0])),
        "radius": 1.4 * radius_scale,
    }


def _desired_positions(formation: dict, n_d: int) -> np.ndarray:
    """Enumerate desired dog positions on the StringNet semi-circle."""
    phi, rad, c = formation["phi"], formation["radius"], formation["center"]
    return np.vstack([
        c + rad * np.array([
            np.cos(phi + np.pi / 2 + np.pi * j / max(1, n_d - 1)),
            np.sin(phi + np.pi / 2 + np.pi * j / max(1, n_d - 1)),
        ]) for j in range(n_d)
    ])


def select_mode_interactive() -> tuple[str, int, int, str]:
    """Matplotlib-based mode selector; returns (mode, n_sheep, n_dogs, qwen_model)."""
    selected = {"mode": None, "n_a": 5, "n_d": 3, "qwen_model": "Qwen/Qwen3-0.6B"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.canvas.manager.set_window_title("StringNet Herding — Mode Selection")

    ax.text(0.5, 0.91, "StringNet Herding", ha="center", fontsize=18,
            fontweight="bold", color="white")
    ax.text(0.5, 0.81, "Chipade & Panagou (2021)  ·  Qwen3 + PyTorch Adapter",
            ha="center", fontsize=9, color="#8888aa")

    ax_btn1 = fig.add_axes([0.07, 0.58, 0.38, 0.13])
    ax_btn2 = fig.add_axes([0.55, 0.58, 0.38, 0.13])
    ax_na   = fig.add_axes([0.07, 0.40, 0.18, 0.09])
    ax_nd   = fig.add_axes([0.27, 0.40, 0.18, 0.09])
    ax_q06  = fig.add_axes([0.55, 0.40, 0.18, 0.09])
    ax_q17  = fig.add_axes([0.75, 0.40, 0.18, 0.09])
    ax_go   = fig.add_axes([0.35, 0.18, 0.30, 0.13])

    btn1   = Button(ax_btn1, "StringNet\n(Rule-Based)", color="#1a5c34", hovercolor="#27ae60")
    btn2   = Button(ax_btn2, "StringNet + Qwen3\n+ PyTorch Adapter", color="#4a1870", hovercolor="#8e44ad")
    btn_na = Button(ax_na, f"Sheep: {selected['n_a']}", color="#1a2a3a", hovercolor="#2c3e50")
    btn_nd = Button(ax_nd, f"Dogs: {selected['n_d']}", color="#1a2a3a", hovercolor="#2c3e50")
    btn_q06 = Button(ax_q06, "Qwen3-0.6B", color="#1a2a3a", hovercolor="#2c3e50")
    btn_q17 = Button(ax_q17, "Qwen3-1.7B", color="#1a2a3a", hovercolor="#2c3e50")
    btn_go  = Button(ax_go, "▶  START", color="#7b1010", hovercolor="#c0392b")

    for btn in (btn1, btn2, btn_na, btn_nd, btn_q06, btn_q17, btn_go):
        btn.label.set_color("white")
        btn.label.set_fontsize(9)
        btn.label.set_fontweight("bold")

    info = ax.text(0.5, 0.30, "Select a mode above", ha="center", fontsize=9, color="#f39c12")

    labels_row = [
        ax.text(0.16, 0.50, "Agents", ha="center", fontsize=7, color="#888899"),
        ax.text(0.64, 0.50, "Qwen model", ha="center", fontsize=7, color="#888899"),
    ]

    def _refresh():
        m = selected["mode"] or "—"
        info.set_text(f"Mode: {m}  |  Sheep: {selected['n_a']}  |  Dogs: {selected['n_d']}"
                      f"  |  Model: {selected['qwen_model'].split('/')[-1]}")
        fig.canvas.draw_idle()

    def on1(e): selected["mode"] = "stringnet"; _refresh()
    def on2(e): selected["mode"] = "llm"; _refresh()
    def on_na(e):
        selected["n_a"] = selected["n_a"] % 9 + 2
        btn_na.label.set_text(f"Sheep: {selected['n_a']}")
        _refresh()
    def on_nd(e):
        selected["n_d"] = (selected["n_d"] - 2) % 4 + 2
        btn_nd.label.set_text(f"Dogs: {selected['n_d']}")
        _refresh()
    def on_q06(e): selected["qwen_model"] = "Qwen/Qwen3-0.6B"; _refresh()
    def on_q17(e): selected["qwen_model"] = "Qwen/Qwen3-1.7B"; _refresh()
    def on_go(e):
        if selected["mode"] is None:
            info.set_text("⚠  Please select a mode first!")
            info.set_color("#e74c3c")
            fig.canvas.draw_idle()
            return
        plt.close(fig)

    btn1.on_clicked(on1)
    btn2.on_clicked(on2)
    btn_na.on_clicked(on_na)
    btn_nd.on_clicked(on_nd)
    btn_q06.on_clicked(on_q06)
    btn_q17.on_clicked(on_q17)
    btn_go.on_clicked(on_go)

    plt.show(block=True)
    return (
        selected["mode"] or "stringnet",
        selected["n_a"],
        selected["n_d"],
        selected["qwen_model"],
    )


def run_simulation(
    mode: str,
    n_a: int = 5,
    n_d: int = 3,
    seed: int = 0,
    qwen_model: str = "Qwen/Qwen3-0.6B",
    render_every: int = 1,
) -> dict:
    """Run one live episode with real-time visualisation."""
    env = ShepherdEnv(config_path=str(Path(__file__).parent / "configs" / "default.yaml"))
    state = env.reset(seed=seed, config={"N_a": n_a, "N_d": n_d})
    config = env.config

    planner: LLMPlanner | None = None
    oracle: OracularPlanner | None = None
    if mode == "llm":
        planner = LLMPlanner(
            adapter_dim=config.get("adapter_dim", 64),
            qwen_model=qwen_model,
            seed=seed,
        )
        oracle = OracularPlanner()
        print(f"Qwen available: {planner.qwen.available}")

    fd = FailureDetector(
        margin_thresh=config.get("containment_margin_thresh", 0.0),
        err_thresh=config.get("formation_error_thresh", 0.4),
        t_fail=config.get("T_fail", 8),
    )

    viz = HerdingVisualizer(mode=mode)
    T_max = config.get("T_max", 1000)
    max_adapter = config.get("max_adapter_updates_per_episode", 3)
    adapter_updates = 0
    done = False
    step = 0
    intent: dict = {"intent_token": "widen_net", "phase": "seek", "params": {}, "source": "rule"}
    metrics: dict = {"escape_prob_est": 0.0, "formation_error": 0.0, "containment_margin": 0.0}
    info: dict = {}

    while not done and step < T_max:
        tokens = feature_extractor(state)
        current_phase = state.get("phase", "seek")

        if mode == "llm" and planner is not None:
            intent = planner.plan(tokens, current_phase=current_phase)
        else:
            esc = tokens["escape_prob_est"]
            spread = tokens["sheep_spread"]
            tok = "tighten_net" if esc > 0.45 else "focus_largest_cluster" if spread > 0.08 else "widen_net"
            intent = {
                "intent_token": tok,
                "phase": current_phase,
                "params": {"radius_scale": 0.9 if tok == "tighten_net" else 1.0},
                "source": "rule",
            }

        rs = float(intent.get("params", {}).get("radius_scale", 1.0))
        formation = _desired_formation(state, radius_scale=rs, d_behind=config.get("d_behind", 1.2))
        xi_des = _desired_positions(formation, n_d)

        phase = intent.get("phase", current_phase)
        acts: dict[int, np.ndarray] = {}
        for j in range(n_d):
            if phase == "seek":
                acts[j] = apply_seeking_controller(j, formation, state, config)
            elif phase == "enclose":
                acts[j] = apply_enclosing_controller(j, formation, state, config)
            else:
                acts[j] = apply_herding_controller(j, formation, state, config)

        state, _, done, info = env.step(acts)
        metrics = fd.step(state, xi_des, tokens)

        if mode == "llm" and planner is not None and oracle is not None:
            if metrics["failure"] and adapter_updates < max_adapter:
                corr = oracle.corrective_intent(tokens)
                planner.logged_update(tokens, corr,
                                      lr=config.get("adapter_lr", 5e-4),
                                      epochs=config.get("adapter_epochs", 3))
                adapter_updates += 1

        step += 1
        if step % render_every == 0:
            viz.update(state, metrics, intent, step, done=done, success=info.get("success", False))
            plt.pause(0.001)

    print(f"\n{'='*50}")
    print(f"Mode: {mode.upper()}")
    print(f"Steps: {step}  |  Success: {info.get('success', False)}")
    print(f"Sheep in goal: {info.get('in_goal', 0)}/{n_a}")
    if mode == "llm":
        print(f"Adapter updates: {adapter_updates}")
        print(f"Replay buffer size: {planner.buffer.__len__() if planner else 0}")
    print(f"{'='*50}")

    input("\nPress Enter to close...")
    viz.close()
    return {"success": info.get("success", False), "steps": step, "adapter_updates": adapter_updates}


def main() -> None:
    """Parse CLI args or show interactive selector, then run simulation."""
    parser = argparse.ArgumentParser(description="StringNet Herding — Qwen3 + PyTorch Adapter")
    parser.add_argument("--mode", choices=["stringnet", "llm", "select"], default="select")
    parser.add_argument("--n-sheep", type=int, default=5)
    parser.add_argument("--n-dogs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen3-1.7B)")
    parser.add_argument("--render-every", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "select":
        try:
            mode, n_a, n_d, qwen_model = select_mode_interactive()
        except Exception:
            print("Interactive selector failed; defaulting to stringnet.")
            mode, n_a, n_d, qwen_model = "stringnet", args.n_sheep, args.n_dogs, args.qwen_model
    else:
        mode, n_a, n_d, qwen_model = args.mode, args.n_sheep, args.n_dogs, args.qwen_model

    if mode is None:
        return

    run_simulation(
        mode=mode, n_a=n_a, n_d=n_d, seed=args.seed,
        qwen_model=qwen_model, render_every=args.render_every,
    )


if __name__ == "__main__":
    main()