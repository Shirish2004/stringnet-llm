from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
if sys.platform != "linux":
    matplotlib.use("TkAgg")
else:
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    matplotlib.use("TkAgg" if has_display else "Agg")

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
from shepherd_env.strombom_controller import compute_strombom_targets, strombom_action, check_termination
from shepherd_env.sensors import feature_extractor
from planner.mock_llm import OracularPlanner, StrombomOracle
from planner.llm_planner import LLMPlanner, StrombomLLMPlanner
from metrics.failure_detector import FailureDetector
from visualizer import HerdingVisualizer


_BASE_COLLECT_RADIUS  = 2.5
_BASE_D_COLLECT       = 1.5
_BASE_D_BEHIND        = 3.0
_BASE_FORMATION_RADIUS = 2.5


def _desired_formation_stringnet(state: dict, radius_scale: float = 1.0, d_behind: float = 1.2) -> dict:
    """Compute semi-circular StringNet formation parameters from state."""
    acom = state["sheep_pos"].mean(axis=0)
    goal = state["goal"]
    u = goal - acom
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    return {
        "center": acom - d_behind * u,
        "phi": float(np.arctan2(u[1], u[0])),
        "radius": 1.4 * radius_scale,
    }


def _desired_positions_stringnet(formation: dict, n_d: int) -> np.ndarray:
    """Enumerate desired dog positions on the StringNet semi-circle."""
    phi, rad, c = formation["phi"], formation["radius"], formation["center"]
    return np.vstack([
        c + rad * np.array([
            np.cos(phi + np.pi / 2 + np.pi * j / max(1, n_d - 1)),
            np.sin(phi + np.pi / 2 + np.pi * j / max(1, n_d - 1)),
        ])
        for j in range(n_d)
    ])


def select_mode_interactive() -> tuple[str, int, int, str]:
    """Matplotlib 4-mode selector; returns (mode, n_sheep, n_dogs, qwen_model)."""
    sel = {"mode": None, "n_a": 5, "n_d": 3, "qwen_model": "Qwen/Qwen3-0.6B"}
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.canvas.manager.set_window_title("Herding Simulation — Mode Selection")

    ax.text(0.5, 0.93, "Herding Simulation", ha="center", fontsize=18, fontweight="bold", color="white")
    ax.text(0.5, 0.84, "Strömbom vs StringNet  ·  With & Without Qwen3 + PyTorch Adapter",
            ha="center", fontsize=9, color="#8888aa")

    btns_def = [
        ([0.04, 0.60, 0.43, 0.14], "strombom",     "Strömbom\n(Rule-Based)",          "#1a3a5c", "#2980b9"),
        ([0.53, 0.60, 0.43, 0.14], "strombom_llm", "Strömbom + Qwen3\n+ Adapter",     "#3d1a5c", "#8e44ad"),
        ([0.04, 0.42, 0.43, 0.14], "stringnet",    "StringNet\n(Rule-Based)",          "#1a5c34", "#27ae60"),
        ([0.53, 0.42, 0.43, 0.14], "stringnet_llm","StringNet + Qwen3\n+ Adapter",     "#4a1870", "#9b59b6"),
    ]
    mode_buttons = []
    for rect, key, label, color, hover in btns_def:
        axb = fig.add_axes(rect)
        b = Button(axb, label, color=color, hovercolor=hover)
        b.label.set_color("white"); b.label.set_fontsize(9); b.label.set_fontweight("bold")
        mode_buttons.append((b, key))

    ax_na  = fig.add_axes([0.04, 0.26, 0.19, 0.09])
    ax_nd  = fig.add_axes([0.27, 0.26, 0.19, 0.09])
    ax_q06 = fig.add_axes([0.53, 0.26, 0.19, 0.09])
    ax_q17 = fig.add_axes([0.76, 0.26, 0.19, 0.09])
    ax_go  = fig.add_axes([0.35, 0.08, 0.30, 0.13])

    btn_na  = Button(ax_na,  f"Sheep: {sel['n_a']}", color="#1a2a3a", hovercolor="#2c3e50")
    btn_nd  = Button(ax_nd,  f"Dogs: {sel['n_d']}",  color="#1a2a3a", hovercolor="#2c3e50")
    btn_q06 = Button(ax_q06, "Qwen3-0.6B",           color="#1a2a3a", hovercolor="#2c3e50")
    btn_q17 = Button(ax_q17, "Qwen3-1.7B",           color="#1a2a3a", hovercolor="#2c3e50")
    btn_go  = Button(ax_go,  "▶  START",              color="#7b1010", hovercolor="#c0392b")

    for btn in (btn_na, btn_nd, btn_q06, btn_q17, btn_go):
        btn.label.set_color("white"); btn.label.set_fontsize(9); btn.label.set_fontweight("bold")

    info = ax.text(0.5, 0.34, "Select a mode above", ha="center", fontsize=9, color="#f39c12")
    ax.text(0.155, 0.36, "Agents", ha="center", fontsize=7, color="#888899")
    ax.text(0.625, 0.36, "Qwen model", ha="center", fontsize=7, color="#888899")

    COLORS_SELECTED = {
        "strombom": "#2980b9", "strombom_llm": "#8e44ad",
        "stringnet": "#27ae60", "stringnet_llm": "#9b59b6",
    }

    def _refresh() -> None:
        m = sel["mode"] or "—"
        info.set_text(
            f"Mode: {m}  |  Sheep: {sel['n_a']}  |  Dogs: {sel['n_d']}"
            f"  |  Model: {sel['qwen_model'].split('/')[-1]}"
        )
        info.set_color("#f39c12")
        fig.canvas.draw_idle()

    def _make_mode_cb(key):
        def cb(e):
            sel["mode"] = key
            _refresh()
        return cb

    for btn, key in mode_buttons:
        btn.on_clicked(_make_mode_cb(key))

    def on_na(e):
        sel["n_a"] = sel["n_a"] % 10 + 2
        btn_na.label.set_text(f"Sheep: {sel['n_a']}")
        _refresh()

    def on_nd(e):
        sel["n_d"] = (sel["n_d"] - 2) % 4 + 2
        btn_nd.label.set_text(f"Dogs: {sel['n_d']}")
        _refresh()

    def on_q06(e): sel["qwen_model"] = "Qwen/Qwen3-0.6B"; _refresh()
    def on_q17(e): sel["qwen_model"] = "Qwen/Qwen3-1.7B"; _refresh()

    def on_go(e):
        if sel["mode"] is None:
            info.set_text("⚠  Please select a mode first!")
            info.set_color("#e74c3c")
            fig.canvas.draw_idle()
            return
        plt.close(fig)

    btn_na.on_clicked(on_na)
    btn_nd.on_clicked(on_nd)
    btn_q06.on_clicked(on_q06)
    btn_q17.on_clicked(on_q17)
    btn_go.on_clicked(on_go)

    plt.show(block=True)
    return sel["mode"] or "stringnet", sel["n_a"], sel["n_d"], sel["qwen_model"]


def run_simulation(
    mode: str,
    n_a: int = 5,
    n_d: int = 3,
    seed: int | None = None,
    qwen_model: str = "Qwen/Qwen3-0.6B",
    render_every: int = 1,
    debug_every: int = 25,
    speed_scale: float = 1.2,
    config_path: str | None = None,
) -> dict:
    """Run one live episode with real-time visualisation for any of the 4 modes."""
    cfg_path = config_path or str(Path(__file__).parent / "configs" / "default.yaml")
    env = ShepherdEnv(config_path=cfg_path)
    state = env.reset(seed=seed, config={"N_a": n_a, "N_d": n_d})
    config = env.config

    is_strombom   = mode in ("strombom", "strombom_llm")
    is_llm        = mode in ("strombom_llm", "stringnet_llm")

    planner: LLMPlanner | StrombomLLMPlanner | None = None
    oracle: OracularPlanner | StrombomOracle | None = None

    if mode == "stringnet_llm":
        planner = LLMPlanner(adapter_dim=config.get("adapter_dim", 64),
                             qwen_model=qwen_model, seed=seed or 0)
        oracle  = OracularPlanner()
        print(f"[StringNet LLM] Qwen available: {planner.qwen.available}")

    elif mode == "strombom_llm":
        planner = StrombomLLMPlanner(adapter_dim=config.get("adapter_dim", 64),
                                     qwen_model=qwen_model, seed=seed or 0)
        oracle  = StrombomOracle()
        print(f"[Strömbom LLM] Qwen available: {planner.qwen.available}")

    fd = FailureDetector(
        margin_thresh=config.get("containment_margin_thresh", 0.0),
        err_thresh=config.get("formation_error_thresh", 0.4),
        t_fail=config.get("T_fail", 8),
    )

    viz = HerdingVisualizer(mode=mode)
    T_max       = config.get("T_max", 1000)
    max_adapter = config.get("max_adapter_updates_per_episode", 3)
    adapter_updates = 0
    done  = False
    step  = 0
    info: dict = {}

    intent: dict = {"intent_token": "widen_net", "phase": "seek",
                    "params": {}, "source": "rule"}
    metrics: dict = {"escape_prob_est": 0.0, "formation_error": 0.0,
                     "containment_margin": 0.0, "failure": False}
    strombom_targets: np.ndarray | None = None
    strombom_phase = "drive"

    collect_radius   = _BASE_COLLECT_RADIUS
    d_collect        = _BASE_D_COLLECT
    d_behind         = _BASE_D_BEHIND
    formation_radius = _BASE_FORMATION_RADIUS
    eff_speed        = speed_scale
    flank_bias       = 0.0

    while not done and step < T_max:
        tokens        = feature_extractor(state)
        current_phase = state.get("phase", "seek")

        if is_strombom:
            if mode == "strombom_llm" and planner is not None:
                intent = planner.plan(tokens, current_phase=strombom_phase)
                p = intent.get("params", {})
                collect_radius   = _BASE_COLLECT_RADIUS   * float(p.get("collect_radius_scale",   1.0))
                d_collect        = _BASE_D_COLLECT        * float(p.get("collect_radius_scale",   1.0))
                d_behind         = _BASE_D_BEHIND         * float(p.get("drive_offset_scale",     1.0))
                formation_radius = _BASE_FORMATION_RADIUS * float(p.get("formation_radius_scale", 1.0))
                eff_speed        = speed_scale             * float(p.get("speed_scale",            1.0))
                flank_bias       = float(p.get("flank_bias", 0.0))
            else:
                esc  = tokens["escape_prob_est"]
                sprd = tokens["sheep_spread"]
                if esc > 0.5:
                    collect_radius, d_collect, d_behind = _BASE_COLLECT_RADIUS*0.75, _BASE_D_COLLECT*1.1, _BASE_D_BEHIND*0.85
                    formation_radius = _BASE_FORMATION_RADIUS * 0.85
                    tok = "tighten_collect"
                elif sprd > 0.10:
                    collect_radius, d_collect, d_behind = _BASE_COLLECT_RADIUS*1.2, _BASE_D_COLLECT*0.9, _BASE_D_BEHIND
                    formation_radius = _BASE_FORMATION_RADIUS * 1.2
                    tok = "spread_formation"
                else:
                    collect_radius, d_collect, d_behind = _BASE_COLLECT_RADIUS, _BASE_D_COLLECT, _BASE_D_BEHIND
                    formation_radius = _BASE_FORMATION_RADIUS
                    tok = "push_harder"
                intent = {"intent_token": tok, "phase": strombom_phase,
                          "params": {}, "source": "rule"}

            np_sheep = np.asarray(state["sheep_pos"])
            np_dogs  = np.asarray(state["dog_pos"])
            goal_pos = np.asarray(state["goal"])
            strombom_targets, strombom_phase = compute_strombom_targets(
                np_sheep, np_dogs, goal_pos,
                collect_radius=float(np.clip(collect_radius, 0.8, 6.0)),
                d_collect=float(np.clip(d_collect, 0.5, 4.0)),
                d_behind=float(np.clip(d_behind, 1.0, 7.0)),
                formation_radius=float(np.clip(formation_radius, 1.0, 5.0)),
                n_collectors=max(1, n_d // 3),
                flank_bias=flank_bias,
            )
            intent["phase"] = strombom_phase
            acts: dict[int, np.ndarray] = {}
            for j in range(n_d):
                acts[j] = strombom_action(j, strombom_targets, state, config,
                                          speed_scale=float(np.clip(eff_speed, 0.5, 2.5)))
            xi_des = strombom_targets

        else:
            if mode == "stringnet_llm" and planner is not None:
                intent = planner.plan(tokens, current_phase=current_phase)
            else:
                esc  = tokens["escape_prob_est"]
                sprd = tokens["sheep_spread"]
                tok  = ("tighten_net" if esc > 0.45 else
                        "focus_largest_cluster" if sprd > 0.08 else "widen_net")
                intent = {
                    "intent_token": tok, "phase": current_phase,
                    "params": {"radius_scale": 0.9 if tok == "tighten_net" else 1.0},
                    "source": "rule",
                }

            rs        = float(intent.get("params", {}).get("radius_scale", 1.0))
            formation = _desired_formation_stringnet(state, radius_scale=rs,
                                                     d_behind=config.get("d_behind", 1.2))
            xi_des    = _desired_positions_stringnet(formation, n_d)
            phase_key = intent.get("phase", current_phase)
            i_spd     = float(intent.get("params", {}).get("speed_scale", 1.0))
            eff_speed = float(np.clip(speed_scale * i_spd, 0.5, 2.5))
            acts = {}
            for j in range(n_d):
                if phase_key == "seek":
                    acts[j] = apply_seeking_controller(j, formation, state, config)
                elif phase_key == "enclose":
                    acts[j] = apply_enclosing_controller(j, formation, state, config)
                else:
                    acts[j] = apply_herding_controller(j, formation, state, config)
                acts[j] = acts[j] * eff_speed

        state, _, done, info = env.step(acts)
        metrics = fd.step(state, xi_des, tokens)

        if debug_every > 0 and step % debug_every == 0:
            acom = np.round(np.mean(state["sheep_pos"], axis=0), 3)
            print(
                f"[step={step:4d}] mode={mode} phase={intent.get('phase','?')} "
                f"intent={intent['intent_token']} esc={metrics['escape_prob_est']:.3f} "
                f"in_goal={info.get('in_goal',0)}/{n_a} acom={acom}"
            )

        if is_llm and planner is not None and oracle is not None:
            if metrics["failure"] and adapter_updates < max_adapter:
                corr = oracle.corrective_intent(tokens)
                planner.logged_update(tokens, corr,
                                      lr=config.get("adapter_lr", 5e-4),
                                      epochs=config.get("adapter_epochs", 3))
                adapter_updates += 1

        step += 1
        if step % render_every == 0:
            viz.update(state, metrics, intent, step,
                       done=done, success=info.get("success", False),
                       strombom_targets=strombom_targets if is_strombom else None)
            plt.pause(0.001)

    print(f"\n{'='*52}")
    print(f"Mode: {mode.upper()}  |  Steps: {step}  |  Success: {info.get('success', False)}")
    print(f"Sheep in goal: {info.get('in_goal', 0)}/{n_a}")
    if is_llm:
        print(f"Adapter updates: {adapter_updates}")
    print(f"{'='*52}")

    input("\nPress Enter to close...")
    viz.close()
    return {
        "mode": mode, "success": info.get("success", False),
        "steps": step, "in_goal": info.get("in_goal", 0),
        "adapter_updates": adapter_updates,
    }


def run_comparison(
    n_a: int = 5,
    n_d: int = 3,
    seed: int = 42,
    episodes: int = 5,
    max_steps: int = 600,
    qwen_model: str = "Qwen/Qwen3-0.6B",
    config_path: str | None = None,
) -> None:
    """Run all 4 modes headlessly for `episodes` episodes and show side-by-side comparison."""
    import csv, time
    from pathlib import Path as P
    from shepherd_env.sensors import feature_extractor as fe
    from metrics.failure_detector import FailureDetector as FD

    cfg_path = config_path or str(Path(__file__).parent / "configs" / "default.yaml")
    modes = ["strombom", "strombom_llm", "stringnet", "stringnet_llm"]
    results: dict[str, list] = {m: [] for m in modes}

    for mode in modes:
        print(f"\n{'='*50}\nRunning comparison: {mode} ({episodes} episodes)\n{'='*50}")
        is_strombom = mode in ("strombom", "strombom_llm")
        is_llm      = mode in ("strombom_llm", "stringnet_llm")

        for ep in range(episodes):
            env = ShepherdEnv(config_path=cfg_path)
            state = env.reset(seed=seed + ep, config={"N_a": n_a, "N_d": n_d})
            config = env.config

            planner = None; oracle = None
            if mode == "stringnet_llm":
                planner = LLMPlanner(adapter_dim=64, qwen_model=qwen_model, seed=seed + ep)
                oracle  = OracularPlanner()
            elif mode == "strombom_llm":
                planner = StrombomLLMPlanner(adapter_dim=64, qwen_model=qwen_model, seed=seed + ep)
                oracle  = StrombomOracle()

            fd = FD(
                margin_thresh=config.get("containment_margin_thresh", 0.0),
                err_thresh=config.get("formation_error_thresh", 0.4),
                t_fail=config.get("T_fail", 8),
            )

            collect_radius   = _BASE_COLLECT_RADIUS
            d_collect        = _BASE_D_COLLECT
            d_behind         = _BASE_D_BEHIND
            formation_radius = _BASE_FORMATION_RADIUS
            eff_speed = 1.2; flank_bias = 0.0
            strombom_phase = "drive"
            done = False; step = 0; info = {}
            adapter_updates = 0
            max_adapter = config.get("max_adapter_updates_per_episode", 3)

            while not done and step < max_steps:
                tokens = fe(state)
                current_phase = state.get("phase", "seek")

                if is_strombom:
                    if planner is not None:
                        intent = planner.plan(tokens, current_phase=strombom_phase)
                        p = intent.get("params", {})
                        collect_radius   = _BASE_COLLECT_RADIUS   * float(p.get("collect_radius_scale", 1.0))
                        d_collect        = _BASE_D_COLLECT        * float(p.get("collect_radius_scale", 1.0))
                        d_behind         = _BASE_D_BEHIND         * float(p.get("drive_offset_scale",   1.0))
                        formation_radius = _BASE_FORMATION_RADIUS * float(p.get("formation_radius_scale",1.0))
                        eff_speed        = 1.2 * float(p.get("speed_scale", 1.0))
                        flank_bias       = float(p.get("flank_bias", 0.0))
                    np_sheep = np.asarray(state["sheep_pos"])
                    np_dogs  = np.asarray(state["dog_pos"])
                    goal_pos = np.asarray(state["goal"])
                    targets, strombom_phase = compute_strombom_targets(
                        np_sheep, np_dogs, goal_pos,
                        collect_radius=float(np.clip(collect_radius, 0.8, 6.0)),
                        d_collect=float(np.clip(d_collect, 0.5, 4.0)),
                        d_behind=float(np.clip(d_behind, 1.0, 7.0)),
                        formation_radius=float(np.clip(formation_radius, 1.0, 5.0)),
                        n_collectors=max(1, n_d // 3),
                        flank_bias=flank_bias,
                    )
                    acts = {j: strombom_action(j, targets, state, config,
                                               speed_scale=float(np.clip(eff_speed, 0.5, 2.5)))
                            for j in range(n_d)}
                    xi_des = targets
                else:
                    if planner is not None:
                        intent = planner.plan(tokens, current_phase=current_phase)
                    else:
                        esc = tokens["escape_prob_est"]
                        sprd = tokens["sheep_spread"]
                        tok = ("tighten_net" if esc > 0.45 else
                               "focus_largest_cluster" if sprd > 0.08 else "widen_net")
                        intent = {"intent_token": tok, "phase": current_phase,
                                  "params": {"radius_scale": 0.9 if tok=="tighten_net" else 1.0},
                                  "source": "rule"}
                    rs = float(intent.get("params", {}).get("radius_scale", 1.0))
                    formation = _desired_formation_stringnet(state, radius_scale=rs)
                    xi_des = _desired_positions_stringnet(formation, n_d)
                    phase_key = intent.get("phase", current_phase)
                    i_spd = float(intent.get("params", {}).get("speed_scale", 1.0))
                    sp = float(np.clip(1.2 * i_spd, 0.5, 2.5))
                    acts = {}
                    for j in range(n_d):
                        if phase_key == "seek":
                            acts[j] = apply_seeking_controller(j, formation, state, config) * sp
                        elif phase_key == "enclose":
                            acts[j] = apply_enclosing_controller(j, formation, state, config) * sp
                        else:
                            acts[j] = apply_herding_controller(j, formation, state, config) * sp

                state, _, done, info = env.step(acts)
                m = fd.step(state, xi_des, tokens)
                if is_llm and planner is not None and oracle is not None:
                    if m["failure"] and adapter_updates < max_adapter:
                        planner.logged_update(tokens, oracle.corrective_intent(tokens),
                                              lr=config.get("adapter_lr", 5e-4),
                                              epochs=config.get("adapter_epochs", 3))
                        adapter_updates += 1
                step += 1

            results[mode].append({
                "success": info.get("success", False),
                "steps": step,
                "in_goal": info.get("in_goal", 0),
            })
            print(f"  ep={ep} steps={step} success={info.get('success',False)} "
                  f"in_goal={info.get('in_goal',0)}/{n_a}")

    _plot_comparison(results, n_a, n_d, episodes)


def _plot_comparison(results: dict, n_a: int, n_d: int, episodes: int) -> None:
    """Render and show a 3-panel bar chart comparing all 4 modes."""
    modes  = list(results.keys())
    labels = [m.replace("_", "\n") for m in modes]
    colors = ["#2980b9", "#8e44ad", "#27ae60", "#9b59b6"]

    sr   = [float(np.mean([r["success"] for r in results[m]])) for m in modes]
    avg_steps = [float(np.mean([r["steps"]   for r in results[m]])) for m in modes]
    avg_goal  = [float(np.mean([r["in_goal"] for r in results[m]])) for m in modes]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#1a1a2e")
    fig.suptitle(
        f"Comparison: {episodes} episodes  |  Sheep={n_a}  Dogs={n_d}",
        color="white", fontsize=13, fontweight="bold",
    )

    bar_cfg = [
        (axes[0], sr,        "Success Rate",       "cornflowerblue", (0, 1.1)),
        (axes[1], avg_steps, "Avg Steps to Finish","salmon",         None),
        (axes[2], avg_goal,  "Avg Sheep in Goal",  "mediumseagreen", (0, n_a + 0.5)),
    ]
    xs = np.arange(len(modes))
    for ax, vals, title, color, ylim in bar_cfg:
        ax.set_facecolor("#0d0d1a")
        bars = ax.bar(xs, vals, color=colors, edgecolor="#444466", linewidth=0.8, width=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, color="#ccccdd", fontsize=8)
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.tick_params(colors="#888899", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")
        if ylim:
            ax.set_ylim(*ylim)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", color="white", fontsize=8)

    plt.tight_layout()
    out_path = Path("comparison_results.png")
    plt.savefig(out_path, dpi=120, facecolor="#1a1a2e")
    print(f"\nComparison chart saved: {out_path.resolve()}")
    plt.show(block=True)


def main() -> None:
    """Parse CLI args or show interactive selector, then run simulation or comparison."""
    parser = argparse.ArgumentParser(description="Herding Simulation — Strömbom vs StringNet × LLM")
    parser.add_argument("--mode",
                        choices=["strombom", "strombom_llm", "stringnet", "stringnet_llm", "select"],
                        default="select")
    parser.add_argument("--compare", action="store_true",
                        help="Run all 4 modes headlessly and show comparison chart")
    parser.add_argument("--episodes",    type=int,   default=5,    help="Episodes per mode for --compare")
    parser.add_argument("--n-sheep",     type=int,   default=5)
    parser.add_argument("--n-dogs",      type=int,   default=3)
    parser.add_argument("--seed",        type=int,   default=None)
    parser.add_argument("--qwen-model",  default="Qwen/Qwen3-0.6B")
    parser.add_argument("--render-every",type=int,   default=1)
    parser.add_argument("--debug-every", type=int,   default=25)
    parser.add_argument("--speed-scale", type=float, default=1.2)
    parser.add_argument("--max-steps",   type=int,   default=600,
                        help="Max steps per episode for --compare")
    args = parser.parse_args()

    if args.compare:
        run_comparison(
            n_a=args.n_sheep, n_d=args.n_dogs,
            seed=args.seed or 42, episodes=args.episodes,
            max_steps=args.max_steps, qwen_model=args.qwen_model,
        )
        return

    if args.mode == "select":
        try:
            mode, n_a, n_d, qwen_model = select_mode_interactive()
        except Exception as exc:
            print(f"Interactive selector failed ({exc}); defaulting to stringnet.")
            mode, n_a, n_d, qwen_model = "stringnet", args.n_sheep, args.n_dogs, args.qwen_model
    else:
        mode, n_a, n_d, qwen_model = args.mode, args.n_sheep, args.n_dogs, args.qwen_model

    if mode is None:
        return

    run_simulation(
        mode=mode, n_a=n_a, n_d=n_d, seed=args.seed,
        qwen_model=qwen_model, render_every=args.render_every,
        debug_every=args.debug_every, speed_scale=args.speed_scale,
    )


if __name__ == "__main__":
    main()