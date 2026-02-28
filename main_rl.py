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
from shepherd_env.sensors import feature_extractor, SceneHistory
from shepherd_env.safety import project_planner_params
from planner.mock_llm import OracularPlanner, StrombomOracle
from planner.llm import LLMPlanner, StrombomLLMPlanner
from metrics.failure_detector import FailureDetector
from visualizer import HerdingVisualizer


_BASE_COLLECT_RADIUS   = 2.5
_BASE_D_COLLECT        = 1.5
_BASE_D_BEHIND         = 3.0
_BASE_FORMATION_RADIUS = 2.5
_BEAM_EVERY            = 50


def _desired_formation_stringnet(
    state: dict,
    radius_scale: float = 1.0,
    d_behind: float = 1.2,
    d_behind_delta: float = 0.0,
) -> dict:
    """Compute semi-circular StringNet formation parameters from state.

    Improvement 1,8: accepts continuous d_behind_delta from DualHeadAdapter output.
    """
    acom = state["sheep_pos"].mean(axis=0)
    goal = state["goal"]
    u    = goal - acom
    u    = u / max(float(np.linalg.norm(u)), 1e-9)
    eff_behind = float(np.clip(d_behind + d_behind_delta, 0.5, 6.0))
    return {
        "center": acom - eff_behind * u,
        "phi":    float(np.arctan2(u[1], u[0])),
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


def _role_based_action(
    j: int,
    role: str,
    formation: dict,
    state: dict,
    config: dict,
    speed_scale: float,
    phase_key: str,
) -> np.ndarray:
    """Improvement 6 — Apply role-specific controller for dog j.

    Roles: 'leader' (front of formation, drives herd), 'flanker_left/right'
    (arc wings), 'collector' (enclose phase retrieval). Falls back to
    standard phase controller when role is unrecognised.
    """
    if role == "collector" or phase_key == "enclose":
        a = apply_enclosing_controller(j, formation, state, config)
    elif role == "leader" and phase_key == "herd":
        a = apply_herding_controller(j, formation, state, config)
    elif phase_key == "seek":
        a = apply_seeking_controller(j, formation, state, config)
    elif phase_key == "herd":
        a = apply_herding_controller(j, formation, state, config)
    else:
        a = apply_seeking_controller(j, formation, state, config)
    return a * speed_scale


def _make_strombom_action_fn(config: dict, n_d: int):
    """Return a closure used by beam_plan rollout evaluation for Strömbom."""
    def action_fn(intent: dict, state: dict, cfg: dict, n_dogs: int):
        p  = intent.get("params", {})
        cr = _BASE_COLLECT_RADIUS   * float(p.get("collect_radius_scale",   1.0))
        dc = _BASE_D_COLLECT        * float(p.get("collect_radius_scale",   1.0))
        db = _BASE_D_BEHIND         * float(p.get("drive_offset_scale",     1.0))
        fr = _BASE_FORMATION_RADIUS * float(p.get("formation_radius_scale", 1.0))
        ss = 1.2                    * float(p.get("speed_scale",            1.0))
        fb = float(p.get("flank_bias", 0.0))
        np_sheep = np.asarray(state["sheep_pos"])
        np_dogs  = np.asarray(state["dog_pos"])
        goal_pos = np.asarray(state["goal"])
        targets, _ = compute_strombom_targets(
            np_sheep, np_dogs, goal_pos,
            collect_radius=float(np.clip(cr, 0.8, 6.0)),
            d_collect=float(np.clip(dc, 0.5, 4.0)),
            d_behind=float(np.clip(db, 1.0, 7.0)),
            formation_radius=float(np.clip(fr, 1.0, 5.0)),
            n_collectors=max(1, n_dogs // 3),
            flank_bias=fb,
        )
        acts = {j: strombom_action(j, targets, state, cfg,
                                   speed_scale=float(np.clip(ss, 0.5, 2.5)))
                for j in range(n_dogs)}
        return acts, targets
    return action_fn


def _make_stringnet_action_fn(config: dict, n_d: int):
    """Return a closure used by beam_plan rollout evaluation for StringNet."""
    def action_fn(intent: dict, state: dict, cfg: dict, n_dogs: int):
        p   = intent.get("params", {})
        rs  = float(p.get("radius_scale", 1.0))
        db  = float(p.get("d_behind_delta", 0.0))
        ss  = float(np.clip(1.2 * float(p.get("speed_scale", 1.0)), 0.5, 2.5))
        ph  = intent.get("phase", "seek")
        formation = _desired_formation_stringnet(
            state, radius_scale=rs,
            d_behind=cfg.get("d_behind", 1.2),
            d_behind_delta=db,
        )
        xi_des = _desired_positions_stringnet(formation, n_dogs)
        acts   = {}
        for j in range(n_dogs):
            if ph == "seek":
                acts[j] = apply_seeking_controller(j, formation, state, cfg) * ss
            elif ph == "enclose":
                acts[j] = apply_enclosing_controller(j, formation, state, cfg) * ss
            else:
                acts[j] = apply_herding_controller(j, formation, state, cfg) * ss
        return acts, xi_des
    return action_fn


def select_mode_interactive() -> tuple[str, int, int, str]:
    """Matplotlib 4-mode selector; returns (mode, n_sheep, n_dogs, qwen_model)."""
    sel = {"mode": None, "n_a": 5, "n_d": 3, "qwen_model": "Qwen/Qwen3-0.6B"}
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.canvas.manager.set_window_title("Herding Simulation — Mode Selection")

    ax.text(0.5, 0.93, "Herding Simulation", ha="center", fontsize=18,
            fontweight="bold", color="white")
    ax.text(0.5, 0.84, "Strömbom vs StringNet  ·  With & Without Qwen3 + PyTorch Adapter",
            ha="center", fontsize=9, color="#8888aa")

    btns_def = [
        ([0.04, 0.60, 0.43, 0.14], "strombom",      "Strömbom\n(Rule-Based)",        "#1a3a5c", "#2980b9"),
        ([0.53, 0.60, 0.43, 0.14], "strombom_llm",  "Strömbom + Qwen3\n+ Adapter",   "#3d1a5c", "#8e44ad"),
        ([0.04, 0.42, 0.43, 0.14], "stringnet",     "StringNet\n(Rule-Based)",        "#1a5c34", "#27ae60"),
        ([0.53, 0.42, 0.43, 0.14], "stringnet_llm", "StringNet + Qwen3\n+ Adapter",  "#4a1870", "#9b59b6"),
    ]
    mode_buttons = []
    for rect, key, label, color, hover in btns_def:
        axb = fig.add_axes(rect)
        b   = Button(axb, label, color=color, hovercolor=hover)
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
    ax.text(0.155, 0.36, "Agents",     ha="center", fontsize=7, color="#888899")
    ax.text(0.625, 0.36, "Qwen model", ha="center", fontsize=7, color="#888899")

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

    btn_na.on_clicked(on_na); btn_nd.on_clicked(on_nd)
    btn_q06.on_clicked(on_q06); btn_q17.on_clicked(on_q17)
    btn_go.on_clicked(on_go)

    plt.show(block=True)
    return sel["mode"] or "stringnet", sel["n_a"], sel["n_d"], sel["qwen_model"]


def run_simulation(
    mode:         str,
    n_a:          int   = 5,
    n_d:          int   = 3,
    seed:         int | None = None,
    qwen_model:   str   = "Qwen/Qwen3-0.6B",
    render_every: int   = 1,
    debug_every:  int   = 25,
    speed_scale:  float = 1.2,
    config_path:  str | None = None,
    use_beam:     bool  = True,
    use_hist:     bool  = True,
) -> dict:
    """Run one live episode with real-time visualisation for any of the 4 modes.

    Improvements integrated:
    1  DualHeadAdapter: continuous params extracted from intent["cont_params"].
    2  beam_plan: every BEAM_EVERY steps for LLM modes when use_beam=True.
    3  Reward passed to logged_update; fd.reset_episode() on done.
    4  Confidence check logged; low-confidence flagged in debug output.
    5  SceneHistory used when use_hist=True; 20-dim features for LLM planners.
    6  hierarchical_plan provides role assignments mapped to _role_based_action.
    7  peft_config passed through (None by default; enable via config YAML).
    8  project_planner_params applied before controllers; safety.py imported.
    """
    cfg_path = config_path or str(Path(__file__).parent / "configs" / "default.yaml")
    env   = ShepherdEnv(config_path=cfg_path)
    state = env.reset(seed=seed, config={"N_a": n_a, "N_d": n_d})
    config = env.config

    is_strombom = mode in ("strombom", "strombom_llm")
    is_llm      = mode in ("strombom_llm", "stringnet_llm")

    planner: LLMPlanner | StrombomLLMPlanner | None = None
    oracle:  OracularPlanner | StrombomOracle | None = None
    scene_hist: SceneHistory | None = None

    peft_config = config.get("peft_config", None)

    if mode == "stringnet_llm":
        planner = LLMPlanner(
            adapter_dim=config.get("adapter_dim", 64),
            qwen_model=qwen_model,
            seed=seed or 0,
            use_hist_features=use_hist,
            peft_config=peft_config,
        )
        oracle     = OracularPlanner()
        planner.set_config(config)
        if use_hist:
            scene_hist = SceneHistory()
        print(f"[StringNet LLM] Qwen available: {planner.qwen.available}")

    elif mode == "strombom_llm":
        planner = StrombomLLMPlanner(
            adapter_dim=config.get("adapter_dim", 64),
            qwen_model=qwen_model,
            seed=seed or 0,
            use_hist_features=use_hist,
        )
        oracle     = StrombomOracle()
        planner.set_config(config)
        if use_hist:
            scene_hist = SceneHistory()
        print(f"[Strömbom LLM] Qwen available: {planner.qwen.available}")

    fd = FailureDetector(
        margin_thresh=config.get("containment_margin_thresh", 0.0),
        err_thresh=config.get("formation_error_thresh", 0.4),
        t_fail=config.get("T_fail", 8),
    )

    strombom_action_fn = _make_strombom_action_fn(config, n_d)
    stringnet_action_fn = _make_stringnet_action_fn(config, n_d)

    viz          = HerdingVisualizer(mode=mode)
    T_max        = config.get("T_max", 1000)
    max_adapter  = config.get("max_adapter_updates_per_episode", 3)
    adapter_updates = 0
    done  = False
    step  = 0
    info: dict = {}

    intent: dict = {"intent_token": "widen_net", "phase": "seek",
                    "params": {}, "source": "rule"}
    metrics: dict = {"escape_prob_est": 0.0, "formation_error": 0.0,
                     "containment_margin": 0.0, "failure": False, "reward": 0.0}
    strombom_targets: np.ndarray | None = None
    strombom_phase = "drive"

    collect_radius   = _BASE_COLLECT_RADIUS
    d_collect        = _BASE_D_COLLECT
    d_behind         = _BASE_D_BEHIND
    formation_radius = _BASE_FORMATION_RADIUS
    eff_speed        = speed_scale
    flank_bias       = 0.0

    while not done and step < T_max:
        if scene_hist is not None:
            tokens = scene_hist.feature_extractor_hist(state)
        else:
            tokens = feature_extractor(state)
        current_phase = state.get("phase", "seek")

        if is_strombom:
            if mode == "strombom_llm" and planner is not None:
                if use_beam and step % _BEAM_EVERY == 0 and step > 0:
                    intent = planner.beam_plan(
                        tokens, strombom_phase, env, state,
                        strombom_action_fn, FailureDetector,
                        n_candidates=config.get("beam_candidates", 4),
                        rollout_steps=config.get("beam_rollout_steps", 40),
                    ) if hasattr(planner, "beam_plan") else planner.plan(tokens, current_phase=strombom_phase)
                else:
                    intent = planner.plan(tokens, current_phase=strombom_phase)
                p = intent.get("params", {})
                p, _ = project_planner_params(p, config)
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
                    collect_radius, d_collect, d_behind = (_BASE_COLLECT_RADIUS * 0.75,
                                                           _BASE_D_COLLECT * 1.1,
                                                           _BASE_D_BEHIND  * 0.85)
                    formation_radius = _BASE_FORMATION_RADIUS * 0.85
                    tok = "tighten_collect"
                elif sprd > 0.10:
                    collect_radius, d_collect, d_behind = (_BASE_COLLECT_RADIUS * 1.2,
                                                           _BASE_D_COLLECT * 0.9,
                                                           _BASE_D_BEHIND)
                    formation_radius = _BASE_FORMATION_RADIUS * 1.2
                    tok = "spread_formation"
                else:
                    collect_radius, d_collect, d_behind = (_BASE_COLLECT_RADIUS,
                                                           _BASE_D_COLLECT,
                                                           _BASE_D_BEHIND)
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
                if use_beam and step % _BEAM_EVERY == 0 and step > 0:
                    intent = planner.beam_plan(
                        tokens, current_phase, env, state,
                        stringnet_action_fn, FailureDetector,
                        n_candidates=config.get("beam_candidates", 4),
                        rollout_steps=config.get("beam_rollout_steps", 40),
                    ) if hasattr(planner, "beam_plan") else planner.plan(tokens, current_phase=current_phase)
                else:
                    intent = planner.hierarchical_plan(tokens, current_phase, n_d)
                p, _ = project_planner_params(intent.get("params", {}), config)
                intent["params"] = p
            else:
                esc  = tokens["escape_prob_est"]
                sprd = tokens["sheep_spread"]
                tok  = ("tighten_net" if esc > 0.45 else
                        "focus_largest_cluster" if sprd > 0.08 else "widen_net")
                intent = {
                    "intent_token": tok, "phase": current_phase,
                    "params": {"radius_scale": 0.9 if tok == "tighten_net" else 1.0},
                    "assignments": {j: "leader" if j == 0 else "flanker_right" if j % 2 else "flanker_left"
                                    for j in range(n_d)},
                    "source": "rule",
                }

            rs        = float(intent.get("params", {}).get("radius_scale", 1.0))
            db_delta  = float(intent.get("params", {}).get("d_behind_delta", 0.0))
            formation = _desired_formation_stringnet(
                state, radius_scale=rs,
                d_behind=config.get("d_behind", 1.2),
                d_behind_delta=db_delta,
            )
            xi_des    = _desired_positions_stringnet(formation, n_d)
            phase_key = intent.get("phase", current_phase)
            i_spd     = float(intent.get("params", {}).get("speed_scale", 1.0))
            eff_speed = float(np.clip(speed_scale * i_spd, 0.5, 2.5))
            assignments = intent.get("assignments", {})
            acts = {}
            for j in range(n_d):
                role = assignments.get(j, "flanker_right")
                acts[j] = _role_based_action(j, role, formation, state, config,
                                              eff_speed, phase_key)

        state, _, done, info = env.step(acts)
        metrics = fd.step(state, xi_des, tokens)

        if debug_every > 0 and step % debug_every == 0:
            acom = np.round(np.mean(state["sheep_pos"], axis=0), 3)
            conf_str = ""
            if is_llm and planner is not None:
                conf = planner.confidence(tokens)
                conf_str = f" conf={conf:.2f}"
            print(
                f"[step={step:4d}] mode={mode} phase={intent.get('phase','?')} "
                f"intent={intent['intent_token']} esc={metrics['escape_prob_est']:.3f} "
                f"reward={metrics.get('reward', 0.0):+.3f} "
                f"in_goal={info.get('in_goal', 0)}/{n_a} acom={acom}{conf_str}"
            )

        if is_llm and planner is not None and oracle is not None:
            if metrics["failure"] and adapter_updates < max_adapter:
                corr = oracle.corrective_intent(tokens)
                planner.logged_update(
                    tokens, corr,
                    lr=config.get("adapter_lr", 5e-4),
                    epochs=config.get("adapter_epochs", 3),
                    reward=metrics.get("reward", 0.0),
                )
                adapter_updates += 1

        step += 1
        if step % render_every == 0:
            viz.update(state, metrics, intent, step,
                       done=done, success=info.get("success", False),
                       strombom_targets=strombom_targets if is_strombom else None)
            plt.pause(0.001)

    if done:
        fd.reset_episode()

    print(f"\n{'='*56}")
    print(f"Mode: {mode.upper()}  |  Steps: {step}  |  Success: {info.get('success', False)}")
    print(f"Sheep in goal: {info.get('in_goal', 0)}/{n_a}")
    print(f"Episode return: {fd.episode_return:.3f}  |  Mean reward: {fd.mean_reward():.3f}")
    if is_llm:
        print(f"Adapter updates: {adapter_updates}")
    print(f"{'='*56}")

    input("\nPress Enter to close...")
    viz.close()
    return {
        "mode": mode,
        "success": info.get("success", False),
        "steps": step,
        "in_goal": info.get("in_goal", 0),
        "adapter_updates": adapter_updates,
        "episode_return": fd.episode_return,
    }


def run_comparison(
    n_a:          int  = 5,
    n_d:          int  = 3,
    seed:         int  = 42,
    episodes:     int  = 5,
    max_steps:    int  = 600,
    qwen_model:   str  = "Qwen/Qwen3-0.6B",
    config_path:  str | None = None,
    use_beam:     bool = False,
    use_hist:     bool = True,
) -> None:
    """Run all 4 modes headlessly for `episodes` episodes and show side-by-side comparison.

    Improvements integrated:
    3  reward tracked per episode, mean_reward recorded.
    4  confidence logged per step for LLM modes.
    5  SceneHistory used when use_hist=True.
    6  hierarchical_plan used in StringNet LLM.
    8  project_planner_params applied before controllers.
    """
    from shepherd_env.sensors import feature_extractor as fe

    cfg_path = config_path or str(Path(__file__).parent / "configs" / "default.yaml")
    modes    = ["strombom", "strombom_llm", "stringnet", "stringnet_llm"]
    results: dict[str, list] = {m: [] for m in modes}

    for mode in modes:
        print(f"\n{'='*52}\nRunning comparison: {mode} ({episodes} episodes)\n{'='*52}")
        is_strombom = mode in ("strombom", "strombom_llm")
        is_llm      = mode in ("strombom_llm", "stringnet_llm")

        for ep in range(episodes):
            env   = ShepherdEnv(config_path=cfg_path)
            state = env.reset(seed=seed + ep, config={"N_a": n_a, "N_d": n_d})
            config = env.config

            planner    = None
            oracle     = None
            scene_hist = None

            if mode == "stringnet_llm":
                planner = LLMPlanner(adapter_dim=64, qwen_model=qwen_model,
                                     seed=seed + ep, use_hist_features=use_hist)
                oracle  = OracularPlanner()
                planner.set_config(config)
                if use_hist:
                    scene_hist = SceneHistory()
            elif mode == "strombom_llm":
                planner = StrombomLLMPlanner(adapter_dim=64, qwen_model=qwen_model,
                                             seed=seed + ep, use_hist_features=use_hist)
                oracle  = StrombomOracle()
                planner.set_config(config)
                if use_hist:
                    scene_hist = SceneHistory()

            fd = FailureDetector(
                margin_thresh=config.get("containment_margin_thresh", 0.0),
                err_thresh=config.get("formation_error_thresh", 0.4),
                t_fail=config.get("T_fail", 8),
            )
            fd.reset_episode()

            collect_radius   = _BASE_COLLECT_RADIUS
            d_collect        = _BASE_D_COLLECT
            d_behind         = _BASE_D_BEHIND
            formation_radius = _BASE_FORMATION_RADIUS
            eff_speed        = 1.2
            flank_bias       = 0.0
            strombom_phase   = "drive"
            intent: dict     = {"intent_token": "push_harder", "phase": "drive",
                                 "params": {}, "source": "rule"}
            done = False; step = 0; info = {}
            adapter_updates = 0
            max_adapter = config.get("max_adapter_updates_per_episode", 3)

            while not done and step < max_steps:
                tokens        = scene_hist.feature_extractor_hist(state) if scene_hist else fe(state)
                current_phase = state.get("phase", "seek")

                if is_strombom:
                    if planner is not None:
                        intent = planner.plan(tokens, current_phase=strombom_phase)
                        p, _   = project_planner_params(intent.get("params", {}), config)
                        collect_radius   = _BASE_COLLECT_RADIUS   * float(p.get("collect_radius_scale",   1.0))
                        d_collect        = _BASE_D_COLLECT        * float(p.get("collect_radius_scale",   1.0))
                        d_behind         = _BASE_D_BEHIND         * float(p.get("drive_offset_scale",     1.0))
                        formation_radius = _BASE_FORMATION_RADIUS * float(p.get("formation_radius_scale", 1.0))
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
                    acts   = {j: strombom_action(j, targets, state, config,
                                                  speed_scale=float(np.clip(eff_speed, 0.5, 2.5)))
                              for j in range(n_d)}
                    xi_des = targets

                else:
                    if planner is not None:
                        intent = planner.hierarchical_plan(tokens, current_phase, n_d)
                        p, _   = project_planner_params(intent.get("params", {}), config)
                        intent["params"] = p
                    else:
                        esc  = tokens["escape_prob_est"]
                        sprd = tokens["sheep_spread"]
                        tok  = ("tighten_net" if esc > 0.45 else
                                "focus_largest_cluster" if sprd > 0.08 else "widen_net")
                        intent = {
                            "intent_token": tok, "phase": current_phase,
                            "params": {"radius_scale": 0.9 if tok == "tighten_net" else 1.0},
                            "assignments": {j: "leader" if j == 0 else "flanker_right"
                                            for j in range(n_d)},
                            "source": "rule",
                        }
                    rs       = float(intent.get("params", {}).get("radius_scale", 1.0))
                    db_delta = float(intent.get("params", {}).get("d_behind_delta", 0.0))
                    formation = _desired_formation_stringnet(
                        state, radius_scale=rs,
                        d_behind=config.get("d_behind", 1.2),
                        d_behind_delta=db_delta,
                    )
                    xi_des     = _desired_positions_stringnet(formation, n_d)
                    phase_key  = intent.get("phase", current_phase)
                    i_spd      = float(intent.get("params", {}).get("speed_scale", 1.0))
                    sp         = float(np.clip(1.2 * i_spd, 0.5, 2.5))
                    assignments = intent.get("assignments", {})
                    acts       = {}
                    for j in range(n_d):
                        role    = assignments.get(j, "flanker_right")
                        acts[j] = _role_based_action(j, role, formation, state, config, sp, phase_key)

                state, _, done, info = env.step(acts)
                m = fd.step(state, xi_des, tokens)

                if is_llm and planner is not None and oracle is not None:
                    if m["failure"] and adapter_updates < max_adapter:
                        planner.logged_update(
                            tokens,
                            oracle.corrective_intent(tokens),
                            lr=config.get("adapter_lr", 5e-4),
                            epochs=config.get("adapter_epochs", 3),
                            reward=m.get("reward", 0.0),
                        )
                        adapter_updates += 1
                step += 1

            fd.reset_episode()
            results[mode].append({
                "success":        info.get("success", False),
                "steps":          step,
                "in_goal":        info.get("in_goal", 0),
                "episode_return": fd.episode_return,
                "mean_reward":    fd.mean_reward(),
            })
            print(f"  ep={ep} steps={step} success={info.get('success', False)} "
                  f"in_goal={info.get('in_goal', 0)}/{n_a} "
                  f"mean_reward={fd.mean_reward():.3f}")

    _plot_comparison(results, n_a, n_d, episodes)


def _plot_comparison(results: dict, n_a: int, n_d: int, episodes: int) -> None:
    """Render and save a 4-panel bar chart comparing all 4 modes.

    Improvement 3: adds mean_reward panel to visualise RL signal quality.
    """
    modes  = list(results.keys())
    labels = [m.replace("_", "\n") for m in modes]
    colors = ["#2980b9", "#8e44ad", "#27ae60", "#9b59b6"]

    sr         = [float(np.mean([r["success"]      for r in results[m]])) for m in modes]
    avg_steps  = [float(np.mean([r["steps"]        for r in results[m]])) for m in modes]
    avg_goal   = [float(np.mean([r["in_goal"]      for r in results[m]])) for m in modes]
    avg_reward = [float(np.mean([r.get("mean_reward", 0.0) for r in results[m]])) for m in modes]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor="#1a1a2e")
    fig.suptitle(
        f"Comparison: {episodes} episodes  |  Sheep={n_a}  Dogs={n_d}",
        color="white", fontsize=13, fontweight="bold",
    )

    bar_cfg = [
        (axes[0], sr,         "Success Rate",        "cornflowerblue", (0, 1.1)),
        (axes[1], avg_steps,  "Avg Steps to Finish", "salmon",         None),
        (axes[2], avg_goal,   "Avg Sheep in Goal",   "mediumseagreen", (0, n_a + 0.5)),
        (axes[3], avg_reward, "Mean Step Reward",    "#f1c40f",        None),
    ]
    xs = np.arange(len(modes))
    for ax, vals, title, _, ylim in bar_cfg:
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
    parser = argparse.ArgumentParser(
        description="Herding Simulation — Strömbom vs StringNet × LLM"
    )
    parser.add_argument("--mode",
                        choices=["strombom", "strombom_llm", "stringnet", "stringnet_llm", "select"],
                        default="select")
    parser.add_argument("--compare",      action="store_true",
                        help="Run all 4 modes headlessly and show comparison chart")
    parser.add_argument("--episodes",     type=int,   default=5,
                        help="Episodes per mode for --compare")
    parser.add_argument("--n-sheep",      type=int,   default=5)
    parser.add_argument("--n-dogs",       type=int,   default=3)
    parser.add_argument("--seed",         type=int,   default=None)
    parser.add_argument("--qwen-model",   default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--render-every", type=int,   default=1)
    parser.add_argument("--debug-every",  type=int,   default=25)
    parser.add_argument("--speed-scale",  type=float, default=1.2)
    parser.add_argument("--max-steps",    type=int,   default=600,
                        help="Max steps per episode for --compare")
    parser.add_argument("--no-beam",      action="store_true",
                        help="Disable beam search rollout evaluation")
    parser.add_argument("--no-hist",      action="store_true",
                        help="Disable trajectory history features")
    args = parser.parse_args()

    use_beam = not args.no_beam
    use_hist = not args.no_hist

    if args.compare:
        run_comparison(
            n_a=args.n_sheep, n_d=args.n_dogs,
            seed=args.seed or 42, episodes=args.episodes,
            max_steps=args.max_steps, qwen_model=args.qwen_model,
            use_beam=use_beam, use_hist=use_hist,
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
        use_beam=use_beam, use_hist=use_hist,
    )


if __name__ == "__main__":
    main()