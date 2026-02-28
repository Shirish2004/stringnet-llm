"""compare.py — Headless evaluation of StringNet variants.

Compares three modes:
  stringnet       — rule-based StringNet, no MLP
  stringnet_llm   — StringNet + DualHeadAdapter (discrete intent + param deltas)
  stringnet_param — StringNet + ParamControlNet (MLP directly controls all params)
Output:
  - Per-episode CSV (compare_results_<tag>.csv)
  - Summary bar chart PNG (compare_chart_<tag>.png)
  - JSON summary (compare_summary_<tag>.json)

Usage:
  python compare.py --episodes 10 --n-sheep 6 --n-dogs 3
  python compare.py --all-modes --ckpt checkpoints/adapter_final.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shepherd_env.env import ShepherdEnv
from shepherd_env.sensors import feature_extractor, SceneHistory
from shepherd_env.strombom_controller import compute_strombom_targets, strombom_action
from shepherd_env.controllers import (
    apply_seeking_controller,
    apply_enclosing_controller,
    apply_herding_controller,
)
from planner.train import SCENE_DIM_HIST, DEFAULT_VOCAB, ParamControlNet, DualHeadAdapter
from planner.llm import LLMPlanner
from planner.rl import apply_collision_avoidance, dense_reward

_BASE_COLLECT_RADIUS   = 2.5
_BASE_D_COLLECT        = 1.5
_BASE_D_BEHIND         = 3.0
_BASE_FORMATION_RADIUS = 2.5


# ── action functions (same as main_rl) ───────────────────────────────────────

def _sn_action(intent, state, config, n_d):
    p   = intent.get("params", {})
    rs  = float(p.get("radius_scale", 1.0))
    db  = float(p.get("d_behind",     _BASE_D_BEHIND))
    ss  = float(p.get("speed_scale",  1.2))
    arc = float(p.get("arc_span",     np.pi * 0.9))
    ph  = intent.get("phase", "seek")

    sheep = np.asarray(state["sheep_pos"])
    goal  = np.asarray(state["goal"])
    acom  = sheep.mean(axis=0)
    u     = goal - acom
    u     = u / max(float(np.linalg.norm(u)), 1e-9)
    center = acom - db * u
    phi    = float(np.arctan2(u[1], u[0]))
    radius = rs * 1.4
    half   = arc / 2.0

    xi_des = np.vstack([
        center + radius * np.array([
            np.cos(phi + half - arc * j / max(n_d - 1, 1)),
            np.sin(phi + half - arc * j / max(n_d - 1, 1)),
        ])
        for j in range(n_d)
    ])
    formation = {"center": center, "phi": phi, "radius": radius}
    acts: dict[int, np.ndarray] = {}
    for j in range(n_d):
        if ph == "enclose":
            a = apply_enclosing_controller(j, formation, state, config)
        elif ph == "herd":
            a = apply_herding_controller(j, formation, state, config)
        else:
            a = apply_seeking_controller(j, formation, state, config)
        acts[j] = a * float(np.clip(ss, 0.5, 2.5))
    return acts, xi_des


def _strombom_action(intent, state, config, n_d):
    p  = intent.get("params", {})
    cr = _BASE_COLLECT_RADIUS   * float(p.get("collect_radius_scale",   1.0))
    dc = _BASE_D_COLLECT        * float(p.get("collect_radius_scale",   1.0))
    db = _BASE_D_BEHIND         * float(p.get("drive_offset_scale",     1.0))
    fr = _BASE_FORMATION_RADIUS * float(p.get("formation_radius_scale", 1.0))
    ss = 1.2 * float(p.get("speed_scale", 1.0))
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
        n_collectors=max(1, n_d // 3),
        flank_bias=fb,
    )
    acts = {j: strombom_action(j, targets, state, config,
                               speed_scale=float(np.clip(ss, 0.5, 2.5)))
            for j in range(n_d)}
    return acts, targets


# ── rule baseline ─────────────────────────────────────────────────────────────

def _rule_intent(tokens: dict, n_d: int) -> dict:
    esc  = tokens["escape_prob_est"]
    sprd = tokens["sheep_spread"]
    tok  = ("tighten_net" if esc > 0.45 else
            "focus_largest_cluster" if sprd > 0.08 else "widen_net")
    return {
        "intent_token": tok,
        "phase":  "herd" if esc > 0.45 else "seek",
        "params": {
            "radius_scale": 0.85 if esc > 0.45 else 1.0,
            "d_behind":     _BASE_D_BEHIND,
            "speed_scale":  1.3  if esc > 0.45 else 1.1,
            "arc_span":     np.pi * 0.9,
            "flank_bias":   0.0,
        },
        "source": "rule",
    }


# ── build planners ────────────────────────────────────────────────────────────

def build_planner_for_mode(
    mode:    str,
    config:  dict,
    ckpt:    str | None,
    qwen:    str,
    seed:    int,
) -> LLMPlanner | None:
    """Return planner for the given mode, or None for pure rule-based."""
    import torch

    if mode == "stringnet":
        return None   # rule-based, no MLP

    # if mode == "strombom_llm":
    #     pl = StrombomLLMPlanner(adapter_dim=64, qwen_model=qwen, seed=seed,
    #                             use_hist_features=True)
    else:
        pl = LLMPlanner(adapter_dim=64, qwen_model=qwen, seed=seed,
                        use_hist_features=True)

    if mode == "stringnet_param":
        param_net = ParamControlNet(in_dim=SCENE_DIM_HIST, hidden=64)
        param_net.to(pl.device).eval()
        pl.adapter = param_net

    if ckpt:
        p = Path(ckpt)
        if p.exists():
            data = torch.load(p, map_location=pl.device)
            sd   = data.get("adapter", data)
            try:
                pl.adapter.load_state_dict(sd)
                print(f"  [{mode}] Loaded checkpoint {p.name}")
            except RuntimeError as e:
                print(f"  [{mode}] Checkpoint shape mismatch — fresh weights. ({e})")

    pl.set_config(config)
    return pl


# ── single episode runner ─────────────────────────────────────────────────────

def run_episode(
    mode:      str,
    planner,
    env,
    config:    dict,
    n_a:       int,
    n_d:       int,
    seed:      int,
    max_steps: int,
) -> dict:
    """Run one evaluation episode; return metrics dict."""
    state      = env.reset(seed=seed, config={"N_a": n_a, "N_d": n_d})
    sh         = SceneHistory()
    is_strombom = mode == "strombom_llm"
    action_fn  = _strombom_action if is_strombom else _sn_action

    total_r    = 0.0
    n_coll     = 0
    prev_state = deepcopy(state)
    done       = False
    step       = 0
    info:  dict = {}
    t_start = time.time()

    for _ in range(max_steps):
        tokens = sh.feature_extractor_hist(state)

        if planner is not None:
            intent = planner.plan(tokens)
        else:
            intent = _rule_intent(tokens, n_d)

        acts, xi_des = action_fn(intent, state, config, n_d)
        acts = apply_collision_avoidance(
            acts, np.asarray(state["dog_pos"]),
            ubar_d=float(config.get("ubar_d", 3.0)),
        )

        prev_state = deepcopy(state)
        state, _, done, info = env.step(acts)
        r, breakdown = dense_reward(state, prev_state, xi_des, tokens, config, n_a)
        total_r += r
        n_coll  += 1 if abs(breakdown["collision_pen"]) > 0.01 else 0
        step    += 1
        if done:
            break

    elapsed = time.time() - t_start
    return {
        "mode":           mode,
        "success":        bool(info.get("success", False)),
        "steps":          step,
        "in_goal":        info.get("in_goal", 0),
        "in_goal_frac":   round(info.get("in_goal", 0) / max(n_a, 1), 4),
        "mean_reward":    round(total_r / max(step, 1), 5),
        "total_reward":   round(total_r, 4),
        "collision_steps": n_coll,
        "elapsed_s":      round(elapsed, 2),
    }


# ── comparison ────────────────────────────────────────────────────────────────

def run_comparison(
    modes:      list[str],
    n_a:        int,
    n_d:        int,
    episodes:   int,
    seed_base:  int,
    max_steps:  int,
    ckpt:       str | None,
    qwen:       str,
    output_dir: str,
) -> None:
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag  = f"na{n_a}_nd{n_d}_ep{episodes}"

    env  = ShepherdEnv()
    cfg  = env.config

    results: dict[str, list[dict]] = {m: [] for m in modes}

    for mode in modes:
        print(f"\n{'─'*50}")
        print(f" Mode: {mode}  ({episodes} episodes, n_a={n_a}, n_d={n_d})")
        print(f"{'─'*50}")
        planner = build_planner_for_mode(mode, cfg, ckpt, qwen, seed_base)
        for ep in range(episodes):
            r = run_episode(mode, planner, env, cfg, n_a, n_d,
                            seed=seed_base + ep, max_steps=max_steps)
            results[mode].append(r)
            print(f"  ep={ep+1:3d}  success={r['success']}  "
                  f"steps={r['steps']:4d}  in_goal={r['in_goal']}/{n_a}  "
                  f"reward={r['mean_reward']:+.4f}  "
                  f"coll_steps={r['collision_steps']}")

    # save CSV
    all_rows = [r for rows in results.values() for r in rows]
    csv_path = out / f"compare_results_{tag}.csv"
    if all_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            dw = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            dw.writeheader(); dw.writerows(all_rows)
    print(f"\nCSV : {csv_path}")

    # summary JSON
    summary = {}
    for mode in modes:
        rs = results[mode]
        summary[mode] = {
            "success_rate":    round(float(np.mean([r["success"]      for r in rs])), 4),
            "mean_steps":      round(float(np.mean([r["steps"]        for r in rs])), 2),
            "mean_in_goal":    round(float(np.mean([r["in_goal"]      for r in rs])), 3),
            "mean_reward":     round(float(np.mean([r["mean_reward"]  for r in rs])), 5),
            "mean_collisions": round(float(np.mean([r["collision_steps"] for r in rs])), 2),
            "episodes":        len(rs),
        }
    json_path = out / f"compare_summary_{tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON: {json_path}")

    # print table
    print(f"\n{'─'*70}")
    print(f"{'Mode':<18} {'SuccRate':>8} {'Steps':>8} "
          f"{'InGoal':>8} {'Reward':>9} {'Collisions':>12}")
    print(f"{'─'*70}")
    for mode, s in summary.items():
        print(f"{mode:<18} {s['success_rate']:>8.2f} {s['mean_steps']:>8.1f} "
              f"{s['mean_in_goal']:>8.2f} {s['mean_reward']:>+9.4f} "
              f"{s['mean_collisions']:>12.1f}")
    print(f"{'─'*70}")

    _plot(summary, n_a, n_d, episodes, out, tag)


def _plot(summary: dict, n_a: int, n_d: int, episodes: int,
          out: Path, tag: str) -> None:
    modes  = list(summary.keys())
    colors = ["#27ae60", "#2980b9", "#e67e22", "#8e44ad",
              "#c0392b", "#1abc9c"][:len(modes)]
    labels = [m.replace("_", "\n") for m in modes]
    xs     = np.arange(len(modes))

    metrics = [
        ("success_rate",    "Success Rate",          (0, 1.1),         "cornflowerblue"),
        ("mean_steps",      "Avg Steps",              None,             "salmon"),
        ("mean_in_goal",    f"Avg Sheep in Goal (/{n_a})",
                                                      (0, n_a + 0.5),   "mediumseagreen"),
        ("mean_reward",     "Mean Step Reward",        None,             "#f1c40f"),
        ("mean_collisions", "Avg Collision Steps",     None,             "#e74c3c"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5),
                             facecolor="#1a1a2e")
    fig.suptitle(f"StringNet Comparison  |  episodes={episodes}  "
                 f"sheep={n_a}  dogs={n_d}",
                 color="white", fontsize=11, fontweight="bold")

    for ax, (key, title, ylim, _) in zip(axes, metrics):
        vals = [summary[m][key] for m in modes]
        ax.set_facecolor("#0d0d1a")
        bars = ax.bar(xs, vals, color=colors, edgecolor="#444466", linewidth=0.8, width=0.65)
        ax.set_xticks(xs); ax.set_xticklabels(labels, color="#ccccdd", fontsize=7)
        ax.set_title(title, color="white", fontsize=8, pad=5)
        ax.tick_params(colors="#888899", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")
        if ylim:
            ax.set_ylim(*ylim)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(abs(bar.get_height()) * 0.02, 0.01),
                    f"{v:.2f}", ha="center", va="bottom", color="white", fontsize=7)

    plt.tight_layout()
    path = out / f"compare_chart_{tag}.png"
    plt.savefig(path, dpi=130, facecolor="#1a1a2e")
    plt.close()
    print(f"Chart: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare StringNet variants (rule-based vs LLM+adapter vs ParamControlNet)"
    )
    parser.add_argument("--modes", nargs="+",
                        default=["stringnet", "stringnet_llm", "stringnet_param"],
                        help="Modes to compare. Add strombom_llm for full 4-way benchmark.")
    parser.add_argument("--all-modes", action="store_true",
                        help="Compare all 4 modes (stringnet, stringnet_llm, "
                             "stringnet_param, strombom_llm)")
    parser.add_argument("--episodes",   type=int,   default=10)
    parser.add_argument("--n-sheep",    type=int,   default=6)
    parser.add_argument("--n-dogs",     type=int,   default=3)
    parser.add_argument("--max-steps",  type=int,   default=600)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--ckpt",       default=None,
                        help="Checkpoint .pt for LLM/param modes")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="compare_results")
    args = parser.parse_args()

    modes = (["stringnet", "stringnet_llm", "stringnet_param", "strombom_llm"]
             if args.all_modes else args.modes)

    run_comparison(
        modes     = modes,
        n_a       = args.n_sheep,
        n_d       = args.n_dogs,
        episodes  = args.episodes,
        seed_base = args.seed,
        max_steps = args.max_steps,
        ckpt      = args.ckpt,
        qwen      = args.qwen_model,
        output_dir= args.output_dir,
    )


if __name__ == "__main__":
    main()