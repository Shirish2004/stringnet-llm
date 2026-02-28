"""inference.py — Test trained StringNet models.

Loads a checkpoint and runs one or more evaluation episodes with:
  - Configurable n_dogs and n_sheep (any combination 1-5 dogs, 3+ sheep)
  - Dog collision avoidance baked in
  - Dense reward breakdown printed per step (optional)
  - GIF/montage saved to --output-dir
  - Per-step metrics CSV
  - Console summary with param trajectory (when ParamControlNet is used)

Works with all three adapter types:
  DualHeadAdapter   (default, trained by main_rl.py --mode stringnet_llm)
  ParamControlNet   (trained by main_rl.py --mode stringnet_param)
Usage:
  python inference.py --ckpt checkpoints/adapter_final.pt --n-sheep 8 --n-dogs 4
  python inference.py --ckpt checkpoints/adapter_final.pt --mode stringnet_param
  python inference.py --ckpt checkpoints/adapter_final.pt --episodes 5 --seed 100
  python inference.py --no-ckpt --mode stringnet --n-sheep 10 --n-dogs 5  # rule baseline
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from shepherd_env.env import ShepherdEnv
from shepherd_env.sensors import SceneHistory, feature_extractor
from shepherd_env.strombom_controller import compute_strombom_targets, strombom_action
from shepherd_env.controllers import (
    apply_seeking_controller,
    apply_enclosing_controller,
    apply_herding_controller,
)
from planner.train import (
    SCENE_DIM_HIST,
    DEFAULT_VOCAB, STROMBOM_VOCAB,
    DualHeadAdapter, ParamControlNet, CombinedNet,
)
from planner.llm import LLMPlanner, StrombomLLMPlanner
from planner.rl import apply_collision_avoidance, dense_reward

_BASE_COLLECT_RADIUS   = 2.5
_BASE_D_COLLECT        = 1.5
_BASE_D_BEHIND         = 3.0
_BASE_FORMATION_RADIUS = 2.5

PHASE_COLORS = {
    "seek":    "#f39c12",
    "enclose": "#e74c3c",
    "herd":    "#2ecc71",
    "drive":   "#9b59b6",
}


# ── action functions ──────────────────────────────────────────────────────────

def _sn_action(intent, state, config, n_d):
    p   = intent.get("params", {})
    rs  = float(p.get("radius_scale", 1.0))
    db  = float(p.get("d_behind",     _BASE_D_BEHIND))
    ss  = float(p.get("speed_scale",  1.2))
    arc = float(p.get("arc_span",     np.pi * 0.9))
    ph  = intent.get("phase", "seek")

    sheep  = np.asarray(state["sheep_pos"])
    goal   = np.asarray(state["goal"])
    acom   = sheep.mean(axis=0)
    u      = goal - acom
    u      = u / max(float(np.linalg.norm(u)), 1e-9)
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


def _rule_intent(tokens, n_d):
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
            "speed_scale":  1.3 if esc > 0.45 else 1.1,
            "arc_span":     np.pi * 0.9,
            "flank_bias":   0.0,
        },
        "source": "rule",
    }


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(
    mode:    str,
    ckpt:    str | None,
    qwen:    str,
    seed:    int,
):
    """Load planner for the given mode and optionally restore checkpoint."""
    import torch

    if mode == "stringnet":
        return None   # pure rule-based

    if mode == "strombom_llm":
        pl = StrombomLLMPlanner(adapter_dim=64, qwen_model=qwen, seed=seed,
                                use_hist_features=True)
    else:
        pl = LLMPlanner(adapter_dim=64, qwen_model=qwen, seed=seed,
                        use_hist_features=True)

    if mode == "stringnet_param":
        param_net = ParamControlNet(in_dim=SCENE_DIM_HIST, hidden=64)
        param_net.to(pl.device).eval()
        pl.adapter = param_net
        print(f"[inference] ParamControlNet swapped in — "
              f"MLP directly controls radius/d_behind/speed/arc/flank")

    if ckpt:
        p = Path(ckpt)
        if p.exists():
            data = torch.load(p, map_location=pl.device)
            # support both plain state_dict and wrapped {"adapter": ...} checkpoints
            if isinstance(data, dict) and "adapter" in data:
                sd = data["adapter"]
                ep = data.get("episode", "?")
                st = data.get("curriculum_stage", "?")
                print(f"[inference] Checkpoint ep={ep}  "
                      f"curriculum_stage={st}")
            else:
                sd = data
            try:
                pl.adapter.load_state_dict(sd)
                pl.adapter.eval()
                print(f"[inference] Weights loaded from {p.name}")
            except RuntimeError as e:
                print(f"[inference] ⚠ Architecture mismatch: {e}")
                print("[inference] Running with random weights.")
        else:
            print(f"[inference] ⚠ Checkpoint not found: {ckpt}")
            print("[inference] Running with random weights.")
    else:
        print("[inference] No checkpoint — using random / untrained weights.")

    return pl


# ── rendering ─────────────────────────────────────────────────────────────────

def render_frame(
    state:    dict,
    step:     int,
    intent:   dict,
    reward:   float,
    breakdown: dict,
    n_a:      int,
    in_goal:  int,
) -> np.ndarray:
    ph   = intent.get("phase", "seek")
    tok  = intent.get("intent_token", "?")
    src  = intent.get("source", "?")
    p    = intent.get("params", {})

    fig = plt.figure(figsize=(16, 5), facecolor="#1a1a2e")
    gs  = fig.add_gridspec(1, 4, wspace=0.35)
    ax  = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])

    fig.suptitle(
        f"Step {step}  |  phase={ph}  intent={tok}  src={src}  "
        f"reward={reward:+.4f}  in_goal={in_goal}/{n_a}",
        color="white", fontsize=9, fontweight="bold",
    )

    ax.set_facecolor("#0d0d1a")
    ax.set_xlim(-0.5, 20.5); ax.set_ylim(-0.5, 20.5)
    ax.set_aspect("equal")

    gr = np.asarray(state.get("goal_region", [[14, 8], [20, 12]]))
    rect = patches.FancyBboxPatch(
        (gr[0, 0], gr[0, 1]), gr[1, 0] - gr[0, 0], gr[1, 1] - gr[0, 1],
        boxstyle="round,pad=0.15", lw=1.5, edgecolor="#2ecc71",
        facecolor="#1a3a1a", alpha=0.6,
    )
    ax.add_patch(rect)
    ax.text(float(np.mean(gr[:, 0])), float(np.mean(gr[:, 1])),
            "GOAL", ha="center", va="center",
            color="#2ecc71", fontsize=8, fontweight="bold")

    sheep  = np.asarray(state["sheep_pos"])
    dogs   = np.asarray(state["dog_pos"])
    d_vel  = np.asarray(state["dog_vel"])

    ax.scatter(sheep[:, 0], sheep[:, 1], s=60, c="#3498db",
               edgecolors="white", linewidths=0.4, zorder=5)
    dog_colors = ["#e74c3c", "#f39c12", "#2ecc71", "#9b59b6", "#1abc9c"]
    for j, (d, v) in enumerate(zip(dogs, d_vel)):
        c = dog_colors[j % len(dog_colors)]
        ax.scatter(d[0], d[1], s=120, c=c, marker="^",
                   edgecolors="white", linewidths=0.6, zorder=6)
        ax.text(d[0] + 0.2, d[1] + 0.2, f"D{j}", color=c, fontsize=6)
        spd = float(np.linalg.norm(v))
        if spd > 0.05:
            tip = d + 0.7 * v / spd
            ax.annotate("", xy=tip, xytext=d,
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.0))
    # StringNet net
    for j in range(len(dogs)):
        nxt = (j + 1) % len(dogs)
        ax.plot([dogs[j, 0], dogs[nxt, 0]], [dogs[j, 1], dogs[nxt, 1]],
                lw=0.8, color="#aaaacc", alpha=0.35, linestyle="--")

    def _fmt_param(value: object) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    ax.set_title(
        f"rs={_fmt_param(p.get('radius_scale','?'))}  "
        f"d_behind={_fmt_param(p.get('d_behind','?'))}  "
        f"spd={_fmt_param(p.get('speed_scale','?'))}  "
        f"arc={_fmt_param(p.get('arc_span','?'))}",
        color=PHASE_COLORS.get(ph, "white"), fontsize=7,
    )
    ax.tick_params(colors="#888899", labelsize=6)

    # reward breakdown bar
    ax2.set_facecolor("#0d0d1a")
    bd_keys = [k for k in breakdown if k != "total"]
    bd_vals = [breakdown[k] for k in bd_keys]
    cols    = ["#2ecc71" if v >= 0 else "#e74c3c" for v in bd_vals]
    ax2.barh(bd_keys, bd_vals, color=cols)
    ax2.axvline(0, color="#888899", lw=0.7)
    ax2.set_title("Reward breakdown", color="white", fontsize=8)
    ax2.tick_params(colors="#888899", labelsize=6)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#444466")
    for i, v in enumerate(bd_vals):
        ax2.text(v + (0.002 if v >= 0 else -0.002), i, f"{v:.3f}",
                 va="center", ha="left" if v >= 0 else "right",
                 color="white", fontsize=5)

    # param values bar
    ax3.set_facecolor("#0d0d1a")
    pnames = ["radius_scale", "d_behind", "speed_scale", "arc_span", "flank_bias"]
    pvals  = [float(p.get(k, 0.0)) for k in pnames]
    ax3.barh(pnames, pvals, color="#2980b9")
    ax3.set_title("MLP params", color="white", fontsize=8)
    ax3.tick_params(colors="#888899", labelsize=6)
    for sp in ax3.spines.values():
        sp.set_edgecolor("#444466")
    for i, v in enumerate(pvals):
        ax3.text(max(v, 0) + 0.02, i, f"{v:.3f}",
                 va="center", color="white", fontsize=5)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf  = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img  = buf[:, :, :3].copy()
    plt.close(fig)
    return img


# ── main inference loop ───────────────────────────────────────────────────────

def run_inference(
    mode:          str,
    ckpt:          str | None,
    n_a:           int,
    n_d:           int,
    episodes:      int,
    max_steps:     int,
    seed:          int,
    qwen:          str,
    output_dir:    str,
    capture_every: int,
    debug_every:   int,
    verbose_reward: bool,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    planner = load_model(mode, ckpt, qwen, seed)
    env     = ShepherdEnv()
    config  = env.config
    if planner is not None:
        planner.set_config(config)

    is_strombom = mode == "strombom_llm"
    action_fn   = _strombom_action if is_strombom else _sn_action

    all_rows:   list[dict] = []
    ep_summaries: list[dict] = []

    for ep in range(episodes):
        print(f"\n{'='*56}")
        print(f"Episode {ep+1}/{episodes}  |  mode={mode}  "
              f"n_sheep={n_a}  n_dogs={n_d}  seed={seed+ep}")
        print(f"{'='*56}")

        state      = env.reset(seed=seed + ep, config={"N_a": n_a, "N_d": n_d})
        sh         = SceneHistory()
        prev_state = deepcopy(state)
        frames:    list[np.ndarray] = []
        ep_reward  = 0.0
        n_coll     = 0
        done       = False
        step       = 0
        info: dict = {}

        # per-episode param trajectory (for ParamControlNet analysis)
        param_traj: list[dict] = []

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
            ep_reward += r
            if abs(breakdown["collision_pen"]) > 0.01:
                n_coll += 1

            p = intent.get("params", {})
            param_traj.append({
                "step":         step,
                "radius_scale": round(float(p.get("radius_scale", 1.0)), 4),
                "d_behind":     round(float(p.get("d_behind",     3.0)), 4),
                "speed_scale":  round(float(p.get("speed_scale",  1.2)), 4),
                "arc_span":     round(float(p.get("arc_span", 2.83)),    4),
                "flank_bias":   round(float(p.get("flank_bias",   0.0)), 4),
                "intent_token": intent.get("intent_token", "?"),
                "reward":       round(r, 5),
            })

            # CSV row
            all_rows.append({
                "episode": ep, "step": step,
                "mode":    mode,
                **param_traj[-1],
                **{f"rew_{k}": v for k, v in breakdown.items()},
                "in_goal": info.get("in_goal", 0),
                "success": info.get("success", False),
            })

            if debug_every > 0 and step % debug_every == 0:
                acom = np.round(np.mean(np.asarray(state["sheep_pos"]), axis=0), 3)
                print(
                    f"  [step={step:4d}]  intent={intent.get('intent_token','?'):24s}  "
                    f"esc={tokens['escape_prob_est']:.3f}  "
                    f"reward={r:+.4f}  in_goal={info.get('in_goal',0)}/{n_a}  "
                    f"acom={acom}"
                )
                if verbose_reward:
                    for k, v in breakdown.items():
                        print(f"             {k:20s}: {v:+.4f}")
                if mode == "stringnet_param":
                    print(f"             params: "
                          + "  ".join(f"{k}={v:.3f}" for k, v in p.items()
                                      if k in ("radius_scale","d_behind","speed_scale",
                                               "arc_span","flank_bias")))

            if step % capture_every == 0:
                frames.append(render_frame(
                    state, step, intent, r, breakdown, n_a,
                    info.get("in_goal", 0),
                ))

            step += 1
            if done:
                break

        mean_r = ep_reward / max(step, 1)
        ep_sum = {
            "episode":      ep,
            "success":      info.get("success", False),
            "steps":        step,
            "in_goal":      info.get("in_goal", 0),
            "in_goal_frac": round(info.get("in_goal", 0) / max(n_a, 1), 4),
            "mean_reward":  round(mean_r, 5),
            "total_reward": round(ep_reward, 4),
            "coll_steps":   n_coll,
        }
        ep_summaries.append(ep_sum)
        print(f"\n  ✔ success={ep_sum['success']}  "
              f"steps={ep_sum['steps']}  in_goal={ep_sum['in_goal']}/{n_a}  "
              f"mean_reward={ep_sum['mean_reward']:+.5f}  coll_steps={n_coll}")

        # save param trajectory CSV for ParamControlNet analysis
        if mode == "stringnet_param" and param_traj:
            ptraj_path = out / f"param_traj_ep{ep}_seed{seed+ep}.csv"
            with open(ptraj_path, "w", newline="", encoding="utf-8") as f:
                dw = csv.DictWriter(f, fieldnames=list(param_traj[0].keys()))
                dw.writeheader(); dw.writerows(param_traj)

        # save GIF
        if frames:
            try:
                import imageio
                gif = str(out / f"inference_{mode}_ep{ep}_seed{seed+ep}.gif")
                imageio.mimsave(gif, frames, fps=8)
                print(f"  GIF: {gif}")
            except ImportError:
                _save_montage(frames, out, f"{mode}_ep{ep}")

    # aggregate CSV
    if all_rows:
        csv_path = out / f"inference_{mode}_na{n_a}_nd{n_d}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            dw = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            dw.writeheader(); dw.writerows(all_rows)
        print(f"\nCSV: {csv_path}")

    # final summary
    print(f"\n{'='*56}")
    print(f" SUMMARY  mode={mode}  n_sheep={n_a}  n_dogs={n_d}")
    print(f"{'='*56}")
    sr   = float(np.mean([e["success"] for e in ep_summaries]))
    mr   = float(np.mean([e["mean_reward"] for e in ep_summaries]))
    mg   = float(np.mean([e["in_goal"]     for e in ep_summaries]))
    ms   = float(np.mean([e["steps"]       for e in ep_summaries]))
    mc   = float(np.mean([e["coll_steps"]  for e in ep_summaries]))
    print(f"  Success rate:      {sr:.2f}")
    print(f"  Avg sheep in goal: {mg:.2f} / {n_a}")
    print(f"  Avg steps:         {ms:.1f}")
    print(f"  Avg mean reward:   {mr:+.5f}")
    print(f"  Avg coll steps:    {mc:.1f}")
    print(f"{'='*56}")


def _save_montage(frames: list, out: Path, tag: str) -> None:
    n    = len(frames)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    h, w = frames[0].shape[:2]
    canvas = np.full((rows * h, cols * w, 3), 26, dtype=np.uint8)
    for i, img in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    fig, ax = plt.subplots(figsize=(cols * 4, rows * 2.5), facecolor="#1a1a2e")
    ax.imshow(canvas); ax.axis("off")
    plt.tight_layout(pad=0)
    path = out / f"inference_{tag}_montage.png"
    fig.savefig(path, dpi=100, facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  Montage: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference / evaluation for trained StringNet adapters"
    )
    parser.add_argument(
        "--mode",
        choices=["stringnet", "stringnet_llm", "stringnet_param", "strombom_llm"],
        default="stringnet_llm",
    )
    parser.add_argument("--ckpt",          default=None,
                        help=".pt checkpoint file (omit for untrained baseline)")
    parser.add_argument("--no-ckpt",       action="store_true",
                        help="Force rule-based inference, ignore --mode")
    parser.add_argument("--n-sheep",       type=int, default=6)
    parser.add_argument("--n-dogs",        type=int, default=3)
    parser.add_argument("--episodes",      type=int, default=3)
    parser.add_argument("--max-steps",     type=int, default=600)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--qwen-model",    default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir",    default="inference_out")
    parser.add_argument("--capture-every", type=int, default=10,
                        help="Capture a frame every N steps")
    parser.add_argument("--debug-every",   type=int, default=50,
                        help="Print step info every N steps (0=off)")
    parser.add_argument("--verbose-reward", action="store_true",
                        help="Print reward breakdown at every debug step")
    args = parser.parse_args()

    mode = "stringnet" if args.no_ckpt else args.mode
    ckpt = None       if args.no_ckpt else args.ckpt

    print(f"\n{'='*56}")
    print(f"  StringNet Inference")
    print(f"  mode={mode}  n_sheep={args.n_sheep}  n_dogs={args.n_dogs}")
    print(f"  episodes={args.episodes}  max_steps={args.max_steps}")
    print(f"  ckpt={ckpt or 'None (random/rule)'}")
    print(f"{'='*56}\n")

    run_inference(
        mode           = mode,
        ckpt           = ckpt,
        n_a            = args.n_sheep,
        n_d            = args.n_dogs,
        episodes       = args.episodes,
        max_steps      = args.max_steps,
        seed           = args.seed,
        qwen           = args.qwen_model,
        output_dir     = args.output_dir,
        capture_every  = args.capture_every,
        debug_every    = args.debug_every,
        verbose_reward = args.verbose_reward,
    )


if __name__ == "__main__":
    main()