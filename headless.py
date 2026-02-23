from __future__ import annotations

import sys
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shepherd_env.env import ShepherdEnv
from shepherd_env.controllers import (
    apply_seeking_controller,
    apply_enclosing_controller,
    apply_herding_controller,
)
from shepherd_env.sensors import feature_extractor
from planner.llm_planner import LLMPlanner
from planner.mock_llm import OracularPlanner
from metrics.failure_detector import FailureDetector

PHASE_COLORS = {"seek": "#f39c12", "enclose": "#e74c3c", "herd": "#2ecc71"}
SHEEP_COLOR = "#3498db"
DOG_COLOR = "#e74c3c"


def _desired_formation(state: dict, rs: float = 1.0, d_behind: float = 1.2) -> dict:
    """Compute StringNet formation from current state and radius scale."""
    acom = state["sheep_pos"].mean(axis=0)
    goal = state["goal"]
    u = goal - acom
    u = u / max(np.linalg.norm(u), 1e-9)
    return {"center": acom - d_behind * u, "phi": float(np.arctan2(u[1], u[0])), "radius": 1.4 * rs}


def _desired_positions(formation: dict, n_d: int) -> np.ndarray:
    """Return n_d desired positions on the StringNet semi-circle."""
    phi, rad, c = formation["phi"], formation["radius"], formation["center"]
    return np.vstack([
        c + rad * np.array([
            np.cos(phi + np.pi / 2 + np.pi * j / max(1, n_d - 1)),
            np.sin(phi + np.pi / 2 + np.pi * j / max(1, n_d - 1)),
        ]) for j in range(n_d)
    ])


def render_frame(state: dict, step: int, mode: str, intent: dict, metrics: dict, n_d: int) -> np.ndarray:
    """Render one simulation frame; returns H x W x 3 uint8 array."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#1a1a2e",
                              gridspec_kw={"width_ratios": [3, 1, 1]})
    src = intent.get("source", "rule")
    mode_label = "StringNet + Qwen3 + Adapter" if mode == "llm" else "StringNet (Rule-Based)"
    fig.suptitle(f"{mode_label}  |  Step {step}  |  Source: {src}",
                 color="white", fontsize=11, fontweight="bold")

    ax = axes[0]
    ax.set_facecolor("#0d0d1a")
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-0.5, 20.5)
    ax.set_aspect("equal")

    goal_rect = patches.FancyBboxPatch((12, 8), 8, 4, boxstyle="round,pad=0.1",
                                        linewidth=1.5, edgecolor="#2ecc71",
                                        facecolor="#1a3a1a", alpha=0.6)
    ax.add_patch(goal_rect)
    ax.text(16, 10, "GOAL", ha="center", va="center",
            color="#2ecc71", fontsize=9, fontweight="bold")

    sheep_pos = np.asarray(state["sheep_pos"])
    dog_pos = np.asarray(state["dog_pos"])
    dog_vel = np.asarray(state["dog_vel"])

    ax.scatter(sheep_pos[:, 0], sheep_pos[:, 1], s=80, c=SHEEP_COLOR,
               marker="o", edgecolors="white", linewidths=0.5, label="Sheep", zorder=5)
    ax.scatter(dog_pos[:, 0], dog_pos[:, 1], s=130, c=DOG_COLOR,
               marker="^", edgecolors="white", linewidths=0.8, label="Dogs", zorder=6)

    for j in range(n_d):
        nxt = (j + 1) % n_d
        ax.plot([dog_pos[j, 0], dog_pos[nxt, 0]], [dog_pos[j, 1], dog_pos[nxt, 1]],
                lw=1.5, color=DOG_COLOR, alpha=0.7, linestyle="--", zorder=4)
        speed = np.linalg.norm(dog_vel[j])
        if speed > 0.05:
            tip = dog_pos[j] + 0.6 * dog_vel[j] / speed
            ax.annotate("", xy=tip, xytext=dog_pos[j],
                        arrowprops=dict(arrowstyle="->", color=DOG_COLOR, lw=1.2))

    phase = state.get("phase", "seek")
    tok = intent.get("intent_token", "?")
    ax.set_title(f"Phase: {phase}  |  Intent: {tok}",
                 color=PHASE_COLORS.get(phase, "white"), fontsize=9)
    ax.legend(loc="lower right", facecolor="#1a1a2e", edgecolor="#444466",
              labelcolor="white", fontsize=7)
    ax.tick_params(colors="#888899", labelsize=7)

    for i, (name, val, color) in enumerate([
        ("Escape Prob", metrics.get("escape_prob_est", 0), "#e74c3c"),
        ("Formation Err", metrics.get("formation_error", 0), "#f39c12"),
    ]):
        axes[i + 1].set_facecolor("#0d0d1a")
        axes[i + 1].barh([name], [val], color=color)
        axes[i + 1].set_xlim(0, max(val * 1.5, 1.0))
        axes[i + 1].set_title(f"{name}: {val:.3f}", color="white", fontsize=8)
        axes[i + 1].tick_params(colors="#888899", labelsize=6)
        for spine in axes[i + 1].spines.values():
            spine.set_edgecolor("#444466")

    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = buf[:, :, :3].copy()
    plt.close(fig)
    return img


def _save_montage(frames: list, out: Path, mode: str) -> None:
    """Save a grid of captured frames as one PNG montage."""
    n = len(frames)
    cols = min(4, n)
    rows_count = (n + cols - 1) // cols
    h, w = frames[0].shape[:2]
    canvas = np.full((rows_count * h, cols * w, 3), 26, dtype=np.uint8)
    for i, img in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    fig, ax = plt.subplots(figsize=(cols * 4, rows_count * 2.5), facecolor="#1a1a2e")
    ax.imshow(canvas)
    ax.axis("off")
    fig.tight_layout(pad=0)
    path = out / f"herding_{mode}_montage.png"
    fig.savefig(path, dpi=100, facecolor="#1a1a2e")
    plt.close(fig)
    print(f"Montage saved: {path}")


def run_headless(
    mode: str = "llm",
    n_a: int = 5,
    n_d: int = 3,
    seed: int = 42,
    max_steps: int = 300,
    capture_every: int = 10,
    output_dir: str = "/mnt/user-data/outputs",
    qwen_model: str = "Qwen/Qwen3-0.6B",
) -> dict:
    """Run one headless episode; saves GIF/montage and metrics CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config_path = str(Path(__file__).parent / "configs" / "default.yaml")
    env = ShepherdEnv(config_path=config_path)
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

    frames: list[np.ndarray] = []
    rows: list[dict] = []
    adapter_updates = 0
    done = False
    intent = {"intent_token": "widen_net", "phase": "seek", "params": {}, "source": "rule"}
    metrics: dict = {"escape_prob_est": 0.0, "formation_error": 0.0, "containment_margin": 0.0}
    info: dict = {}

    for step in range(1, max_steps + 1):
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
        formation = _desired_formation(state, rs=rs, d_behind=config.get("d_behind", 1.2))
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
            max_upd = config.get("max_adapter_updates_per_episode", 3)
            if metrics["failure"] and adapter_updates < max_upd:
                corr = oracle.corrective_intent(tokens)
                planner.logged_update(tokens, corr,
                                      lr=config.get("adapter_lr", 5e-4),
                                      epochs=config.get("adapter_epochs", 3))
                adapter_updates += 1

        rows.append({
            "step": step,
            "intent_token": intent["intent_token"],
            "source": intent.get("source", "rule"),
            "escape_prob": round(metrics["escape_prob_est"], 4),
            "formation_error": round(metrics["formation_error"], 4),
            "containment_margin": round(metrics["containment_margin"], 4),
            "phase": current_phase,
            "in_goal": info.get("in_goal", 0),
        })

        if step % capture_every == 0:
            img = render_frame(state, step, mode, intent, metrics, n_d)
            frames.append(img)
            print(f"  step={step:4d} | phase={current_phase:7s} | "
                  f"intent={intent['intent_token']:22s} | "
                  f"esc={metrics['escape_prob_est']:.2f} | "
                  f"in_goal={info.get('in_goal', 0)}/{n_a} | "
                  f"src={intent.get('source', '?')}")

        if done:
            break

    csv_path = out / f"metrics_{mode}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV saved: {csv_path}")

    if frames:
        try:
            import imageio
            gif_path = str(out / f"herding_{mode}.gif")
            imageio.mimsave(gif_path, frames, fps=10)
            print(f"GIF saved: {gif_path}")
        except ImportError:
            _save_montage(frames, out, mode)

    return {
        "success": info.get("success", False),
        "steps": step,
        "adapter_updates": adapter_updates,
        "in_goal": info.get("in_goal", 0),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="StringNet headless runner")
    parser.add_argument("--mode", choices=["stringnet", "llm"], default="llm")
    parser.add_argument("--n-sheep", type=int, default=5)
    parser.add_argument("--n-dogs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--capture-every", type=int, default=10)
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-0.6B",
                        help="HuggingFace model ID, e.g. Qwen/Qwen3-0.6B or Qwen/Qwen3-1.7B")
    args = parser.parse_args()

    print(f"\nRunning headless | mode={args.mode} | sheep={args.n_sheep} | dogs={args.n_dogs}")
    print(f"Qwen model: {args.qwen_model}")
    print("=" * 60)
    result = run_headless(
        mode=args.mode,
        n_a=args.n_sheep,
        n_d=args.n_dogs,
        seed=args.seed,
        max_steps=args.steps,
        capture_every=args.capture_every,
        qwen_model=args.qwen_model,
    )
    print(f"\nFinal result: {result}")