"""Unified entry point for StringNet herding: train, inference, headless, compare, simulate."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import matplotlib
if sys.platform != "linux":
    matplotlib.use("TkAgg")
else:
    _has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    matplotlib.use("TkAgg" if _has_display else "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from shepherd_env.env import ShepherdEnv
from shepherd_env.sensors import feature_extractor, SceneHistory
from shepherd_env.controllers import (
    apply_seeking_controller,
    apply_enclosing_controller,
    apply_herding_controller,
)
from shepherd_env.strombom_controller import compute_strombom_targets, strombom_action
from shepherd_env.safety import project_planner_params
from planner.llm import LLMPlanner, StrombomLLMPlanner
from planner.mock_llm import OracularPlanner, StrombomOracle, MockLLM, StrombomMockLLM
from planner.train import (
    AdapterMLP, DualHeadAdapter, ParamControlNet, CombinedNet,
    ReplayBuffer, train_adapter,
    scene_to_tensor_hist, intent_to_idx, oracle_cont_params,
    SCENE_DIM_HIST, DEFAULT_VOCAB, STROMBOM_VOCAB,
)
from metrics.failure_detector import FailureDetector
from planner.rl import apply_collision_avoidance, dense_reward
from visualizer import HerdingVisualizer
from headless import run_headless
from inference import (
    run_inference,
    load_model,
    render_frame as inference_render_frame,
    _sn_action,
    _strombom_action,
    _rule_intent,
    _save_montage,
)
from compare import (
    run_comparison as compare_run_comparison,
    build_planner_for_mode,
    run_episode,
)
from main_rl import (
    run_simulation,
    run_comparison as rl_run_comparison,
    select_mode_interactive,
    _desired_formation_stringnet,
    _desired_positions_stringnet,
    _role_based_action,
)

_BASE_D_BEHIND = 3.0


def save_video(frames: list, out_path: Path, fps: int = 10, no_video: bool = False) -> None:
    """Save frames as MP4, falling back to GIF then PNG montage."""
    if not frames or no_video:
        return

    try:
        import imageio
        mp4 = str(out_path.with_suffix(".mp4"))
        writer = imageio.get_writer(mp4, fps=fps, codec="libx264",
                                    output_params=["-pix_fmt", "yuv420p"])
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"  Video (MP4): {mp4}")
        return
    except Exception:
        pass

    try:
        import imageio
        gif = str(out_path.with_suffix(".gif"))
        imageio.mimsave(gif, frames, fps=fps)
        print(f"  Video (GIF): {gif}")
        return
    except Exception:
        pass

    _save_montage(frames, out_path.parent, out_path.stem)


def _gif_to_mp4(gif_path: Path) -> None:
    """Convert an existing GIF to MP4 in the same directory."""
    try:
        import imageio
        frames = list(imageio.mimread(str(gif_path)))
        mp4 = gif_path.with_suffix(".mp4")
        writer = imageio.get_writer(str(mp4), fps=10, codec="libx264",
                                    output_params=["-pix_fmt", "yuv420p"])
        for f in frames:
            writer.append_data(f[:, :, :3] if f.shape[-1] == 4 else f)
        writer.close()
        print(f"  MP4 (from GIF): {mp4}")
    except Exception as exc:
        print(f"  GIF→MP4 skipped: {exc}")


def cmd_train(args: argparse.Namespace) -> None:
    """Train the adapter or ParamControlNet with RL + oracle corrective supervision."""
    import torch

    out      = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir);  ckpt_dir.mkdir(parents=True, exist_ok=True)

    config_path = str(Path(__file__).parent / "configs" / "default.yaml")
    env    = ShepherdEnv(config_path=config_path)
    config = env.config

    if args.mode == "strombom_llm":
        planner = StrombomLLMPlanner(adapter_dim=config.get("adapter_dim", 64),
                                     qwen_model=args.qwen_model, seed=args.seed,
                                     use_hist_features=True)
        oracle  = StrombomOracle()
    else:
        planner = LLMPlanner(adapter_dim=config.get("adapter_dim", 64),
                             qwen_model=args.qwen_model, seed=args.seed,
                             use_hist_features=True)
        oracle  = OracularPlanner()

    if args.mode == "stringnet_param":
        param_net = ParamControlNet(in_dim=SCENE_DIM_HIST, hidden=64)
        param_net.to(planner.device).eval()
        planner.adapter = param_net

    planner.set_config(config)
    buffer = ReplayBuffer(capacity=2048)
    fd     = FailureDetector(
        margin_thresh=config.get("containment_margin_thresh", 0.0),
        err_thresh=config.get("formation_error_thresh", 0.4),
        t_fail=config.get("T_fail", 8),
    )

    is_strombom = args.mode == "strombom_llm"
    action_fn   = _strombom_action if is_strombom else _sn_action

    best_sr = 0.0
    ep_log: list[dict] = []

    print(f"\n[train] mode={args.mode}  episodes={args.episodes}  "
          f"n_sheep={args.n_sheep}  n_dogs={args.n_dogs}")
    print(f"[train] Qwen available: {planner.qwen.available}")

    for ep in range(1, args.episodes + 1):
        sh    = SceneHistory()
        state = env.reset(seed=args.seed + ep,
                          config={"N_a": args.n_sheep, "N_d": args.n_dogs})
        ep_r            = 0.0
        frames: list    = []
        adapter_updates = 0
        max_adapter     = config.get("max_adapter_updates_per_episode", 3)

        for step in range(1, args.max_steps + 1):
            tokens = sh.feature_extractor_hist(state)
            # intent = planner.plan(tokens)
            if args.mode == "stringnet_param":
                scene_tensor = scene_to_tensor_hist(tokens).to(planner.device)
                # intent = planner.adapter.decode(scene_tensor)
                params = planner.adapter.decode(scene_tensor)
                intent = {
                    "intent_token": "param_control",
                    "phase":        state.get("phase", "seek"),
                    "params":       params,
                    "source":       "adapter",
                }

                # intent.setdefault("phase", state.get("phase", "seek"))
                # intent.setdefault("source", "adapter")
            else:
                intent = planner.plan(tokens)

            acts, xi_des = action_fn(intent, state, config, args.n_dogs)
            acts = apply_collision_avoidance(
                acts, np.asarray(state["dog_pos"]),
                ubar_d=float(config.get("ubar_d", 3.0)),
            )

            prev_state = deepcopy(state)
            state, _, done, info = env.step(acts)
            r, breakdown = dense_reward(state, prev_state, xi_des, tokens, config, args.n_sheep)
            ep_r   += r
            metrics = fd.step(state, xi_des, tokens)

            # buffer.push(
            #     scene_to_tensor_hist(tokens),
            #     intent_to_idx(intent, DEFAULT_VOCAB),
            #     oracle_cont_params(intent),
            #     reward=float(r),
            # )
            xt = scene_to_tensor_hist(tokens).to(planner.device)
            if args.mode == "stringnet_param":
                y_cont = planner.adapter.oracle_targets(intent)
            else:
                y_cont = oracle_cont_params(intent)
            buffer.push(xt, intent_to_idx(intent, DEFAULT_VOCAB), y_cont, reward=float(r))

            # if metrics.get("failure", False) and adapter_updates < max_adapter:
            #     planner.logged_update(
            #         tokens, oracle.corrective_intent(tokens),
            #         lr=config.get("adapter_lr", 5e-4),
            #         epochs=config.get("adapter_epochs", 3),
            #         reward=r,
            #     )
            if metrics.get("failure", False) and adapter_updates < max_adapter:
                if args.mode == "stringnet_param":
                    # corr = oracle.corrective_intent(tokens)
                    # corr_cont = oracle_cont_params(corr)
                    # xt = scene_to_tensor_hist(tokens).to(planner.device)
                    # buffer.push(xt, intent_to_idx(corr, DEFAULT_VOCAB), corr_cont, reward=0.5)
                    # train_adapter(planner.adapter, buffer,
                    #             lr=config.get("adapter_lr", 5e-4),
                    #             epochs=config.get("adapter_epochs", 3))
                    corr = oracle.corrective_intent(tokens)
                    adapter_device = next(planner.adapter.parameters()).device
                    xt = scene_to_tensor_hist(tokens).to(adapter_device)
                    with torch.no_grad():
                        _ = planner.adapter(xt.unsqueeze(0)).squeeze(0)
                    corr_cont_5 = planner.adapter.oracle_targets(corr).to(adapter_device)
                    opt = torch.optim.AdamW(planner.adapter.parameters(),
                                            lr=config.get("adapter_lr", 5e-4))
                    for _ in range(config.get("adapter_epochs", 3)):
                        opt.zero_grad()
                        pred = planner.adapter(xt.unsqueeze(0)).squeeze(0)
                        loss = nn.functional.mse_loss(pred, corr_cont_5)
                        loss.backward()
                        opt.step()
                else:
                    planner.logged_update(
                        tokens, oracle.corrective_intent(tokens),
                        lr=config.get("adapter_lr", 5e-4),
                        epochs=config.get("adapter_epochs", 3),
                        reward=r,
                    )
                adapter_updates += 1

            if args.debug_every > 0 and step % args.debug_every == 0:
                print(f"  [ep={ep} step={step:4d}]  "
                      f"intent={intent.get('intent_token','?'):22s}  "
                      f"in_goal={info.get('in_goal',0)}/{args.n_sheep}  "
                      f"r={r:+.4f}")

            if not args.no_video and step % args.capture_every == 0:
                frames.append(inference_render_frame(
                    state, step, intent, r, breakdown,
                    args.n_sheep, info.get("in_goal", 0),
                ))

            if done:
                break

        if len(buffer) >= 16:
            losses = train_adapter(planner.adapter, buffer,
                                   lr=args.lr, epochs=5,
                                   batch_size=args.batch_size)
            loss_str = f"  loss={losses[-1]:.5f}" if losses else ""
        else:
            loss_str = ""

        ep_log.append({
            "episode":   ep,
            "success":   info.get("success", False),
            "steps":     step,
            "in_goal":   info.get("in_goal", 0),
            "ep_reward": round(ep_r, 4),
        })
        print(f"  [ep={ep:4d}]  success={info.get('success',False)}  "
              f"steps={step}  in_goal={info.get('in_goal',0)}/{args.n_sheep}  "
              f"ep_r={ep_r:+.3f}{loss_str}")

        if not args.no_video and frames:
            save_video(frames, out / f"train_{args.mode}_ep{ep:04d}", fps=args.fps)

        if ep % args.save_every == 0 or ep == args.episodes:
            ckpt_path = ckpt_dir / f"adapter_ep{ep:05d}.pt"
            torch.save({"adapter": planner.adapter.state_dict(), "episode": ep}, ckpt_path)
            torch.save({"adapter": planner.adapter.state_dict(), "episode": ep},
                       ckpt_dir / "adapter_latest.pt")
            print(f"  Checkpoint → {ckpt_path}")

        sr = float(np.mean([e["success"] for e in ep_log[-20:]]))
        if sr > best_sr:
            best_sr = sr
            torch.save({"adapter": planner.adapter.state_dict(),
                        "episode": ep, "success_rate": sr},
                       ckpt_dir / "adapter_best.pt")

    fd.reset_episode()

    csv_path = out / f"train_log_{args.mode}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ep_log[0].keys()))
        w.writeheader(); w.writerows(ep_log)

    sr_all = float(np.mean([e["success"] for e in ep_log]))
    print(f"\n[train] CSV: {csv_path}")
    print(f"[train] Overall success rate: {sr_all:.3f}  best={best_sr:.3f}")


def cmd_inference(args: argparse.Namespace) -> None:
    """Evaluate a checkpoint over N episodes and save video + CSV via inference.run_inference."""
    mode = "stringnet" if args.no_ckpt else args.mode
    ckpt = None        if args.no_ckpt else args.ckpt

    print(f"\n[inference] mode={mode}  n_sheep={args.n_sheep}  "
          f"n_dogs={args.n_dogs}  episodes={args.episodes}")
    print(f"[inference] ckpt={ckpt or 'None (rule/untrained)'}")

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

    if not args.no_video:
        for gif in Path(args.output_dir).glob(f"inference_{mode}_ep*.gif"):
            _gif_to_mp4(gif)


def cmd_headless(args: argparse.Namespace) -> None:
    """Run a headless episode via headless.run_headless and convert the output GIF to MP4."""
    print(f"\n[headless] mode={args.mode}  sheep={args.n_sheep}  "
          f"dogs={args.n_dogs}  steps={args.max_steps}")

    result = run_headless(
        mode          = args.mode,
        n_a           = args.n_sheep,
        n_d           = args.n_dogs,
        seed          = args.seed,
        max_steps     = args.max_steps,
        capture_every = args.capture_every,
        debug_every   = args.debug_every,
        output_dir    = args.output_dir,
        qwen_model    = args.qwen_model,
        speed_scale   = args.speed_scale,
    )
    print(f"\n[headless] Result: {result}")

    if not args.no_video:
        gif = Path(args.output_dir) / f"herding_{args.mode}.gif"
        if gif.exists():
            _gif_to_mp4(gif)


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare modes via compare.run_comparison, saving chart PNG, JSON summary, and CSV."""
    modes = (["stringnet", "stringnet_llm", "stringnet_param", "strombom_llm"]
             if getattr(args, "all_modes", False) else args.compare_modes)

    print(f"\n[compare] modes={modes}  n_sheep={args.n_sheep}  "
          f"n_dogs={args.n_dogs}  episodes={args.episodes}")

    compare_run_comparison(
        modes      = modes,
        n_a        = args.n_sheep,
        n_d        = args.n_dogs,
        episodes   = args.episodes,
        seed_base  = args.seed,
        max_steps  = args.max_steps,
        ckpt       = args.ckpt,
        qwen       = args.qwen_model,
        output_dir = args.output_dir,
    )


def cmd_simulate(args: argparse.Namespace) -> None:
    """Launch a live visualised episode via main_rl.run_simulation."""
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

    result = run_simulation(
        mode         = mode,
        n_a          = n_a,
        n_d          = n_d,
        seed         = args.seed,
        qwen_model   = qwen_model,
        render_every = args.render_every,
        debug_every  = args.debug_every,
        speed_scale  = args.speed_scale,
        config_path  = str(Path(__file__).parent / "configs" / "default.yaml"),
        use_beam     = not args.no_beam,
        use_hist     = not args.no_hist,
    )
    print(f"\n[simulate] Result: {result}")


def _common_args(p: argparse.ArgumentParser) -> None:
    """Register arguments shared by all sub-commands."""
    p.add_argument("--n-sheep",       type=int,   default=6)
    p.add_argument("--n-dogs",        type=int,   default=3)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--max-steps",     type=int,   default=600)
    p.add_argument("--output-dir",    default="outputs")
    p.add_argument("--capture-every", type=int,   default=10)
    p.add_argument("--debug-every",   type=int,   default=50)
    p.add_argument("--fps",           type=int,   default=10)
    p.add_argument("--no-video",      action="store_true")
    p.add_argument("--qwen-model",    default="Qwen/Qwen2.5-7B-Instruct")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser with all five sub-commands."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="StringNet herding — unified entry point",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train adapter with RL + oracle supervision")
    _common_args(p_train)
    p_train.add_argument("--mode",
                         choices=["stringnet_llm", "stringnet_param", "strombom_llm"],
                         default="stringnet_llm")
    p_train.add_argument("--episodes",   type=int,   default=200)
    p_train.add_argument("--ckpt-dir",   default="checkpoints")
    p_train.add_argument("--save-every", type=int,   default=50)
    p_train.add_argument("--lr",         type=float, default=5e-4)
    p_train.add_argument("--batch-size", type=int,   default=32)

    p_inf = sub.add_parser("inference", help="Evaluate a trained checkpoint")
    _common_args(p_inf)
    p_inf.add_argument("--mode",
                       choices=["stringnet", "stringnet_llm", "stringnet_param", "strombom_llm"],
                       default="stringnet_llm")
    p_inf.add_argument("--ckpt",           default=None)
    p_inf.add_argument("--no-ckpt",        action="store_true")
    p_inf.add_argument("--episodes",       type=int, default=3)
    p_inf.add_argument("--verbose-reward", action="store_true")

    p_hl = sub.add_parser("headless", help="Headless episode runner (no display required)")
    _common_args(p_hl)
    p_hl.add_argument("--mode",        choices=["stringnet", "llm"], default="llm")
    p_hl.add_argument("--speed-scale", type=float, default=1.2)

    p_cmp = sub.add_parser("compare", help="Compare multiple modes side-by-side")
    _common_args(p_cmp)
    p_cmp.add_argument("--episodes",  type=int, default=10)
    p_cmp.add_argument("--ckpt",      default=None)
    p_cmp.add_argument("--modes",
                       dest="compare_modes", nargs="+",
                       default=["stringnet", "stringnet_llm"],
                       choices=["stringnet", "stringnet_llm", "stringnet_param", "strombom_llm"])
    p_cmp.add_argument("--all-modes", action="store_true")

    p_sim = sub.add_parser("simulate", help="Live interactive simulation with real-time visualiser")
    _common_args(p_sim)
    p_sim.add_argument("--mode",
                       choices=["strombom", "strombom_llm", "stringnet", "stringnet_llm", "select"],
                       default="select")
    p_sim.add_argument("--render-every", type=int,   default=1)
    p_sim.add_argument("--speed-scale",  type=float, default=1.2)
    p_sim.add_argument("--no-beam",      action="store_true")
    p_sim.add_argument("--no-hist",      action="store_true")

    return parser


def main() -> None:
    """Parse CLI arguments and dispatch to the matching sub-command."""
    parser = build_parser()
    args   = parser.parse_args()

    t0 = time.perf_counter()

    dispatch = {
        "train":    cmd_train,
        "inference": cmd_inference,
        "headless": cmd_headless,
        "compare":  cmd_compare,
        "simulate": cmd_simulate,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)
    print(f"\n[main] Finished in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()