"""Evaluation loops and CSV/plot outputs for planner-env experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metrics.failure_detector import FailureDetector
from planner.mock_llm import OracularPlanner
from shepherd_env.env import ShepherdEnv
from shepherd_env.controllers import apply_enclosing_controller, apply_herding_controller, apply_seeking_controller
from shepherd_env.sensors import feature_extractor


def _desired_formation(state: dict, n_d: int, d_behind: float = 1.2, radius: float = 1.4):
    acom = state["sheep_pos"].mean(axis=0)
    goal = state["goal"]
    u = goal - acom
    u = u / max(np.linalg.norm(u), 1e-9)
    center = acom - d_behind * u
    return {"center": center, "phi": float(np.arctan2(u[1], u[0])), "radius": radius}


def run_scenario(planner, n_a: int, n_d: int, episodes: int = 50, seed0: int = 0):
    env = ShepherdEnv()
    oracle = OracularPlanner(planner.vocab)
    successes, times, failures, updates = [], [], 0, 0
    for ep in range(episodes):
        st = env.reset(seed=seed0 + ep, config={"N_a": n_a, "N_d": n_d})
        fd = FailureDetector()
        done = False
        step = 0
        while not done and step < env.config.get("T_max", 1000):
            tokens = feature_extractor(st)
            intent = planner.plan(tokens)
            form = _desired_formation(st, n_d, env.config.get("d_behind", 1.2), 1.3 * intent["params"].get("radius_scale", 1.0))
            xi_des = np.vstack([
                form["center"] + form["radius"] * np.array([
                    np.cos(form["phi"] + np.pi / 2 + (np.pi * j / max(1, n_d - 1))),
                    np.sin(form["phi"] + np.pi / 2 + (np.pi * j / max(1, n_d - 1))),
                ])
                for j in range(n_d)
            ])
            acts = {}
            for j in range(n_d):
                if intent["phase"] == "seek":
                    acts[j] = apply_seeking_controller(j, form, st, env.config)
                elif intent["phase"] == "enclose":
                    acts[j] = apply_enclosing_controller(j, form, st, env.config)
                else:
                    acts[j] = apply_herding_controller(j, form, st, env.config)
            st, _, done, info = env.step(acts)
            m = fd.step(st, xi_des, tokens)
            if m["failure"] and updates < env.config.get("max_adapter_updates_per_episode", 3):
                failures += 1
                updates += 1
                planner.logged_update(tokens, oracle.corrective_intent(tokens), lr=env.config.get("adapter_lr", 5e-4), epochs=env.config.get("adapter_epochs", 3), seed=seed0 + ep)
            step += 1
        successes.append(1.0 if info["success"] else 0.0)
        times.append(step)
    return {
        "success_rate": float(np.mean(successes)),
        "mean_herding_time": float(np.mean(times)),
        "num_failures": int(failures),
        "num_adapter_updates": int(updates),
    }


def run_evaluation(planner, out_csv: str = "shepherd_codex/eval_metrics.csv"):
    scenarios = [(3, 2), (5, 3), (8, 5)]
    rows = []
    for na, nd in scenarios:
        res = run_scenario(planner, na, nd, episodes=50)
        res.update({"N_a": na, "N_d": nd})
        rows.append(res)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    xs = np.arange(len(rows))
    plt.figure(figsize=(6, 3))
    plt.bar(xs, [r["success_rate"] for r in rows])
    plt.xticks(xs, [f"{r['N_a']}/{r['N_d']}" for r in rows])
    plt.ylabel("success_rate")
    plt.tight_layout()
    fig_path = str(Path(out_csv).with_suffix(".png"))
    plt.savefig(fig_path)
    return rows, fig_path
