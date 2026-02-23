"""Demo run for LLM-augmented StringNet shepherding with online adapter updates."""

from __future__ import annotations

import yaml

from metrics.failure_detector import FailureDetector
from planner.llm_planner import LLMPlanner
from planner.mock_llm import OracularPlanner
from shepherd_env.env import ShepherdEnv
from shepherd_env.controllers import apply_enclosing_controller, apply_herding_controller, apply_seeking_controller
from shepherd_env.sensors import feature_extractor


def main() -> None:
    with open("./configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env = ShepherdEnv("./configs/default.yaml")
    planner = LLMPlanner(adapter_dim=cfg["adapter_dim"], seed=cfg["seed"])
    oracle = OracularPlanner(planner.vocab)

    state = env.reset(seed=42)
    fail = FailureDetector(cfg["containment_margin_thresh"], cfg["formation_error_thresh"], cfg["T_fail"])

    updates = 0
    for t in range(cfg["T_max"]):
        tokens = feature_extractor(state)
        intent_pre = planner.plan(tokens)
        print(f"t={t:03d} intent={intent_pre['intent_token']} phase={intent_pre['phase']}")

        form = {
            "center": state["sheep_pos"].mean(axis=0),
            "phi": 0.0,
            "radius": 1.3 * intent_pre["params"]["radius_scale"],
        }
        xi_des = state["dog_pos"] * 0.0
        actions = {}
        for j in range(state["dog_pos"].shape[0]):
            if intent_pre["phase"] == "seek":
                actions[j] = apply_seeking_controller(j, form, state, cfg)
            elif intent_pre["phase"] == "enclose":
                actions[j] = apply_enclosing_controller(j, form, state, cfg)
            else:
                actions[j] = apply_herding_controller(j, form, state, cfg)
            xi_des[j] = form["center"]

        state, _, done, info = env.step(actions)
        m = fail.step(state, xi_des, tokens)

        if m["failure"] and updates < cfg["max_adapter_updates_per_episode"]:
            corrective = oracle.corrective_intent(tokens)
            log = planner.logged_update(tokens, corrective, lr=cfg["adapter_lr"], epochs=cfg["adapter_epochs"], seed=cfg["seed"])
            updates += 1
            print("Failure detected. Adapter updated.")
            print("pre:", log["pre_plan"]["intent_token"], "post:", log["post_plan"]["intent_token"], "oracle:", corrective["intent_token"])
        if done:
            print(f"Episode ended at t={t}, success={info['success']}, updates={updates}")
            break


if __name__ == "__main__":
    main()
