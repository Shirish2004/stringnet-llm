from shepherd_codex.planner.llm_planner import LLMPlanner


def test_adapter_update_moves_towards_oracle() -> None:
    vocab = ["tighten_net", "widen_net", "flank_left"]
    planner = LLMPlanner(vocab=vocab, adapter_dim=16, seed=0)
    scene_tokens_fail = {
        "ACoM": [0.15, 0.75],
        "sheep_spread": 0.12,
        "largest_cluster_dist": 0.3,
        "escape_prob_est": 0.7,
        "obstacle_density_nearby": 0.1,
    }
    oracle = {"intent_token": "tighten_net", "phase": "herd", "params": {"speed_scale": 1.1}}
    pre = planner.plan(scene_tokens_fail)
    planner.update_adapter([(scene_tokens_fail, oracle)], lr=1e-2, epochs=20, batch_size=1)
    post = planner.plan(scene_tokens_fail)
    assert pre["intent_token"] != post["intent_token"] or post["intent_token"] == "tighten_net"
    assert post["logits"][vocab.index("tighten_net")] >= pre["logits"][vocab.index("tighten_net")]
