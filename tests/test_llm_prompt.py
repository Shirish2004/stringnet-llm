from shepherd_codex.planner.llm_planner import LLMPlanner


def test_planner_builds_prompt_with_scene_tokens_and_vocab() -> None:
    planner = LLMPlanner(seed=1)
    scene = {
        "ACoM": [0.2, 0.4],
        "sheep_spread": 0.03,
        "largest_cluster_dist": 0.2,
        "escape_prob_est": 0.22,
        "obstacle_density_nearby": 0.05,
    }
    prompt = planner.build_prompt(scene)
    assert "StringNet herding" in prompt
    assert "intent_token" in prompt
    assert "tighten_net" in prompt
    out = planner.plan(scene)
    assert out["llm_prompt"] == prompt
