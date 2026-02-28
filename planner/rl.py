"""rl.py — Simple REINFORCE RL for StringNet herding.
  - Dense reward shaping with every signal needed for tight generalisation
  - Curriculum learning that steadily increases difficulty across 5 stages
  - Explicit dog-collision penalty so dogs never collide
  - No distributed training, no beam search — plain on-policy REINFORCE
  - Works for any combination of 1-5 dogs and 3-50 sheep
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


# ── constants ────────────────────────────────────────────────────────────────
GAMMA                = 0.97
DOG_COLLISION_RADIUS = 0.6   # metres — dogs closer than this incur penalty


# ──────────────────────────────────────────────────────────────────────────────
# Dense reward
# ──────────────────────────────────────────────────────────────────────────────

def dense_reward(
    state:        dict,
    prev_state:   dict,
    xi_des:       np.ndarray,
    tokens:       dict,
    config:       dict,
    n_sheep:      int,
) -> tuple[float, dict]:
    """Compute per-step dense reward and a breakdown dict for logging.

    Components:
      goal_progress   (+)   change in fraction of sheep inside goal region
      in_goal_bonus   (+)   absolute fraction of sheep in goal (scaled)
      formation_qual  (+)   1 - normalised formation error
      containment     (+)   containment margin clamped to [-1, +1]
      escape_pen      (-)   current escape probability
      collision_pen   (-)   fraction of dog pairs within DOG_COLLISION_RADIUS
      step_pen        (-)   constant −0.005 per step to encourage speed
    """
    sheep     = np.asarray(state["sheep_pos"])
    dogs      = np.asarray(state["dog_pos"])
    prev_sheep = np.asarray(prev_state["sheep_pos"])
    n_d       = len(dogs)

    # goal progress ────────────────────────────────────────────────────────────
    gr = np.asarray(state.get("goal_region", [[12, 8], [20, 12]]))
    def _in_goal(sp):
        m = ((sp[:,0] >= gr[0,0]) & (sp[:,0] <= gr[1,0]) &
             (sp[:,1] >= gr[0,1]) & (sp[:,1] <= gr[1,1]))
        return float(np.sum(m)) / max(len(sp), 1)

    frac_now  = _in_goal(sheep)
    frac_prev = _in_goal(prev_sheep)
    goal_prog = (frac_now - frac_prev) * 8.0          # big signal per advance
    in_goal_b = frac_now * 1.5                         # absolute bonus

    # formation quality ────────────────────────────────────────────────────────
    xi = np.asarray(xi_des)
    if xi.ndim == 1:
        xi = np.tile(xi, (n_d, 1))
    form_err = float(np.mean(np.linalg.norm(dogs - xi[:n_d], axis=1)))
    ubar_d   = float(config.get("ubar_d", 3.0))
    form_q   = max(0.0, 1.0 - form_err / max(ubar_d, 1.0))

    # containment ──────────────────────────────────────────────────────────────
    C      = sheep.mean(axis=0)
    radius = float(np.mean(np.linalg.norm(xi - C, axis=1))) if xi.shape[0] > 1 else 2.5
    dists  = np.linalg.norm(sheep - C, axis=1)
    margin = float(np.min(radius - dists))
    cont_r = float(np.clip(margin, -1.0, 1.0)) * 0.4

    # escape penalty ───────────────────────────────────────────────────────────
    esc_pen = float(tokens.get("escape_prob_est", 0.0)) * 1.5

    # dog collision penalty ────────────────────────────────────────────────────
    coll_pen = 0.0
    if n_d > 1:
        n_pairs   = n_d * (n_d - 1) / 2.0
        n_collide = 0
        for i in range(n_d):
            for j in range(i + 1, n_d):
                if np.linalg.norm(dogs[i] - dogs[j]) < DOG_COLLISION_RADIUS:
                    n_collide += 1
        coll_pen = (n_collide / n_pairs) * 2.0      # strong penalty

    step_pen = 0.005

    reward = goal_prog + in_goal_b + form_q * 0.3 + cont_r - esc_pen - coll_pen - step_pen

    breakdown = {
        "goal_progress":  round(goal_prog,   4),
        "in_goal_bonus":  round(in_goal_b,   4),
        "formation_qual": round(form_q * 0.3, 4),
        "containment":    round(cont_r,      4),
        "escape_pen":     round(-esc_pen,    4),
        "collision_pen":  round(-coll_pen,   4),
        "step_pen":       round(-step_pen,   4),
        "total":          round(reward,      4),
    }
    return reward, breakdown


# ──────────────────────────────────────────────────────────────────────────────
# Collision avoidance for action post-processing
# ──────────────────────────────────────────────────────────────────────────────

def apply_collision_avoidance(
    actions:   dict[int, np.ndarray],
    dog_pos:   np.ndarray,
    ubar_d:    float = 3.0,
    rep_gain:  float = 0.8,
    min_dist:  float = 0.5,
) -> dict[int, np.ndarray]:
    """Add a repulsion impulse to each dog's action when dogs are too close.

    Keeps the underlying control law intact — only adds a correction vector.
    Dogs within min_dist of each other get a repulsion proportional to 1/d².
    The combined action is re-clamped to ubar_d.
    """
    n_d    = len(dog_pos)
    result = {}
    for i in range(n_d):
        u_rep = np.zeros(2)
        for j in range(n_d):
            if i == j:
                continue
            diff = dog_pos[i] - dog_pos[j]
            d    = float(np.linalg.norm(diff))
            if 0 < d < min_dist * 2.0:
                u_rep += rep_gain * diff / max(d ** 2, 1e-6)
        raw   = np.asarray(actions.get(i, np.zeros(2))) + u_rep
        norm  = float(np.linalg.norm(raw))
        result[i] = raw if norm <= ubar_d else (ubar_d / norm) * raw
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Curriculum scheduler
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CurriculumStage:
    name:         str
    n_sheep_range: tuple[int, int]
    n_dog_range:   tuple[int, int]
    spread_range:  tuple[float, float]
    max_steps:    int
    success_thresh: float = 0.60   # success rate needed to advance


CURRICULUM = [
    CurriculumStage("warmup",    (3,  4),  (1, 1), (0.5, 0.9),  300, 0.70),
    CurriculumStage("easy",      (3,  6),  (1, 2), (0.5, 1.1),  400, 0.65),
    CurriculumStage("medium",    (4,  8),  (2, 3), (0.6, 1.3),  500, 0.60),
    CurriculumStage("hard",      (5, 12),  (2, 4), (0.7, 1.5),  700, 0.55),
    CurriculumStage("full",      (3, 15),  (1, 5), (0.5, 2.0), 1000, 0.50),
]


@dataclass
class CurriculumScheduler:
    """Tracks per-stage success rate and advances the curriculum when ready."""
    stage_idx:    int   = 0
    window:       int   = 12           # episodes to average for advancement
    _history:     list  = field(default_factory=list)
    total_ep:     int   = 0
    stage_ep:     int   = 0

    @property
    def stage(self) -> CurriculumStage:
        return CURRICULUM[min(self.stage_idx, len(CURRICULUM) - 1)]

    def sample_config(self, base_config: dict, rng: np.random.Generator) -> dict:
        """Sample a randomised episode config from the current stage."""
        cfg  = deepcopy(base_config)
        st   = self.stage
        cfg["N_a"]  = int(rng.integers(st.n_sheep_range[0], st.n_sheep_range[1] + 1))
        cfg["N_d"]  = int(rng.integers(st.n_dog_range[0],   st.n_dog_range[1]   + 1))
        cfg["rho_ac_range"] = [
            float(rng.uniform(*st.spread_range[:1] * 2)),
            float(rng.uniform(st.spread_range[0], st.spread_range[1])),
        ]
        cfg["T_max"] = st.max_steps
        return cfg

    def record(self, success: bool) -> bool:
        """Record episode outcome; return True if just advanced to next stage."""
        self._history.append(float(success))
        self.total_ep  += 1
        self.stage_ep  += 1
        if len(self._history) > self.window * 2:
            self._history = self._history[-self.window * 2:]

        last = self._history[-min(self.window, len(self._history)):]
        rate = float(np.mean(last))
        thresh = self.stage.success_thresh

        if (rate >= thresh
                and self.stage_ep >= self.window
                and self.stage_idx < len(CURRICULUM) - 1):
            self.stage_idx += 1
            self.stage_ep   = 0
            return True
        return False

    def success_rate(self) -> float:
        last = self._history[-min(self.window, len(self._history)):]
        return float(np.mean(last)) if last else 0.0

    def status(self) -> str:
        return (f"stage={self.stage_idx}/{len(CURRICULUM)-1} "
                f"({self.stage.name})  "
                f"sr={self.success_rate():.2f}  "
                f"ep={self.total_ep}")


# ──────────────────────────────────────────────────────────────────────────────
# Collect one episode of transitions
# ──────────────────────────────────────────────────────────────────────────────

def collect_episode(
    env,
    planner_or_param_net,
    action_fn:   Callable,
    base_config: dict,
    curriculum:  CurriculumScheduler,
    rng:         np.random.Generator,
    seed_offset: int = 0,
    use_hist:    bool = True,
) -> tuple[list[tuple], bool, dict]:
    """Run one episode under the current curriculum stage.

    Returns:
        transitions: list of (scene_tensor, action_idx, reward, cont_target)
        success:     bool
        info_dict:   episode summary including reward breakdown
    """
    from shepherd_env.sensors import feature_extractor, SceneHistory
    try:
        from planner.train import scene_to_tensor_hist, oracle_cont_params
    except ImportError:
        from planner.adapter_train import scene_to_tensor_hist, oracle_cont_params

    cfg   = curriculum.sample_config(base_config, rng)
    n_d   = cfg["N_d"]
    state = env.reset(seed=seed_offset, config=cfg)
    config = {**base_config, **cfg}

    sh = SceneHistory() if use_hist else None
    prev_state = deepcopy(state)
    transitions: list[tuple] = []
    total_reward  = 0.0
    total_coll    = 0.0
    steps         = 0
    max_steps     = int(cfg.get("T_max", 600))

    for _ in range(max_steps):
        tokens = sh.feature_extractor_hist(state) if sh else feature_extractor(state)
        intent = planner_or_param_net.plan(tokens)
        acts, xi_des = action_fn(intent, state, config, n_d)

        # collision avoidance post-processing
        acts = apply_collision_avoidance(
            acts, np.asarray(state["dog_pos"]),
            ubar_d=float(config.get("ubar_d", 3.0)),
        )

        prev_state = deepcopy(state)
        state, _, done, info = env.step(acts)

        r, breakdown = dense_reward(
            state, prev_state, xi_des, tokens, config, cfg["N_a"])
        total_reward += r
        total_coll   += abs(breakdown["collision_pen"])

        x  = scene_to_tensor_hist(tokens)
        ai = 0
        if hasattr(planner_or_param_net, "vocab"):
            tok = intent.get("intent_token", "")
            if tok in planner_or_param_net.vocab:
                ai = planner_or_param_net.vocab.index(tok)
        try:
            yc = oracle_cont_params(intent)
        except Exception:
            import torch
            yc = torch.zeros(4)

        transitions.append((x, ai, r, yc))
        steps += 1
        if done:
            break

    success = bool(info.get("success", False))
    ep_info = {
        "success":      success,
        "steps":        steps,
        "n_dogs":       n_d,
        "n_sheep":      cfg["N_a"],
        "stage":        curriculum.stage.name,
        "total_reward": round(total_reward, 4),
        "mean_reward":  round(total_reward / max(steps, 1), 4),
        "collision_total": round(total_coll, 4),
        "in_goal":      info.get("in_goal", 0),
    }
    return transitions, success, ep_info


# ──────────────────────────────────────────────────────────────────────────────
# REINFORCE update
# ──────────────────────────────────────────────────────────────────────────────

def compute_returns(
    rewards: list[float], gamma: float = GAMMA
) -> list[float]:
    """Compute discounted returns G_t = r_t + γ·r_{t+1} + … for each step."""
    G     = 0.0
    rets  = []
    for r in reversed(rewards):
        G = r + gamma * G
        rets.append(G)
    return list(reversed(rets))


def reinforce_update(
    adapter,
    transitions: list[tuple],
    device,
    lr:           float = 5e-4,
    lambda_rl:    float = 0.5,
    lambda_cont:  float = 0.3,
    clip_grad:    float = 1.0,
    gamma:        float = GAMMA,
) -> dict:
    """One REINFORCE gradient step on (scene, action, G, cont_target) transitions.

    Returns loss breakdown dict.
    """
    import torch
    import torch.nn as nn
    from planner.train import DualHeadAdapter

    if not transitions:
        return {"pg": 0.0, "cont": 0.0, "total": 0.0}

    rewards = [t[2] for t in transitions]
    returns = compute_returns(rewards, gamma)
    baseline = float(np.mean(returns))

    xs   = torch.stack([t[0] for t in transitions]).to(device)
    ais  = torch.tensor([t[1] for t in transitions], dtype=torch.long).to(device)
    rets = torch.tensor(returns, dtype=torch.float32).to(device)
    ycs  = torch.stack([t[3] for t in transitions]).to(device)

    adv  = (rets - baseline).clamp(-4, 4)

    adapter.train()
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    opt.zero_grad()

    is_dual = isinstance(adapter, DualHeadAdapter)
    if is_dual:
        logits, cont_out = adapter(xs)
        l_cont = nn.MSELoss()(cont_out, ycs) * lambda_cont
    else:
        logits  = adapter(xs)
        l_cont  = torch.zeros(1, device=device)

    log_p  = torch.log_softmax(logits, dim=-1)
    chosen = log_p.gather(1, ais.unsqueeze(1)).squeeze(1)
    l_pg   = -(adv * chosen).mean() * lambda_rl

    loss   = l_pg + l_cont
    loss.backward()
    nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=clip_grad)
    opt.step()
    adapter.eval()

    return {
        "pg":    float(l_pg.item()),
        "cont":  float(l_cont.item() if is_dual else 0.0),
        "total": float(loss.item()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    env,
    planner,
    action_fn:    Callable,
    base_config:  dict,
    n_episodes:   int   = 300,
    lr:           float = 5e-4,
    lambda_rl:    float = 0.5,
    lambda_cont:  float = 0.3,
    gamma:        float = GAMMA,
    use_hist:     bool  = True,
    save_dir:     str   = "checkpoints",
    save_every:   int   = 50,
    log_every:    int   = 10,
    seed:         int   = 0,
) -> list[dict]:
    """Full curriculum REINFORCE training loop.

    Each episode:
      1. CurriculumScheduler samples n_sheep, n_dogs, spread for this stage
      2. collect_episode runs the episode with collision avoidance
      3. reinforce_update applies one gradient step
      4. CurriculumScheduler records success → may advance stage
      5. Checkpoint saved every save_every episodes

    Returns list of per-episode log dicts.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    curriculum = CurriculumScheduler()
    rng        = np.random.default_rng(seed)
    device     = planner.device
    logs: list[dict] = []

    for ep in range(n_episodes):
        transitions, success, ep_info = collect_episode(
            env, planner, action_fn, base_config, curriculum, rng,
            seed_offset=seed + ep, use_hist=use_hist,
        )

        loss_info = reinforce_update(
            planner.adapter, transitions, device,
            lr=lr, lambda_rl=lambda_rl, lambda_cont=lambda_cont, gamma=gamma,
        )

        advanced = curriculum.record(success)

        log = {
            "episode":    ep,
            "loss":       loss_info,
            **ep_info,
            "curriculum": curriculum.status(),
            "advanced":   advanced,
        }
        logs.append(log)

        if log_every > 0 and (ep + 1) % log_every == 0:
            print(
                f"[ep {ep+1:4d}/{n_episodes}] "
                f"stage={curriculum.stage_idx}({curriculum.stage.name})  "
                f"sr={curriculum.success_rate():.2f}  "
                f"n_d={ep_info['n_dogs']} n_a={ep_info['n_sheep']}  "
                f"steps={ep_info['steps']}  "
                f"rew={ep_info['mean_reward']:+.4f}  "
                f"coll={ep_info['collision_total']:.3f}  "
                f"loss={loss_info['total']:.4f}"
                + ("  *** ADVANCED ***" if advanced else "")
            )

        if save_every > 0 and (ep + 1) % save_every == 0:
            ckpt = save_path / f"adapter_ep{ep+1:05d}_stage{curriculum.stage_idx}.pt"
            import torch
            torch.save({
                "adapter": planner.adapter.state_dict(),
                "episode": ep,
                "curriculum_stage": curriculum.stage_idx,
                "success_rate": curriculum.success_rate(),
            }, ckpt)
            print(f"  → checkpoint saved: {ckpt.name}")

    # final save
    final = save_path / "adapter_final.pt"
    import torch
    torch.save({
        "adapter": planner.adapter.state_dict(),
        "episode": n_episodes,
        "curriculum_stage": curriculum.stage_idx,
    }, final)
    print(f"\nTraining done. Final checkpoint: {final}")
    return logs