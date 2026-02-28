"""train.py — Adapter and parameter-control networks for StringNet herding.

  ParamControlNet  — MLP that directly outputs 5 continuous StringNet params:
                     [radius_scale, d_behind, speed_scale, arc_span, flank_bias]
                     These replace hand-tuned defaults and are conditioned on
                     the scene feature vector, giving the MLP full parameter control.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCENE_DIM_BASE = 6
SCENE_DIM_HIST  = 20
CONT_DIM        = 4

# Bounds for ParamControlNet outputs (used in sigmoid re-scaling)
PARAM_BOUNDS = {
    "radius_scale": (0.60, 1.50),
    "d_behind":     (0.80, 6.00),
    "speed_scale":  (0.50, 2.00),
    "arc_span":     (0.30, 3.14),   # radians
    "flank_bias":   (-2.0, 2.00),
}
PARAM_NAMES     = list(PARAM_BOUNDS.keys())   # length 5
PARAM_DIM       = len(PARAM_NAMES)            # 5

DEFAULT_VOCAB = [
    "tighten_net", "widen_net", "flank_left", "flank_right",
    "split", "focus_largest_cluster", "delay_enclose",
    "increase_speed", "reduce_speed", "move_leader_to",
]

STROMBOM_VOCAB = [
    "tighten_collect", "loosen_collect", "push_harder", "back_off",
    "spread_formation", "compress_formation", "speed_up", "slow_down",
    "flank_left", "flank_right",
]

CONT_PARAM_NAMES = ["radius_scale_delta", "d_behind_delta", "speed_scale_delta", "flank_bias_delta"]


# ── feature encoding ──────────────────────────────────────────────────────────

def scene_to_tensor(scene_tokens: dict) -> torch.Tensor:
    """Base 6-dim scene encoding (backward compatible)."""
    return torch.tensor([
        float(scene_tokens["ACoM"][0]),
        float(scene_tokens["ACoM"][1]),
        float(scene_tokens["sheep_spread"]),
        float(scene_tokens["largest_cluster_dist"]),
        float(scene_tokens["escape_prob_est"]),
        float(scene_tokens["obstacle_density_nearby"]),
    ], dtype=torch.float32)


def scene_to_tensor_hist(scene_tokens: dict) -> torch.Tensor:
    """20-dim encoding: base + ACoM history + escape history + pair stats."""
    base = [
        float(scene_tokens["ACoM"][0]),
        float(scene_tokens["ACoM"][1]),
        float(scene_tokens["sheep_spread"]),
        float(scene_tokens["largest_cluster_dist"]),
        float(scene_tokens["escape_prob_est"]),
        float(scene_tokens["obstacle_density_nearby"]),
    ]
    hist_acom = scene_tokens.get("history_acom",  [[0, 0]] * 4)
    hist_esc  = scene_tokens.get("history_escape", [0.0]   * 4)
    mean_pair = float(scene_tokens.get("mean_pair_dist", 0.0))
    min_pair  = float(scene_tokens.get("min_pair_dist",  0.0))
    acom_flat = [v for row in hist_acom for v in row]
    return torch.tensor(base + acom_flat + list(hist_esc) + [mean_pair, min_pair],
                        dtype=torch.float32)


def intent_to_idx(intent: dict, vocab: list[str]) -> int:
    tok = intent.get("intent_token", vocab[0])
    return vocab.index(tok) if tok in vocab else 0


def oracle_cont_params(intent: dict) -> torch.Tensor:
    p  = intent.get("params", {})
    rs = float(p.get("radius_scale",   p.get("radius_scale_delta",  0.0)))
    db = float(p.get("d_behind_scale", p.get("d_behind_delta",      0.0)))
    ss = float(p.get("speed_scale",    p.get("speed_scale_delta",   0.0)))
    fb = float(p.get("flank_bias",     p.get("flank_bias_delta",    0.0)))
    tok = intent.get("intent_token", "")
    if rs == 0.0:
        rs = -0.1 if tok in {"tighten_net","tighten_collect","compress_formation"} else \
              0.1 if tok in {"widen_net","loosen_collect","spread_formation"} else 0.0
    if ss == 0.0:
        ss =  0.1 if tok in {"increase_speed","speed_up","push_harder"} else \
             -0.1 if tok in {"reduce_speed","slow_down","back_off"} else 0.0
    return torch.tensor([rs, db, ss, fb], dtype=torch.float32)


# ── networks ──────────────────────────────────────────────────────────────────

def _kaiming_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)


class AdapterMLP(nn.Module):
    """Backward-compatible single-head: scene → discrete intent logits."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
        )
        self.apply(_kaiming_init)
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualHeadAdapter(nn.Module):
    """Shared trunk → discrete logits + continuous parameter deltas."""

    def __init__(self, in_dim: int, hidden: int, n_vocab: int, cont_dim: int = CONT_DIM) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.disc_head = nn.Linear(hidden // 2, n_vocab)
        self.cont_head = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, cont_dim), nn.Tanh(),
        )
        self.apply(_kaiming_init)
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.disc_head(h), self.cont_head(h)

    def logits_only(self, x: torch.Tensor) -> torch.Tensor:
        return self.disc_head(self.trunk(x))

    def softmax_entropy(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(self.logits_only(x), dim=-1)
        return -torch.sum(p * torch.log(p + 1e-9), dim=-1)


class ParamControlNet(nn.Module):
    """MLP that directly controls all StringNet formation parameters.

    Input:  scene feature vector (SCENE_DIM_HIST = 20 dims by default)
    Output: 5 continuous formation parameters in their physical ranges:
              radius_scale  ∈ [0.60, 1.50]
              d_behind      ∈ [0.80, 6.00]  (metres behind herd centre)
              speed_scale   ∈ [0.50, 2.00]
              arc_span      ∈ [0.30, π]     (radians of formation arc)
              flank_bias    ∈ [−2.0, +2.0]  (left/right lean)

    Architecture: input → 2 hidden layers → 5 sigmoid-rescaled outputs.
    The network can be used standalone or alongside a discrete intent head.

    Usage example:
        net   = ParamControlNet(SCENE_DIM_HIST, hidden=64)
        params = net.decode(scene_tensor)
        # params is a dict: {"radius_scale": 1.1, "d_behind": 2.4, ...}
    """

    def __init__(self, in_dim: int = SCENE_DIM_HIST, hidden: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        # separate output heads per parameter for independent gradient flow
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(hidden // 2, 16), nn.ReLU(),
                                 nn.Linear(16, 1), nn.Sigmoid())
            for name in PARAM_NAMES
        })
        self.apply(_kaiming_init)
        self.in_dim  = in_dim
        self._bounds = PARAM_BOUNDS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw sigmoid outputs [B, PARAM_DIM] ∈ (0,1)."""
        h = self.trunk(x)
        return torch.cat([self.heads[n](h) for n in PARAM_NAMES], dim=-1)

    def decode(self, x: torch.Tensor) -> dict:
        """Forward pass + rescale to physical bounds. Returns a plain dict.

        Can accept a 1D tensor (single scene) or 2D batch (takes first row).
        """
        with torch.no_grad():
            xb  = x.unsqueeze(0) if x.dim() == 1 else x
            xb  = xb.to(next(self.parameters()).device)
            raw = self.forward(xb).squeeze(0).cpu().numpy()
        out = {}
        for i, name in enumerate(PARAM_NAMES):
            lo, hi  = self._bounds[name]
            out[name] = float(lo + raw[i] * (hi - lo))
        return out

    def oracle_targets(self, intent: dict) -> torch.Tensor:
        """Compute normalised (0-1) targets from an intent dict for supervised training."""
        p   = intent.get("params", {})
        tok = intent.get("intent_token", "")

        defaults = {
            "radius_scale": 1.0,
            "d_behind":     3.0,
            "speed_scale":  1.2,
            "arc_span":     2.83,
            "flank_bias":   0.0,
        }
        values = {k: float(p.get(k, defaults[k])) for k in PARAM_NAMES}
        if values["radius_scale"] == 1.0:
            if tok in {"tighten_net","tighten_collect","compress_formation"}:
                values["radius_scale"] = 0.75
            elif tok in {"widen_net","loosen_collect","spread_formation"}:
                values["radius_scale"] = 1.25
        if values["speed_scale"] == 1.2:
            if tok in {"increase_speed","speed_up","push_harder"}:
                values["speed_scale"] = 1.5
            elif tok in {"reduce_speed","slow_down","back_off"}:
                values["speed_scale"] = 0.8

        normed = []
        for name in PARAM_NAMES:
            lo, hi  = self._bounds[name]
            normed.append(float(np.clip((values[name] - lo) / (hi - lo), 0.0, 1.0)))
        return torch.tensor(normed, dtype=torch.float32)

    def softmax_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Dummy entropy (constant 0) for compatibility with planner confidence API."""
        return torch.zeros(x.shape[0] if x.dim() > 1 else 1)


class CombinedNet(nn.Module):
    """Joint network: shared trunk → discrete intent logits + ParamControlNet outputs.

    This is the recommended architecture when you want both:
    - Discrete intent token selection (for logging/LLM override)
    - Direct continuous parameter control (radius, d_behind, speed, arc, flank)
    """

    def __init__(
        self,
        in_dim:  int = SCENE_DIM_HIST,
        hidden:  int = 128,
        n_vocab: int = 10,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.intent_head = nn.Linear(hidden // 2, n_vocab)
        self.param_heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(hidden // 2, 32), nn.ReLU(),
                                 nn.Linear(32, 1), nn.Sigmoid())
            for name in PARAM_NAMES
        })
        self.apply(_kaiming_init)
        self.in_dim    = in_dim
        self._bounds   = PARAM_BOUNDS
        self._n_vocab  = n_vocab

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        params = torch.cat([self.param_heads[n](h) for n in PARAM_NAMES], dim=-1)
        return {
            "intent_logits": self.intent_head(h),
            "params":        params,
        }

    def decode(self, x: torch.Tensor, vocab: list[str]) -> dict:
        """Return intent token + physical param dict from scene tensor."""
        with torch.no_grad():
            xb  = x.unsqueeze(0) if x.dim() == 1 else x
            out = self.forward(xb)
        logits = out["intent_logits"].squeeze(0).cpu().numpy()
        raw    = out["params"].squeeze(0).cpu().numpy()
        tok    = vocab[int(np.argmax(logits))]
        params = {}
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = self._bounds[name]
            params[name] = float(lo + raw[i] * (hi - lo))
        return {"intent_token": tok, "params": params, "logits": logits.tolist()}

    def softmax_entropy(self, x: torch.Tensor) -> torch.Tensor:
        out   = self.forward(x)
        probs = torch.softmax(out["intent_logits"], dim=-1)
        return -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

    # alias so it behaves like DualHeadAdapter for reinforce_update
    def __call__(self, x):
        out = super().__call__(x)
        return out["intent_logits"], out["params"]


# ── replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Ring buffer storing (scene, disc_label, cont_label, reward) tuples."""

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        self._xs:    list[torch.Tensor] = []
        self._ys:    list[int]          = []
        self._yc:    list[torch.Tensor] = []
        self._rews:  list[float]        = []
        self._ptr = 0

    def push(self, x: torch.Tensor, y_disc: int,
             y_cont: torch.Tensor | None = None, reward: float = 0.0) -> None:
        xc = x.detach().to("cpu")
        yc = y_cont.detach().to("cpu") if y_cont is not None else torch.zeros(CONT_DIM)
        if len(self._xs) < self.capacity:
            self._xs.append(xc)
            self._ys.append(y_disc)
            self._yc.append(yc)
            self._rews.append(reward)
        else:
            self._xs[self._ptr] = xc
            self._ys[self._ptr] = y_disc
            self._yc[self._ptr] = yc
            self._rews[self._ptr] = reward
        self._ptr = (self._ptr + 1) % self.capacity

    def __len__(self) -> int:
        return len(self._xs)

    def as_dataset(self) -> TensorDataset:
        return TensorDataset(
            torch.stack(self._xs),
            torch.tensor(self._ys, dtype=torch.long),
            torch.stack(self._yc),
            torch.tensor(self._rews, dtype=torch.float32),
        )

    def mean_reward(self, last_n: int = 50) -> float:
        r = self._rews[-last_n:]
        return float(np.mean(r)) if r else 0.0


# ── supervised training ───────────────────────────────────────────────────────

def train_adapter(
    adapter:     AdapterMLP | DualHeadAdapter | ParamControlNet | CombinedNet,
    buffer:      ReplayBuffer,
    lr:          float = 5e-4,
    epochs:      int   = 3,
    batch_size:  int   = 16,
    device:      torch.device | None = None,
    lambda_cat:  float = 1.0,
    lambda_cont: float = 0.3,
    lambda_rl:   float = 0.1,
) -> list[float]:
    """Supervised + REINFORCE training for any adapter variant.

    DualHeadAdapter / CombinedNet: CE (intent) + MSE (params) + REINFORCE.
    AdapterMLP:                    CE + REINFORCE.
    ParamControlNet:               MSE (params) only.
    """
    if device is None:
        device = torch.device("cpu")
    if len(buffer) == 0:
        return []

    is_dual  = isinstance(adapter, DualHeadAdapter)
    is_param = isinstance(adapter, ParamControlNet)
    is_comb  = isinstance(adapter, CombinedNet)

    dl   = DataLoader(buffer.as_dataset(),
                      batch_size=min(batch_size, len(buffer)), shuffle=True)
    adapter.to(device).train()
    opt  = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    ce   = nn.CrossEntropyLoss(label_smoothing=0.05)
    mse  = nn.MSELoss()
    base = buffer.mean_reward()
    hist: list[float] = []

    for _ in range(epochs):
        ep = 0.0
        for xb, yd, yc, rw in dl:
            xb = xb.to(device)
            yd = yd.to(device)
            yc = yc.to(device)
            rw = rw.to(device)
            opt.zero_grad()
            adv = (rw - base).clamp(-2, 2)

            if is_param:
                raw   = adapter(xb)
                loss  = mse(raw, yc[:, :PARAM_DIM]) * lambda_cont
            elif is_comb or is_dual:
                logits, pout = adapter(xb)
                l_cat = ce(logits, yd) * lambda_cat
                l_con = mse(pout[:, :CONT_DIM], yc) * lambda_cont
                lp    = torch.log_softmax(logits, dim=-1)
                ch    = lp.gather(1, yd.unsqueeze(1)).squeeze(1)
                l_rl  = -(adv * ch).mean() * lambda_rl
                loss  = l_cat + l_con + l_rl
            else:
                logits = adapter(xb)
                l_cat  = ce(logits, yd) * lambda_cat
                lp     = torch.log_softmax(logits, dim=-1)
                ch     = lp.gather(1, yd.unsqueeze(1)).squeeze(1)
                l_rl   = -(adv * ch).mean() * lambda_rl
                loss   = l_cat + l_rl

            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            opt.step()
            ep += loss.item()
        sch.step()
        hist.append(ep / max(len(dl), 1))

    adapter.eval()
    return hist


def train_param_net_supervised(
    param_net:   ParamControlNet,
    transitions: list[tuple],
    intent_list: list[dict],
    device:      torch.device | None = None,
    lr:          float = 1e-3,
    epochs:      int   = 5,
) -> list[float]:
    """Pure supervised training for ParamControlNet from oracle intents.

    transitions:  list of (scene_tensor, …) — only scene tensor used
    intent_list:  matching list of oracle intent dicts for oracle_targets()
    """
    if device is None:
        device = torch.device("cpu")
    param_net.to(device).train()
    xs   = torch.stack([t[0] for t in transitions]).to(device)
    ys   = torch.stack([param_net.oracle_targets(i) for i in intent_list]).to(device)
    opt  = torch.optim.AdamW(param_net.parameters(), lr=lr, weight_decay=1e-4)
    mse  = nn.MSELoss()
    hist = []
    bs   = min(32, len(xs))
    for _ in range(epochs):
        perm = torch.randperm(len(xs))
        ep   = 0.0
        for start in range(0, len(xs), bs):
            idx = perm[start:start + bs]
            opt.zero_grad()
            loss = mse(param_net(xs[idx]), ys[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(param_net.parameters(), 1.0)
            opt.step()
            ep += loss.item()
        hist.append(ep / max(len(xs) // bs, 1))
    param_net.eval()
    return hist