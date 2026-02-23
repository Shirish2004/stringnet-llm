from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SCENE_DIM = 6
DEFAULT_VOCAB = [
    "tighten_net", "widen_net", "flank_left", "flank_right",
    "split", "focus_largest_cluster", "delay_enclose",
    "increase_speed", "reduce_speed", "move_leader_to",
]


def scene_to_tensor(scene_tokens: dict) -> torch.Tensor:
    """Encode symbolic scene tokens into a fixed-size float32 tensor."""
    return torch.tensor([
        float(scene_tokens["ACoM"][0]),
        float(scene_tokens["ACoM"][1]),
        float(scene_tokens["sheep_spread"]),
        float(scene_tokens["largest_cluster_dist"]),
        float(scene_tokens["escape_prob_est"]),
        float(scene_tokens["obstacle_density_nearby"]),
    ], dtype=torch.float32)


def intent_to_idx(intent: dict, vocab: list[str]) -> int:
    """Map intent dict to vocabulary index; falls back to 0 on unknown token."""
    tok = intent.get("intent_token", vocab[0])
    return vocab.index(tok) if tok in vocab else 0


class AdapterMLP(nn.Module):
    """Trainable adapter: scene features -> intent-logit deltas."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-capacity ring buffer for (scene_tensor, label_idx) online training pairs."""

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self._xs: list[torch.Tensor] = []
        self._ys: list[int] = []
        self._ptr = 0

    def push(self, x: torch.Tensor, y: int) -> None:
        if len(self._xs) < self.capacity:
            self._xs.append(x)
            self._ys.append(y)
        else:
            self._xs[self._ptr] = x
            self._ys[self._ptr] = y
        self._ptr = (self._ptr + 1) % self.capacity

    def __len__(self) -> int:
        return len(self._xs)

    def as_dataset(self) -> TensorDataset:
        return TensorDataset(
            torch.stack(self._xs),
            torch.tensor(self._ys, dtype=torch.long),
        )


def train_adapter(
    adapter: AdapterMLP,
    buffer: ReplayBuffer,
    lr: float = 5e-4,
    epochs: int = 3,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> list[float]:
    """Train adapter on replay buffer with AdamW + cosine LR; returns epoch losses."""
    if device is None:
        device = torch.device("cpu")
    if len(buffer) == 0:
        return []
    dl = DataLoader(buffer.as_dataset(), batch_size=min(batch_size, len(buffer)), shuffle=True)
    adapter.to(device).train()
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    history: list[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(adapter(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item()
        scheduler.step()
        history.append(epoch_loss / max(len(dl), 1))
    adapter.eval()
    return history