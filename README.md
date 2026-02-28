# StringNet Herding Simulation

A multi-agent shepherding simulation where autonomous shepherd dogs herd a flock of sheep into a goal region. The codebase implements two herding algorithms (**StringNet** and **Strömbom**), each optionally augmented by a Qwen3 LLM planner and a trainable PyTorch adapter network.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [File Reference](#file-reference)
- [Installation](#installation)
- [Running the Simulation](#running-the-simulation)
- [Training](#training)
- [Known Issues and Fixes](#known-issues-and-fixes)
- [Configuration](#configuration)

---

## Project Overview

The shepherding problem: given N sheep and M dogs in a 2D arena, the dogs must herd all sheep into a rectangular goal region. Each sheep has no intelligence — it only responds to nearby dogs by fleeing. The dogs must cooperatively form structures that funnel the flock without letting any sheep escape.

Two classical algorithms are implemented:

| Algorithm | Strategy |
|---|---|
| **StringNet** | Dogs form a dynamic semi-circular arc behind the flock and push it as a unit toward the goal |
| **Strömbom** | Dogs alternate between collecting strays and driving the consolidated herd forward |

Both are enhanced with an optional **LLM + adapter** layer:
- A **Qwen3** language model reads a symbolic scene description and proposes a high-level intent token
- A small **PyTorch adapter** (DualHeadAdapter or ParamControlNet) learns to refine the token selection and continuously adjust formation parameters
- An **OracularPlanner** provides corrective supervision when the adapter fails

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │         ShepherdEnv (env.py)    │
                        │  sheep dynamics · dog dynamics  │
                        │  phase tracking · goal check    │
                        └────────────────┬────────────────┘
                                         │ state dict
                             ┌───────────▼────────────┐
                             │    SceneHistory         │
                             │    (sensors.py)         │
                             │  20-dim feature vector  │
                             └───────────┬────────────┘
                                         │ tokens
              ┌──────────────────────────▼────────────────────────────┐
              │                    LLMPlanner / StrombomLLMPlanner      │
              │  (llm.py)                                               │
              │                                                          │
              │  MockLLM ──────┐                                        │
              │  QwenPlanner ──┼──► blend logits ──► DualHeadAdapter   │
              │  Adapter ──────┘        ▼                  ▼           │
              │                   intent token      cont. deltas       │
              └──────────────────────────┬────────────────────────────┘
                                         │ intent dict
                    ┌────────────────────▼──────────────────────────┐
                    │            Action Function                      │
                    │  _sn_action() / _strombom_action()             │
                    │  formation geometry · controller selection     │
                    └────────────────────┬──────────────────────────┘
                                         │ per-dog action vectors
                    ┌────────────────────▼──────────────────────────┐
                    │            Controllers (controllers.py)        │
                    │  StringNet: sig_alpha + saturation + collision │
                    │  Strömbom: PD locomotion toward target         │
                    └───────────────────────────────────────────────┘
```

---

## File Reference

### `shepherd_env/env.py`
The core simulation environment. Implements a gym-like `reset() / step()` interface.

- **`ShepherdEnv`** loads a YAML config, spawns agents, and advances dynamics each step
- **Sheep dynamics**: each sheep integrates a 2D force (currently: flee from nearby dogs within 2 m radius). Needs cohesion forces added — see [Known Issues](#known-issues-and-fixes)
- **Dog dynamics**: semi-implicit Euler integrator driven by controller outputs
- **Phase tracking**: automatically advances `seek → enclose → herd` based on dog-sheep proximity
- **Termination**: episode ends when all sheep are in the goal region or `T_max` steps pass
- The goal region is **randomised** in the right half of the arena on each `reset()`
- Note: the `seed` argument to `reset()` is currently not used — the RNG is always freshly seeded randomly

---

### `shepherd_env/dynamics.py`
Low-level physics integrator.

- **`semi_implicit_euler_step`**: one integration step for the damped double-integrator `ṙ = v`, `v̇ = u - C_D|v|v`. Updates both sheep and dogs each timestep
- **`saturate_norm`**: clips a 2D control vector to a maximum norm (speed limit)

---

### `shepherd_env/controllers.py`
StringNet controller implementing the Chipade & Panagou (2021) formation control laws.

- **`_desired_point(j, formation_params, n_d)`**: computes dog `j`'s desired arc position. Places dogs evenly across a semicircle centered at `formation_params["center"]` with angular offset from `phi`. **Bug**: can return negative x values — see [Known Issues](#known-issues-and-fixes)
- **`_stringnet_control`**: core control law combining:
  - Position error: `h1 = -k1 * sig_alpha(rd - xi, alpha1)` (finite-time convergence)
  - Velocity damping: `h2 = -k2 * sig_alpha(vd, alpha2) + cd|vd|vd`
  - Collision avoidance: repulsion between dogs within 0.6 m
  - Phase gain multiplier: 1.0 / 1.2 / 1.4 for seek / enclose / herd
- **`apply_seeking_controller`**, **`apply_enclosing_controller`**, **`apply_herding_controller`**: wrappers that select the phase gain

---

### `shepherd_env/sensors.py`
Scene feature extraction and temporal history.

- **`feature_extractor(state)`**: computes 6 base features from raw state:
  - `ACoM` — normalised sheep centre of mass (x, y)
  - `sheep_spread` — mean distance of sheep from their centroid
  - `largest_cluster_dist` — max sheep-to-goal distance
  - `escape_prob_est` — function of velocity variance and spread; ranges [0, 1]
  - `obstacle_density_nearby` — always 0 unless obstacles are added
- **`SceneHistory`**: rolling buffer of the last 4 steps. Extends the 6 base features to 20 dims by appending ACoM history (8 dims), escape probability history (4 dims), mean pairwise sheep distance, and min pairwise sheep distance. This lets the adapter detect rising/falling trends, not just the current value
- **`lidar_scan`**: 16-ray lidar returning `[distance, class_id]` per ray; not currently used in the main loop but available for richer observations

---

### `shepherd_env/strombom_controller.py`
Faithful implementation of the Strömbom et al. (2014) multi-dog herding rules.

- **Rule 2**: `C = (1/N)Σxᵢ` (herd centre); `sf = argmax‖xᵢ - C‖` (furthest stray)
- **Rule 3 (Collect)**: if furthest sheep is outside `collect_radius`, move behind it to push it back toward `C`
- **Rule 4 (Drive)**: form a semicircular arc at `C - d_behind * ĝ` where `ĝ` points toward goal
- **Rule 6**: assign the `n_collectors` closest dogs to the stray; remaining dogs form the drive arc
- **`compute_strombom_targets`**: returns `(per_dog_target_positions, phase_string)` where phase is `"collect"`, `"drive"`, or `"done"`
- **`strombom_action`**: PD controller moving each dog toward its target; output saturated to `ubar_d * speed_scale`

---

### `shepherd_env/spawn.py`
Agent initialisation.

- **`spawn_sheep`**: Poisson-disc samples `n_sheep` positions inside a circle of radius `rho_ac` centred at a random point in the left quadrant (`x ∈ [2,6], y ∈ [7,13]`). Raises `RuntimeError` if 20 000 attempts fail — can happen with very large `n_sheep` and small `rho_ac`
- **`spawn_dogs`**: places dogs at 5 hard-coded left-periphery nodes plus Gaussian jitter. Dogs start at `x ≈ 0.5–1.0` near the left edge

---

### `shepherd_env/safety.py`
Parameter feasibility and action projection.

- **`project_planner_params`**: clips all continuous planner outputs to physical bounds (radius_scale ∈ [0.60, 1.50], d_behind ∈ [0.80, 6.00], etc.) and enforces the safety rule that dogs must be faster than sheep (`speed_scale * ubar_d > ubar_a`)
- **`project_action`**: simulates `k_steps` ahead to detect imminent dog-dog collision; halves the action if collision predicted
- **`validate_params_with_rollout`**: runs two short rollouts (proposed vs baseline params) and rejects proposed params if their reward is worse

---

### `planner/train.py`
All neural network architectures and the replay buffer.

**Networks:**

| Class | Input | Output | Use case |
|---|---|---|---|
| `AdapterMLP` | 20-dim scene | 10 logits | Baseline single-head; discrete intent only |
| `DualHeadAdapter` | 20-dim scene | 10 logits + 4 continuous deltas | Default for `stringnet_llm` and `strombom_llm` |
| `ParamControlNet` | 20-dim scene | 5 absolute parameters | `stringnet_param` — MLP directly controls all formation params |
| `CombinedNet` | 20-dim scene | 10 logits + 5 params | Joint architecture; best of both worlds |

**`DualHeadAdapter` detail:**
```
20-dim → Linear(64) → LayerNorm → ReLU → Dropout(0.1)
       → Linear(32) → ReLU
              ↓
   ┌──────────┴──────────┐
Linear(10)           Linear(8)→ReLU→Linear(4)→Tanh
logits [10]          deltas [4]: [Δradius, Δd_behind, Δspeed, Δflank]
```

**`ParamControlNet` detail:**
```
20-dim → Linear(64) → LayerNorm → ReLU → Dropout(0.1)
       → Linear(64) → ReLU → Linear(32) → ReLU
                  ↓
      5 independent heads, each:
      Linear(16) → ReLU → Linear(1) → Sigmoid → rescale to physical range
```
The independent heads give separate gradient flows per parameter.

**Training (`train_adapter`):** Mixed loss combining:
- Cross-entropy on discrete intent head (oracle supervision)
- MSE on continuous head (oracle param targets)
- REINFORCE: `loss_rl = -mean(advantage * log_prob)` where `advantage = reward - mean_buffer_reward`

**`ReplayBuffer`**: fixed-capacity ring buffer storing `(scene_tensor, disc_label, cont_label, reward)` tuples. Used by `train_adapter` and accessible directly for manual pushes.

---

### `planner/llm.py`
The high-level planners.

**`QwenPlanner`**: wraps Hugging Face `AutoModelForCausalLM` (Qwen3-0.6B or 1.7B). Builds a structured prompt from scene tokens, runs greedy decoding, parses JSON response into an intent dict. Falls back silently if model is unavailable. Supports LoRA injection via `peft_config`.

**`LLMPlanner`** (StringNet):

`plan()` blends three logit sources:
```
combined = 0.4 * MockLLM + 0.4 * QwenPlanner + 0.2 * DualHeadAdapter
```
The continuous deltas from the adapter are applied on top of base parameter values from the winning token. Result is safety-projected via `project_planner_params`.

Additional methods:
- `beam_plan()`: proposes N candidate intents by adding noise to adapter logits, evaluates each via short rollout in a copy of the environment, returns the best scoring candidate
- `hierarchical_plan()`: returns phase + per-dog role assignments (`leader`, `flanker_left`, `flanker_right`, `collector`) + reeval timing
- `logged_update()`: on failure, saves a snapshot, pushes a corrective sample, retrains adapter, logs before/after intent tokens and loss history

**`StrombomLLMPlanner`**: mirrors `LLMPlanner` but targets Strömbom's collect/drive parameters instead of StringNet formation params.

---

### `planner/mock_llm.py`
Rule-based fallback planners (no model required).

- **`MockLLM`**: emits `tighten_net` when `escape_prob > 0.45`, `focus_largest_cluster` when spread is large, else `widen_net`. Returns logit-like numpy arrays
- **`OracularPlanner`**: all-knowing corrective oracle; produces the optimal intent given scene tokens
- **`StrombomMockLLM`** / **`StrombomOracle`**: same pattern for Strömbom vocab

---

### `planner/rl.py`
Reward, curriculum, and collision avoidance.

**`dense_reward`** components per step:

| Signal | Direction | Weight |
|---|---|---|
| Goal progress (Δ sheep fraction in goal) | + | ×8.0 |
| In-goal bonus (absolute fraction) | + | ×1.5 |
| Formation quality (1 − normalised error) | + | ×0.3 |
| Containment margin | + | ×0.4 |
| Escape probability | − | ×1.5 |
| Dog collision fraction | − | ×2.0 |
| Step penalty | − | −0.005 |

**`apply_collision_avoidance`**: post-processes all dog actions, adding a `1/d²` repulsion impulse when any two dogs are within `min_dist * 2`. Result is re-clamped to `ubar_d`.

**`CurriculumScheduler`**: 5-stage curriculum (warmup → easy → medium → hard → full). Each stage defines n_sheep, n_dogs, and spread ranges. Advances when a rolling 12-episode success rate exceeds the stage threshold.

---

### `metrics/failure_detector.py`
Online failure detection used to trigger corrective training.

- Tracks formation error, containment margin, and escape probability each step
- Fires `failure = True` when formation error exceeds threshold for `T_fail` consecutive steps
- Tracks `episode_return` and `mean_reward()` across the episode

---

### `visualizer.py`
Live matplotlib visualiser for interactive simulation. Renders arena, sheep (blue circles), dogs (coloured triangles), StringNet arc lines, velocity arrows, phase label, and a metrics panel. Used by `main_rl.run_simulation()`.

---

### `headless.py`
Headless runner for GIF/montage output with no display. Calls `run_headless()` which runs one episode using StringNet or LLM mode, captures frames at a configurable interval, and saves an animated GIF and/or a PNG montage.

---

### `inference.py`
Checkpoint evaluation. `run_inference()` loads a trained adapter, runs N episodes, saves per-step metrics to CSV, optionally saves GIF output, and prints a summary table. Supports `DualHeadAdapter`, `ParamControlNet`, and rule-based baselines.

---

### `compare.py`
Side-by-side comparison of multiple modes. `run_comparison()` runs each mode for N episodes with matched seeds, aggregates results into a CSV and JSON summary, and saves a multi-panel bar chart (success rate, avg steps, avg sheep in goal, mean reward, avg collision steps).

---

### `main_rl.py`
Live interactive simulation with real-time visualisation. Provides:
- `select_mode_interactive()`: matplotlib GUI to pick mode, agent counts, and Qwen model
- `run_simulation()`: single episode with live viz, beam planning, hierarchical role assignment, online adapter updates
- `run_comparison()`: headless multi-mode comparison used by the original standalone script

---

### `sim.py` (unified entry point)
The main CLI wrapping all of the above. Five sub-commands:

| Command | What it does |
|---|---|
| `train` | RL + oracle training loop; saves checkpoints and training log CSV |
| `inference` | Load checkpoint, evaluate N episodes, save GIF/CSV |
| `headless` | Headless episode via `headless.run_headless()`; converts GIF to MP4 |
| `compare` | Side-by-side mode comparison via `compare.run_comparison()` |
| `simulate` | Live interactive episode via `main_rl.run_simulation()` |

---

## Installation

```bash
conda create -n sheep python=3.10
conda activate sheep

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers numpy matplotlib imageio imageio-ffmpeg pyyaml

# optional: LoRA fine-tuning
pip install peft

# optional: video output
pip install imageio[ffmpeg]
```

---

## Running the Simulation

### Quick rule-based test (no GPU, no model download)
```bash
# Headless StringNet, 10 sheep, 3 dogs, 300 steps
python sim.py headless --mode stringnet --n-sheep 10 --n-dogs 3 --max-steps 300

# Rule-based inference, no checkpoint needed
python sim.py inference --no-ckpt --mode stringnet --n-sheep 8 --n-dogs 4 --episodes 3

# Compare all 4 modes (rule-based only, no checkpoint)
python sim.py compare --modes stringnet --episodes 5
```

### Interactive live simulation (requires display)
```bash
# Mode selector GUI
python sim.py simulate

# Directly pick mode
python sim.py simulate --mode stringnet_llm --n-sheep 6 --n-dogs 3
python sim.py simulate --mode strombom --n-sheep 8 --n-dogs 3 --no-beam
```

### Training
```bash
# Train DualHeadAdapter with StringNet + Qwen3-0.6B
python sim.py train --mode stringnet_llm --episodes 200 --n-sheep 6 --n-dogs 3

# Train ParamControlNet (MLP controls all 5 formation params directly)
python sim.py train --mode stringnet_param --episodes 150 --n-sheep 8 --n-dogs 4

# Train Strömbom LLM adapter
python sim.py train --mode strombom_llm --episodes 100 --lr 1e-4
```

### Evaluation
```bash
# Evaluate saved checkpoint
python sim.py inference --ckpt checkpoints/adapter_best.pt --mode stringnet_llm --episodes 10

# Compare best checkpoint against rule baseline
python sim.py compare --modes stringnet stringnet_llm --episodes 20 \
    --ckpt checkpoints/adapter_best.pt --n-sheep 8 --n-dogs 3
```

### Common flags (all sub-commands)
```
--n-sheep N        Number of sheep agents (default: 6)
--n-dogs N         Number of dog agents (default: 3)
--seed N           Random seed (default: 42)
--max-steps N      Steps per episode (default: 600)
--output-dir PATH  Output directory for videos/CSVs/charts
--capture-every N  Capture one video frame every N steps (default: 10)
--debug-every N    Print step debug info every N steps (default: 50)
--fps N            Video frame rate (default: 10)
--no-video         Skip all video output
--qwen-model STR   HuggingFace model ID (default: Qwen/Qwen3-0.6B)
```

---

## Training Details

### Adapter types and when to use them

| Mode | Adapter | Use when |
|---|---|---|
| `stringnet_llm` | `DualHeadAdapter` | You want interpretable intent tokens + parameter fine-tuning |
| `stringnet_param` | `ParamControlNet` | You want maximum parameter control, no token overhead |
| `strombom_llm` | `DualHeadAdapter` | Strömbom controller with LLM guidance |

### Checkpoint format
```python
{
    "adapter": state_dict,    # adapter weights
    "episode": int,           # episode number
    "success_rate": float,    # only in adapter_best.pt
}
```

### Saved files
```
checkpoints/
    adapter_ep00050.pt      # periodic checkpoint
    adapter_latest.pt       # last episode
    adapter_best.pt         # best success rate seen

outputs/
    train_log_<mode>.csv    # per-episode: success, steps, in_goal, reward
    train_<mode>_ep<N>.mp4  # episode video (if imageio-ffmpeg installed)
```

---

## Known Issues and Fixes

### 1. Dogs stand still (StringNet mode)

**Cause:** `_desired_point()` in `controllers.py` can compute negative x positions (outside arena). Dogs press against the left boundary and cannot move toward out-of-bounds targets. Also, `_BASE_D_BEHIND = 3.0` places the formation arc too far behind sheep for the flee response to activate.

**Fix A — Clip desired positions in `controllers.py`:**
```python
def _desired_point(j, formation_params, n_d):
    center   = np.asarray(formation_params["center"], dtype=float)
    phi0     = float(formation_params.get("phi", 0.0))
    rad      = float(formation_params.get("radius", 1.5))
    arena    = float(formation_params.get("arena_size", 20.0))
    ang      = phi0 if n_d == 1 else phi0 + np.pi / 2 + np.pi * j / (n_d - 1)
    pt       = center + rad * np.array([np.cos(ang), np.sin(ang)])
    return np.clip(pt, 0.5, arena - 0.5)
```

**Fix B — Pass `arena_size` into the formation dict wherever `_sn_action` builds it:**
```python
formation = {
    "center":     center,
    "phi":        phi,
    "radius":     radius,
    "arena_size": float(state.get("arena_size", 20.0)),
}
```

**Fix C — Reduce `_BASE_D_BEHIND` so dogs stay within flee distance of rear sheep:**
```python
_BASE_D_BEHIND = 1.5   # was 3.0
```

---

### 2. Sheep disperse instead of flocking

**Cause:** `env.py` sheep dynamics only have a flee-from-dogs force. There are no inter-sheep cohesion or repulsion forces. Without cohesion, sheep scatter the moment any dog comes within 2 m.

**Fix — Add Strömbom-style flocking to `env.py step()`:**

Replace the sheep update block with:
```python
sheep_u   = np.zeros((na, 2), dtype=float)
R_FLEE    = 2.0
R_REP     = 0.5
R_ATT     = 3.0
C_FLEE    = 0.9
C_REP     = 0.4
C_ATT     = 0.08
C_DRAG    = 0.05

flock_com = self.state.sheep_pos.mean(axis=0)

for i in range(na):
    for d in self.state.dog_pos:
        diff = self.state.sheep_pos[i] - d
        dist = float(np.linalg.norm(diff))
        if 1e-6 < dist < R_FLEE:
            sheep_u[i] += C_FLEE * diff / (dist ** 2)

    for k in range(na):
        if k == i:
            continue
        diff = self.state.sheep_pos[i] - self.state.sheep_pos[k]
        dist = float(np.linalg.norm(diff))
        if 1e-6 < dist < R_REP:
            sheep_u[i] += C_REP * diff / (dist ** 2)
        elif R_REP <= dist < R_ATT:
            sheep_u[i] -= C_ATT * diff / dist

    to_com = flock_com - self.state.sheep_pos[i]
    d_com  = float(np.linalg.norm(to_com))
    if d_com > 1e-6:
        sheep_u[i] += C_DRAG * to_com / d_com
```

---

### 3. `env.py` seed is ignored

`reset(seed=...)` creates a new random RNG regardless of the seed argument. Simulations are not reproducible. Fix:

```python
# In reset():
if seed is not None:
    self.rng = np.random.default_rng(seed)
```

---

### 4. Large `--n-sheep` values crash spawn

`sample_poisson_disc_in_circle` in `spawn.py` can fail with `RuntimeError` when asked for many sheep in a small radius. Increase `rho_ac_range` in `configs/default.yaml` for large flocks:
```yaml
rho_ac_range: [1.5, 3.0]   # safe for n_sheep up to ~40
```

---

## Configuration (`configs/default.yaml`)

Key parameters:

```yaml
# Arena
arena_size: 20.0
dt: 0.05
T_max: 1000

# Agent counts (overridden by --n-sheep / --n-dogs)
N_a: 6
N_d: 3

# Dynamics
ubar_d: 3.0          # max dog speed
ubar_a: 2.0          # max sheep speed
C_D: 0.1             # drag coefficient

# Controller gains (StringNet)
k1: 2.0
k2: 1.5
alpha1: 0.7
alpha2: 0.8

# Spawn
rho_ac_range: [0.6, 1.2]   # sheep cluster radius range
d_min: 0.15                # min sheep-sheep distance at spawn

# Adapter training
adapter_dim: 64
adapter_lr: 5e-4
adapter_epochs: 3
max_adapter_updates_per_episode: 3

# Failure detection
containment_margin_thresh: 0.0
formation_error_thresh: 0.4
T_fail: 8
```

---

## Project Structure

```
shepherd_codex/
├── sim.py                      # unified CLI entry point
├── configs/
│   └── default.yaml            # simulation parameters
├── shepherd_env/
│   ├── env.py                  # gym-like environment
│   ├── dynamics.py             # semi-implicit Euler integrator
│   ├── controllers.py          # StringNet formation control laws
│   ├── sensors.py              # scene feature extraction + history
│   ├── spawn.py                # agent initialisation
│   ├── strombom_controller.py  # Strömbom multi-dog rules
│   └── safety.py               # parameter feasibility projection
├── planner/
│   ├── llm.py                  # LLMPlanner, StrombomLLMPlanner, QwenPlanner
│   ├── mock_llm.py             # rule-based fallback planners
│   ├── train.py                # adapter networks + training
│   └── rl.py                   # reward, curriculum, collision avoidance
├── metrics/
│   └── failure_detector.py     # online failure detection
├── visualizer.py               # live matplotlib visualiser
├── headless.py                 # headless GIF runner
├── inference.py                # checkpoint evaluation
├── compare.py                  # multi-mode comparison
└── main_rl.py                  # live simulation + comparison runner
```

---

## References

- Chipade, V. S., & Panagou, E. (2021). *StringNet: Multi-Agent Herding Using a Semantic Network.* IEEE CDC.
- Strömbom, D. et al. (2014). *Solving the shepherding problem: heuristics for herding autonomous, interacting agents.* Journal of the Royal Society Interface.
