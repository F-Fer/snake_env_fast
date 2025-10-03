# Snake Environment (Fast C++ Implementation)

A high-performance, parallel Snake environment implemented in C++ with Python bindings, designed for reinforcement learning.

## Build

```bash
# Install dependencies
uv pip install scikit-build-core pybind11

# Build and install
uv --refresh pip install -e .
```

## Quick Start

```python
from snake_env_fast.gym_wrapper import FastVectorEnv
import numpy as np

# Environment setup
env = FastVectorEnv(1, render_mode="rgb_array")
obs, _ = env.reset()

# Step through environment
for _ in range(100):
    actions = np.random.uniform(-0.5, 0.5, (env.N, env.act_dim)).astype(np.float32)
    env.step(actions)
    obs = env.obs  # Shape: (N, 7)
    reward = env.reward
```

## Play Interactively

```bash
python -m snake_env_fast.play
```

## Demo Bot Snakes

```bash
python demo_bots.py
```

## Configuration

Key parameters:
- `num_envs`: Number of parallel environments
- `num_bots`: Bot snakes per environment (default: 3)
- `max_bot_segments`: Max segments per bot (default: 12)
- `max_segments`: Max segments for player (default: 64)
- `map_size`: Square map size (default: 100)
- `mode`: `RenderMode.Headless` or `RenderMode.RGB`

### BatchedEnv Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `num_envs` | int | required | Number of environments stepped in parallel |
| `mode` | `RenderMode` | `RenderMode::Headless` | Observation/rendering mode |
| `map_size` | int | 100 | World bounds (square) |
| `step_size` | int | 1 | Units the head advances per step |
| `max_steps` | int | 1000 | Auto-truncate episode after this many steps |
| `max_turn` | float | π/4 | Maximum steering delta per step (radians) |
| `eat_radius` | float | 1.0 | Distance threshold to consume food |
| `seed` | uint64 | 0 | Base RNG seed (env _i_ uses `seed + i`) |
| `max_segments` | int | 64 | Capacity of player snake segments |
| `initial_segments` | int | 4 | Starting player length |
| `segment_radius` | float | 2.0 | Collision/render radius of segments |
| `min_segment_distance` | float | 3.0 | Target spacing between consecutive segments |
| `cell_size` | float | 3.0 | Spatial hash cell width/height |
| `num_bots` | int | 3 | Bot snakes per environment |
| `max_bot_segments` | int | 12 | Capacity of each bot snake |
| `num_food` | int | 5 | Food items maintained per environment |

> All parameters are exposed via `snake_env_fast._fastenv.BatchedEnv` and forwarded by the `FastVectorEnv` wrapper. Floating-point defaults are expressed in world units.

## TODO

- Larger map support
- RGB rendering focus on snake head
- More realistic rendering
- Make bots die when they collide with one another
- Respawn bots if they die