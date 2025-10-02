# Snake Environment (Fast C++ Implementation)

A high-performance, parallel Snake environment implemented in C++ with Python bindings, designed for reinforcement learning.

## Features

✅ **Bot Snakes** - Configurable AI opponents that compete for food
- Default: 3 bots with max 12 segments each (memory-efficient)
- Simple food-seeking AI
- Collision detection with player
- Orange/yellow rendering in RGB mode

🚀 **Batched Parallel Execution** - Run thousands of environments simultaneously
🎮 **Vector Observations** - 7D state: (head_x, head_y, dir_angle, snake_len, food_x, food_y, food_dist)
🎨 **RGB Rendering** - 84×84×3 visual output for each environment

## Build

```bash
# Install dependencies
uv pip install scikit-build-core pybind11

# Build and install
uv pip install -e . --no-build-isolation
```

## Quick Start

```python
from snake_env_fast._fastenv import BatchedEnv, RenderMode
import numpy as np

# Create environment with bot snakes
env = BatchedEnv(
    num_envs=4,
    mode=RenderMode.RGB,
    num_bots=5,           # 5 bot opponents
    max_bot_segments=12,  # Max segments per bot
    map_size=100
)

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

## TODO

- ✅ Bot snakes (COMPLETED)
- More Food items (multiple spawn points)
- Larger map support
- RGB rendering focus on snake head
- More realistic rendering
- Bot-vs-bot collision