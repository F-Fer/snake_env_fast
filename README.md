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

## TODO

- More Food items (multiple spawn points)
- Larger map support
- RGB rendering focus on snake head
- More realistic rendering
- Bot-vs-bot collision