# Snake Environment

A high-performance, parallel Snake environment implemented in C++ with Python bindings, designed for reinforcement learning.

## Build

For optimal performance, ensure OpenMP is available for parallelization. The build now attempts to auto-detect Homebrew's `libomp` on macOS and will emit a warning if OpenMP is missing (falling back to single-threaded execution).

On macOS (Homebrew):
- Install libomp: `brew install libomp`
- If detection still fails, set `OpenMP_ROOT` explicitly (e.g. `export OpenMP_ROOT=/opt/homebrew/opt/libomp`).

Then build/install:
```bash
# Install dependencies
uv pip install scikit-build-core pybind11

# Build and install
uv run --refresh python -m pip install -e .  
```

## Play Interactively

```bash
uv run python snake_gym/play.py
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

> All parameters are exposed via `snake_gym_core.snake_gym_core.SnakeGymCore` and forwarded by the `SnakeGym` wrapper.

## Testing

```bash
uv run pytest tests/test_player_bot_collision.py
```

## TODO

- More realistic rendering
- Make snake colors changable