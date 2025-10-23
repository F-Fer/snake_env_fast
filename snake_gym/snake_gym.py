# snake_env_fast/gym_wrapper.py
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
import numpy as np
import logging
from snake_gym_core.snake_gym_core import SnakeGymCore, RenderMode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SnakeGym(VectorEnv):
    metadata = {"render_modes": ["rgb_array", "headless"], "render_fps": 30}
    def __init__(
        self,
        num_envs: int,
        *,
        map_size: int = 100,
        step_size: int = 1,
        max_steps: int = 1000,
        max_turn: float = np.pi / 4.0,
        eat_radius: float = 1.0,
        seed: int | None = None,
        render_mode: str | None = None,
        num_bots: int = 3,
        max_bot_segments: int = 12,
        num_food: int = 5,
    ):
        self._render_mode = RenderMode.RGB if render_mode == "rgb_array" else RenderMode.Headless
        self._core = SnakeGymCore(
            num_envs,
            map_size,
            step_size,
            max_steps,
            float(max_turn),
            float(eat_radius),
            0 if seed is None else int(seed),
            max_segments=32,
            initial_segments=12,
            segment_radius=1.0,
            min_segment_distance=1.0,
            cell_size=1.0,
            num_bots=num_bots,
            max_bot_segments=max_bot_segments,
            num_food=num_food,
        )
        self.num_envs = num_envs
        if self._render_mode == RenderMode.RGB:
            obs_dim = [84, 84, 3]
            self.single_observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32
            )
        else:
            obs_dim = [7]
            self.single_observation_space = spaces.Box(
                low=0, high=255, shape=obs_dim, dtype=np.uint8
            )
        self.single_action_space = spaces.Box(
            low=np.float32(-self._core.max_turn),
            high=np.float32(self._core.max_turn),
            shape=(self._core.act_dim,),
            dtype=np.float32,
        )
        # Batched spaces required by VectorEnv
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._core.set_seed(int(seed))
        mask = np.ones(self.num_envs, dtype=np.uint8)
        self._core.reset(mask)
        if self._render_mode == RenderMode.RGB:
            self._core.render_rgb()
            obs = np.asarray(self._core.rgb, dtype=np.uint8).reshape(self.num_envs, *self.single_observation_space.shape)
        else:
            obs = np.asarray(self._core.obs, dtype=np.float32).reshape(self.num_envs, *self.single_observation_space.shape)
        return obs, {}

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        # Ensure shape (num_envs, act_dim)
        act_dim = self._core.act_dim
        if actions.ndim == 1:
            actions = actions.reshape(self.num_envs, act_dim)
        if actions.shape != (self.num_envs, act_dim):
            raise ValueError(f"actions must have shape ({self.num_envs}, {act_dim}), got {actions.shape}")
        actions = np.ascontiguousarray(actions, dtype=np.float32)
        self._core.step(actions)
        if self._render_mode == RenderMode.RGB:
            self._core.render_rgb()
            obs = np.asarray(self._core.rgb, dtype=np.uint8).reshape(self.num_envs, *self.single_observation_space.shape)
        else:
            obs = np.asarray(self._core.obs, dtype=np.float32).reshape(self.num_envs, *self.single_observation_space.shape)
        rew = np.asarray(self._core.reward, dtype=np.float32)
        term = np.asarray(self._core.terminated, dtype=bool)
        trunc = np.asarray(self._core.truncated, dtype=bool)
        if term.any() or trunc.any():
            self._core.reset((term | trunc).astype(np.uint8))
        return obs, rew, term, trunc, {}

    def render(self):
        if self._render_mode == RenderMode.RGB:
            self._core.render_rgb()
            frames = np.asarray(self._core.rgb, dtype=np.uint8)
            return frames.reshape(self.num_envs, *self.single_observation_space.shape)
        else:
            return None

if __name__ == "__main__":
    env = SnakeGym(16, render_mode="rgb_array")
    obs, _ = env.reset()
    print(f"obs shape: {obs.shape}")
    frames = env.render()
    print(f"frames[0] shape: {frames[0].shape}")
    action = env.action_space.sample()
    obs, rew, term, trunc, _ = env.step(action)
    frames = env.render()
    print(f"step frames[0] shape: {frames[0].shape}")