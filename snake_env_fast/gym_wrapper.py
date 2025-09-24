# snake_env_fast/gym_wrapper.py
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
import numpy as np
import logging
from ._fastenv import BatchedEnv, RenderMode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FastVectorEnv(VectorEnv):
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
    ):
        self._render_mode = render_mode
        mode = RenderMode.RGB if render_mode == "rgb_array" else RenderMode.Headless
        self._core = BatchedEnv(
            num_envs,
            mode,
            map_size,
            step_size,
            max_steps,
            float(max_turn),
            float(eat_radius),
            0 if seed is None else int(seed),
        )
        self.num_envs = num_envs
        # Build spaces from core dims (source of truth)
        obs_dim = int(self._core.obs_dim)
        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        core_act = self._core.single_action_space
        self.single_action_space = spaces.Box(
            low=np.float32(core_act["low"]),
            high=np.float32(core_act["high"]),
            shape=tuple(core_act["shape"]),
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
        obs_dim = self.single_observation_space.shape[0]
        obs = np.asarray(self._core.obs, dtype=np.float32).reshape(self.num_envs, obs_dim)
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
        obs_dim = self.single_observation_space.shape[0]
        obs = np.asarray(self._core.obs, dtype=np.float32).reshape(self.num_envs, obs_dim)
        rew = np.asarray(self._core.reward, dtype=np.float32)
        term = np.asarray(self._core.terminated, dtype=bool)
        trunc = np.asarray(self._core.truncated, dtype=bool)
        if term.any() or trunc.any():
            self._core.reset((term | trunc).astype(np.uint8))
        return obs, rew, term, trunc, {}

    def render(self):
        if self._render_mode != "rgb_array":
            return None
        self._core.render_rgb()
        # Return a tuple of (H, W, C) arrays as expected by vector API
        frames = np.asarray(self._core.rgb, dtype=np.uint8)
        # frames shape is (N, 84, 84, 3)
        return tuple(frames[i] for i in range(self.num_envs))

if __name__ == "__main__":
    env = FastVectorEnv(16, render_mode="rgb_array")
    obs, _ = env.reset()
    print(f"obs shape: {obs.shape}")
    frames = env.render()
    print(f"frames[0] shape: {frames[0].shape}")
    action = env.action_space.sample()
    obs, rew, term, trunc, _ = env.step(action)
    frames = env.render()
    print(f"step frames[0] shape: {frames[0].shape}")