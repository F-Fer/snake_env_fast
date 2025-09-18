# snake_env_fast/gym_wrapper.py
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
import numpy as np
from ._fastenv import BatchedEnv

class FastVectorEnv(VectorEnv):
    metadata = {"render_modes": []}
    def __init__(self, num_envs: int):
        self._core = BatchedEnv(num_envs)
        self.num_envs = num_envs
        # Build spaces from core dims (source of truth)
        obs_dim = int(self._core.obs_dim)
        act_dim = int(self._core.act_dim)
        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=0.0, high=2*np.pi, shape=(act_dim,), dtype=np.float32
        )
        # Batched spaces required by VectorEnv
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)

    def reset(self, seed=None, options=None):
        mask = np.ones(self.num_envs, dtype=np.uint8)
        self._core.reset(mask)
        obs_dim = self.single_observation_space.shape[0]
        obs = np.asarray(self._core.obs, dtype=np.float32)
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

if __name__ == "__main__":
    env = FastVectorEnv(1)
    obs, _ = env.reset()
    print(f"obs: {obs}")
    action = env.action_space.sample()
    obs, rew, term, trunc, _ = env.step(action)
    print(f"obs: {obs}")
    print(f"rew: {rew}")
    print(f"term: {term}")
    print(f"trunc: {trunc}")