# snake_env_fast/gym_wrapper.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ._fastenv import BatchedEnv

class FastVectorEnv(gym.vector.VectorEnv):
    metadata = {"render_modes": []}
    def __init__(self, num_envs: int):
        self._core = BatchedEnv(num_envs)
        self.single_observation_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        # C++ expects an angle in [0, 2*pi)
        self.single_action_space = spaces.Box(0.0, 2*np.pi, shape=(1,), dtype=np.float32)
        super().__init__(num_envs, self.single_observation_space, self.single_action_space)

    def reset(self, seed=None, options=None):
        mask = np.ones(self.num_envs, dtype=np.uint8)
        self._core.reset(mask)
        obs = np.asarray(self._core.obs, dtype=np.float32).reshape(self.num_envs, 1)
        return obs, {}

    def step(self, actions):
        actions = np.ascontiguousarray(actions, dtype=np.float32)
        self._core.step(actions)
        obs = np.asarray(self._core.obs, dtype=np.float32).reshape(self.num_envs, 1)
        rew = np.asarray(self._core.reward, dtype=np.float32)
        term = np.asarray(self._core.terminated, dtype=bool)
        trunc = np.asarray(self._core.truncated, dtype=bool)
        if term.any() or trunc.any():
            self._core.reset((term | trunc).astype(np.uint8))
        return obs, rew, term, trunc, {}