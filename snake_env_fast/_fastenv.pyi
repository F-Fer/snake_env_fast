# Type stubs for _fastenv C++ extension
from enum import IntEnum
from typing import Dict, Any
import numpy as np
import numpy.typing as npt

class RenderMode(IntEnum):
    Headless: int
    RGB: int

class BatchedEnv:
    N: int
    obs_dim: int
    act_dim: int
    grid_w: int
    grid_h: int
    
    def __init__(
        self,
        num_envs: int,
        mode: RenderMode = RenderMode.Headless,
        map_size: int = 100,
        step_size: int = 1,
        max_steps: int = 1000,
        max_turn: float = ...,
        eat_radius: float = 1.0,
        seed: int = 0,
        max_segments: int = 64,
        initial_segments: int = 4,
        segment_radius: float = 2.0,
        min_segment_distance: float = 3.0,
        cell_size: float = 3.0,
        num_bots: int = 3,
        max_bot_segments: int = 12,
        num_food: int = 5,
        food_reward: float = 1.0,
        kill_reward: float = 5.0,
        death_reward: float = -1.0,
        bot_ai_enabled: bool = True,
    ) -> None: ...
    
    @property
    def single_observation_space(self) -> Dict[str, Any]: ...
    
    @property
    def single_action_space(self) -> Dict[str, Any]: ...
    
    @property
    def obs(self) -> npt.NDArray[np.float32]: ...
    
    @property
    def rgb(self) -> npt.NDArray[np.uint8]: ...
    
    @property
    def reward(self) -> npt.NDArray[np.float32]: ...
    
    @property
    def terminated(self) -> npt.NDArray[np.uint8]: ...
    
    @property
    def truncated(self) -> npt.NDArray[np.uint8]: ...
    
    @property
    def grid(self) -> npt.NDArray[np.int32]: ...
    
    @property
    def bot_alive(self) -> npt.NDArray[np.uint8]: ...
    
    def reset(self, mask: npt.NDArray[np.uint8]) -> None: ...
    
    def step(self, actions: npt.NDArray[np.float32]) -> None: ...
    
    def set_seed(self, seed: int) -> None: ...
    
    def render_rgb(self) -> None: ...
    
    def debug_set_player_state(
        self,
        env_idx: int,
        xs: npt.NDArray[np.float32],
        ys: npt.NDArray[np.float32],
        angle: float,
    ) -> None: ...
    
    def debug_set_bot_state(
        self,
        env_idx: int,
        bot_idx: int,
        xs: npt.NDArray[np.float32],
        ys: npt.NDArray[np.float32],
        angle: float,
        alive: bool,
    ) -> None: ...
    
    def debug_rebuild_spatial_hash(self, env_idx: int) -> None: ...


