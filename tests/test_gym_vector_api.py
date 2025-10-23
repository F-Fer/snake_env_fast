import numpy as np
import pytest

from gymnasium.vector import VectorEnv

from snake_gym.snake_gym import SnakeGym


@pytest.mark.parametrize("render_mode", [None, "rgb_array"])
def test_vector_env_inheritance(render_mode):
    env = SnakeGym(num_envs=2, render_mode=render_mode)
    try:
        assert isinstance(env, VectorEnv)
        assert env.num_envs == 2
    finally:
        env.close()


@pytest.mark.parametrize(
    "render_mode,expected_shape,expected_dtype",
    [
        (None, (7,), np.float32),
        ("rgb_array", (84, 84, 3), np.uint8),
    ],
)
def test_single_and_batched_observation_spaces(render_mode, expected_shape, expected_dtype):
    env = SnakeGym(num_envs=3, render_mode=render_mode)
    try:
        assert env.single_observation_space.shape == expected_shape
        assert env.single_observation_space.dtype == expected_dtype
        assert env.observation_space.shape == (env.num_envs,) + expected_shape
    finally:
        env.close()


def test_action_space_matches_core():
    env = SnakeGym(num_envs=2, render_mode=None)
    try:
        assert env.single_action_space.shape == (env._core.act_dim,)
        assert env.action_space.shape == (env.num_envs, env._core.act_dim)
    finally:
        env.close()


@pytest.mark.parametrize(
    "render_mode,expected_dtype",
    [
        (None, np.float32),
        ("rgb_array", np.uint8),
    ],
)
def test_reset_returns_valid_batch(render_mode, expected_dtype):
    env = SnakeGym(num_envs=4, render_mode=render_mode)
    try:
        obs, info = env.reset()
        assert isinstance(info, dict)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == expected_dtype
    finally:
        env.close()


@pytest.mark.parametrize("render_mode", [None, "rgb_array"])
def test_reset_seed_deterministic_per_mode(render_mode):
    env = SnakeGym(num_envs=2, render_mode=render_mode)
    try:
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        assert np.allclose(obs1, obs2)

        obs3, _ = env.reset(seed=999)
        assert not np.allclose(obs1, obs3)
    finally:
        env.close()


@pytest.mark.parametrize(
    "render_mode,expected_obs_dtype",
    [
        (None, np.float32),
        ("rgb_array", np.uint8),
    ],
)
def test_step_contract(render_mode, expected_obs_dtype):
    env = SnakeGym(num_envs=3, render_mode=render_mode)
    try:
        env.reset()
        actions = env.action_space.sample()
        assert env.action_space.contains(actions)

        obs, rew, term, trunc, info = env.step(actions)

        assert obs.shape == env.observation_space.shape
        assert obs.dtype == expected_obs_dtype
        assert isinstance(info, dict)

        assert rew.shape == (env.num_envs,)
        assert rew.dtype == np.float32
        assert term.shape == (env.num_envs,)
        assert term.dtype == np.bool_
        assert trunc.shape == (env.num_envs,)
        assert trunc.dtype == np.bool_
    finally:
        env.close()


@pytest.mark.parametrize("render_mode", [None, "rgb_array"])
def test_step_rejects_wrong_action_shape(render_mode):
    env = SnakeGym(num_envs=2, render_mode=render_mode)
    try:
        env.reset()
        bad_actions = np.zeros((env.num_envs + 1, env._core.act_dim), dtype=np.float32)
        with pytest.raises(ValueError):
            env.step(bad_actions)
    finally:
        env.close()


@pytest.mark.parametrize("render_mode", [None, "rgb_array"])
def test_render_output_shape(render_mode):
    env = SnakeGym(num_envs=2, render_mode=render_mode)
    try:
        env.reset()
        if render_mode == "rgb_array":
            frames = env.render()
            assert isinstance(frames, np.ndarray)
            assert frames.shape == (env.num_envs, 84, 84, 3)
            assert frames.dtype == np.uint8
        else:
            assert env.render() is None
    finally:
        env.close()


def test_autoreset_on_termination():
    env = SnakeGym(num_envs=1, render_mode=None)
    try:
        env.reset()

        term = np.zeros(env.num_envs, dtype=bool)
        trunc = np.zeros(env.num_envs, dtype=bool)

        for _ in range(env._core.max_steps + 5):
            actions = np.zeros((env.num_envs, env._core.act_dim), dtype=np.float32)
            obs, rew, term, trunc, info = env.step(actions)
            assert obs.shape == env.observation_space.shape
            assert isinstance(info, dict)

            if term.any() or trunc.any():
                mask = np.asarray(env._core.terminated, dtype=bool) | np.asarray(env._core.truncated, dtype=bool)
                assert mask.dtype == np.bool_
                env._core.reset(mask.astype(np.uint8))
                term = np.zeros(env.num_envs, dtype=bool)
                trunc = np.zeros(env.num_envs, dtype=bool)
    finally:
        env.close()

