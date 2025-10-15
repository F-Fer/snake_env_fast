import numpy as np
from snake_env_fast._fastenv import BatchedEnv, RenderMode
from tests.utils import show_state


def make_env():
    return BatchedEnv(
        num_envs=1,
        mode=RenderMode.RGB,
        map_size=20,
        step_size=1,
        max_steps=50,
        max_turn=np.pi / 4.0,
        eat_radius=0.5,
        max_segments=8,
        initial_segments=3,
        segment_radius=0.5,
        min_segment_distance=1.0,
        cell_size=1.0,
        num_bots=1,
        max_bot_segments=4,
        num_food=0,
        bot_ai_enabled=False,
    )


def to_array(coords):
    xs, ys = zip(*coords)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def setup_state(env, player_coords, bot_coords, player_angle=0.0, bot_angle=1.5 * np.pi):
    px, py = to_array(player_coords)
    bx, by = to_array(bot_coords)
    env.debug_set_player_state(0, px, py, player_angle)
    env.debug_set_bot_state(0, 0, bx, by, bot_angle, True)
    env.debug_rebuild_spatial_hash(0)
    env.reward[0] = 0.0
    env.terminated[0] = 0


def step(env):
    env.step(np.zeros((env.N, env.act_dim), dtype=np.float32))


def test_head_to_head_resets_environment():
    env = make_env()
    setup_state(env, [(5, 5), (4, 5), (3, 5)], [(5, 5), (6, 5), (7, 5)])
    step(env)
    assert env.terminated[0] == 1


def test_bot_hits_player_body():
    env = make_env()
    setup_state(env, [(5, 5), (4, 5), (3, 5), (2, 5)], [(3, 4), (3, 3), (3, 2)], bot_angle=0.0, player_angle=0.0)
    show_state(env)
    step(env)
    show_state(env)
    step(env)
    show_state(env)
    step(env)
    show_state(env)
    print(env.terminated)
    print(env.bot_alive)
    print(env.reward)
    assert env.terminated[0] == 0
    assert env.reward[0] > 0


def test_player_head_hits_bot():
    env = make_env()
    setup_state(env, [(4, 5), (3, 5), (2, 5)], [(6, 5), (5, 5), (4, 5)])
    step(env)
    assert env.terminated[0] == 1

if __name__ == "__main__":
    test_bot_hits_player_body()
