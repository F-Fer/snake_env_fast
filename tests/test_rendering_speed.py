import numpy as np
import time
import matplotlib.pyplot as plt
from snake_env_fast._fastenv import BatchedEnv, RenderMode


def test_rendering_speed(num_envs=64):
    NUM_ENVS = num_envs
    MAP_SIZE = 100
    INITIAL_SEGMENTS = 6
    NUM_STEPS = 1000

    env = BatchedEnv(
        num_envs=num_envs,
        mode=RenderMode.RGB,
        map_size=MAP_SIZE,
        step_size=1,
        max_steps=50,
        max_turn=np.pi / 4.0,
        eat_radius=0.5,
        max_segments=8,
        initial_segments=INITIAL_SEGMENTS,
        segment_radius=0.5,
        min_segment_distance=1.0,
        cell_size=1.0,
        num_bots=1,
        max_bot_segments=4,
        num_food=3,
        bot_ai_enabled=True,
    )

    env.reset(np.ones(NUM_ENVS, dtype=np.uint8))
    start_time = time.time()
    for _ in range(NUM_STEPS):
        env.step(np.zeros((NUM_ENVS, env.act_dim), dtype=np.float32))
        env.render_rgb()
    end_time = time.time()
    print(f"Number of environments: {NUM_ENVS}")
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Time per step: {(end_time - start_time) / NUM_STEPS} seconds")

    print(f"Rendering speed: {NUM_ENVS * NUM_STEPS / (end_time - start_time)} fps")
    return NUM_ENVS * NUM_STEPS / (end_time - start_time)


if __name__ == "__main__":
    num_envs = [1, 4, 8, 16, 32, 64, 128, 256, 512]
    fps = []
    for num_env in num_envs:
        print(f"Testing with {num_env} environments")
        fps.append(test_rendering_speed(num_env))
        print("-" * 50)
    plt.plot(num_envs, fps)
    plt.xlabel("Number of environments")
    plt.ylabel("Rendering speed (fps)")
    plt.title("Rendering speed vs. number of environments")
    plt.show()