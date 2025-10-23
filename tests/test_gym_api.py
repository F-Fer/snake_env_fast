from snake_gym.snake_gym import SnakeGym

def test_observation_space():
    env = SnakeGym(num_envs=1, render_mode="rgb_array")
    print(env.observation_space.shape)
    print(env.observation_space.low)
    print(env.observation_space.high)
    print(env.observation_space.dtype)
    print(env.observation_space.shape)
    print(env.observation_space.low)
    print(env.observation_space.high)
    print(env.observation_space.dtype)

env = SnakeGym(num_envs=1, render_mode="rgb_array")
obs, _ = env.reset()
print(obs.shape)
action = env.action_space.sample()
print(action.shape)
obs, rew, term, trunc, _ = env.step(action)
print(obs.shape)
print(rew)
print(term)
print(trunc)