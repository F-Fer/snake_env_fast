from snake_env_fast.gym_wrapper import FastVectorEnv
import cv2

env = FastVectorEnv(1, render_mode="rgb_array")
obs, _ = env.reset()

while True:
    action = env.action_space.sample()
    obs, rew, term, trunc, _ = env.step(action)
    frames = env.render()
    cv2.imshow("frame", frames[0])
    if term.any() or trunc.any():
        # reset
        obs, _ = env.reset()
    cv2.waitKey(1)