import argparse
from snake_gym import SnakeGym
import cv2
import numpy as np
import time

# Global mouse position
mouse_x, mouse_y = 0, 0

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

def render_grid(env_core, scale_factor):
    grid = env_core.grid
    grid_w = env_core.grid_w
    grid_h = env_core.grid_h
    cell_vis = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for y in range(grid_h):
        for x in range(grid_w):
            v = grid[0, y, x]
            if v == -1:
                cell_vis[y, x] = (0, 0, 0)
            elif v == 0:
                cell_vis[y, x] = (0, 0, 255)
            elif v == 1:
                cell_vis[y, x] = (0, 255, 0)
            else:
                cell_vis[y, x] = (0, 165, 255)
    cell_vis = cv2.resize(cell_vis, (grid_w * scale_factor, grid_h * scale_factor), interpolation=cv2.INTER_NEAREST)
    return cell_vis

def main():
    parser = argparse.ArgumentParser(description="Snake viewer")
    parser.add_argument("--mode", choices=["rgb", "grid"], default="rgb", help="Visualization mode")
    args = parser.parse_args()

    num_envs = 2
    env = SnakeGym(num_envs=num_envs, render_mode="rgb_array")
    obs, _ = env.reset()

    window_name = "Snake Game"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    scale_factor = 8
    window_w = 84 * scale_factor
    window_h = 84 * scale_factor
    center_x = window_w // 2
    center_y = window_h // 2

    print("Controls: Move mouse to steer snake. Press 'q' to quit, 'r' to reset.")

    while True:
        dx = mouse_x - center_x
        dy = mouse_y - center_y
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > 10:
            target_angle = np.arctan2(dy, dx)
            current_angle = env._core.obs[0][2]
            angle_diff = target_angle - current_angle
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            max_turn = np.pi / 4.0
            action_val = np.clip(angle_diff, -max_turn, max_turn)
        else:
            action_val = 0.0

        action = np.array([[action_val] for _ in range(num_envs)], dtype=np.float32)
        obs, rew, term, trunc, info = env.step(action)

        if args.mode == "rgb":
            frames = env.render()
            frame = frames[0]
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_scaled = cv2.resize(frame_bgr, (window_w, window_h), interpolation=cv2.INTER_NEAREST)
        else:
            core = env._core
            grid_img = render_grid(core, scale_factor)
            frame_scaled = cv2.resize(grid_img, (window_w, window_h), interpolation=cv2.INTER_NEAREST)

        cv2.line(frame_scaled, (center_x - 10, center_y), (center_x + 10, center_y), (255, 255, 255), 1)
        cv2.line(frame_scaled, (center_x, center_y - 10), (center_x, center_y + 10), (255, 255, 255), 1)
        cv2.circle(frame_scaled, (mouse_x, mouse_y), 5, (0, 255, 255), 2)
        score_text = f"Score: {int(env._core.obs[0][3]) - 4}"
        cv2.putText(frame_scaled, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(window_name, frame_scaled)

        if term.any() or trunc.any():
            print(f"Game Over! Final score: {int(env._core.obs[0][3]) - 4}")
            obs, _ = env.reset()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            obs, _ = env.reset()
            print("Manual reset")
        time.sleep(0.1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()