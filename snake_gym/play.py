import argparse
from snake_gym import SnakeGym
import cv2
import numpy as np
import time
from pathlib import Path
import threading
from dataclasses import dataclass
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import repo_exists, interpreter_login

features_schema = {
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["head_x", "head_y", "dir_angle", "snake_len", "nearest_food_x", "nearest_food_y", "nearest_food_dist"],
    },
    "reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["reward"],
    },
    "terminated": {
        "dtype": "bool",
        "shape": (1,),
        "names": ["terminated"],
    },
    "truncated": {
        "dtype": "bool",
        "shape": (1,),
        "names": ["truncated"],
    },
    "action": {
        "dtype": "float32",
        "shape": (1,),
        "names": ["action_val"],
    },
    "observation.images.main_cam": {
        "dtype": "video",
        "shape": (84, 84, 3), # (Height, Width, Channels)
        "names": ["height", "width", "channel"],
        "info": {"video.fps": 30},
    }
}

task = "Play the snake game."
min_episode_length = 15

@dataclass
class RecordFrame:
    state: np.ndarray # [head_x, head_y, dir_angle, snake_len, nearest_food_x, nearest_food_y, nearest_food_dist]
    image: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    task: str
    def to_dict(self):
        return {
            "observation.state": np.asarray(self.state, dtype=np.float32).reshape(7),
            "observation.images.main_cam": self.image,
            "action": np.asarray(self.action, dtype=np.float32).reshape(1),
            "reward": np.asarray(self.reward, dtype=np.float32).reshape(1),
            "terminated": np.asarray(self.terminated, dtype=np.bool_).reshape(1),
            "truncated": np.asarray(self.truncated, dtype=np.bool_).reshape(1),
            "task": self.task,
        }

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
    parser.add_argument("--record", action="store_true", help="Record the game to lerobot format")
    parser.add_argument("--repo-id", type=str, default="snake-gym", help="Repository ID for Lerobot")
    args = parser.parse_args()
    if args.record:
        if not args.repo_id:
            parser.error("--repo-id is required for recording")

        # Check if user is logged in
        interpreter_login(new_session=False)

        # Check if the dataset already exists
        exists_on_hub = repo_exists(args.repo_id, repo_type="dataset")
        local_path = Path.home() / ".cache" / "huggingface" / "lerobot" / args.repo_id
        if not exists_on_hub and not local_path.exists():
            dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=30, features=features_schema)
        else:
            dataset = LeRobotDataset(repo_id=args.repo_id)

        # Initialize the record frames
        last_frame: RecordFrame = None

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
            if args.record:
                if last_frame is not None:
                    last_frame.action = action[0]
                    dataset.add_frame(last_frame.to_dict())
                last_frame = RecordFrame(
                    state=np.copy(env._core.obs[0]),
                    image=frame.copy(),
                    action=np.zeros(1, dtype=np.float32),
                    reward=np.zeros(1, dtype=np.float32),
                    terminated=np.zeros(1, dtype=np.bool_),
                    truncated=np.zeros(1, dtype=np.bool_),
                    task=task,
                )
                last_frame.reward[0] = rew[0]
                last_frame.terminated[0] = term[0]
                last_frame.truncated[0] = trunc[0]
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
            if args.record:
                if dataset.episode_buffer["size"] >= min_episode_length:
                    dataset.save_episode()
                    threading.Thread(target=dataset.push_to_hub).start()
                else:
                    print(f"Episode length {dataset.episode_buffer['size']} is less than {min_episode_length}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if args.record:
                if dataset.episode_buffer["size"] >= min_episode_length:
                    dataset.save_episode()
                dataset.finalize()
                dataset.push_to_hub()
            break
        elif key == ord('r'):
            obs, _ = env.reset()
            print("Manual reset")
            if args.record:
                dataset.clear_episode_buffer()
        time.sleep(0.1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()