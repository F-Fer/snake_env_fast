import numpy as np
import cv2
import matplotlib.pyplot as plt


def show_state(env, grid=True):
    # Helper function to render the grid
    def render_grid(env, scale_factor):
        grid = env.grid
        grid_w = env.grid_w
        grid_h = env.grid_h
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

    # Render
    if grid:
        frame_bgr = render_grid(env, 8)
    else:
        env.render_rgb()
        frames = np.asarray(env.rgb, dtype=np.uint8)
        frame_bgr = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
    plt.imshow(frame_bgr)
    plt.show()