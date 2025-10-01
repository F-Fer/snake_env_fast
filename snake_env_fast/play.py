from snake_env_fast.gym_wrapper import FastVectorEnv
import cv2
import numpy as np

# Global mouse position
mouse_x, mouse_y = 0, 0

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

# Environment setup
env = FastVectorEnv(1, render_mode="rgb_array")
obs, _ = env.reset()

# Window setup
window_name = "Snake Game"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

# Display scale factor (e.g., 8x for 84x84 -> 672x672)
scale_factor = 8
window_w = 84 * scale_factor
window_h = 84 * scale_factor
center_x = window_w // 2
center_y = window_h // 2

print("Controls: Move mouse to steer snake. Press 'q' to quit, 'r' to reset.")

while True:
    # Compute action: angle from window center to mouse cursor
    dx = mouse_x - center_x
    dy = mouse_y - center_y
    dist = np.sqrt(dx*dx + dy*dy)
    
    if dist > 10:  # deadzone
        # Compute target angle in [-pi, pi]
        target_angle = np.arctan2(dy, dx)
        # Current heading from observation (obs[0][2])
        current_angle = obs[0][2]
        # Compute angular difference wrapped to [-pi, pi]
        angle_diff = target_angle - current_angle
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        # Clamp to max_turn (pi/4 by default)
        max_turn = np.pi / 4.0
        action_val = np.clip(angle_diff, -max_turn, max_turn)
    else:
        # No steering if mouse near center
        action_val = 0.0
    
    action = np.array([[action_val]], dtype=np.float32)
    obs, rew, term, trunc, _ = env.step(action)
    
    # Render and scale frame
    frames = env.render()
    frame = frames[0]  # shape (84, 84, 3)
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Scale up using nearest neighbor (pixelated look)
    frame_scaled = cv2.resize(frame_bgr, (window_w, window_h), interpolation=cv2.INTER_NEAREST)
    
    # Draw crosshair at center
    cv2.line(frame_scaled, (center_x - 10, center_y), (center_x + 10, center_y), (255, 255, 255), 1)
    cv2.line(frame_scaled, (center_x, center_y - 10), (center_x, center_y + 10), (255, 255, 255), 1)
    
    # Draw mouse cursor position
    cv2.circle(frame_scaled, (mouse_x, mouse_y), 5, (0, 255, 255), 2)
    
    # Display score/info
    score_text = f"Score: {int(obs[0][3]) - 4}"  # segments - initial
    cv2.putText(frame_scaled, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow(window_name, frame_scaled)
    
    # Auto-reset on termination
    if term.any() or trunc.any():
        print(f"Game Over! Final score: {int(obs[0][3]) - 4}")
        obs, _ = env.reset()
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        obs, _ = env.reset()
        print("Manual reset")

cv2.destroyAllWindows()