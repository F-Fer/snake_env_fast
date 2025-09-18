#include "env_core.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

BatchedEnv::BatchedEnv(int num_envs, RenderMode mode, int map_size, int step_size)
  : N(num_envs),
    obs_dim(mode == RenderMode::Headless ? static_cast<int>(ObservationSize::Headless) : throw std::runtime_error("RenderMode not supported")),
    act_dim(1),
    map_size(map_size),
    step_size(step_size),
    single_action_space(0.0f, kTwoPi, {1}, "float32"),
    single_observation_space(mode == RenderMode::Headless ? BoxSpace(-INFINITY, INFINITY, {static_cast<int>(ObservationSize::Headless)}, "float32") : BoxSpace(0, 0, {0}, "float32")),
    render_mode(mode),
    obs(N * (mode == RenderMode::Headless ? static_cast<int>(ObservationSize::Headless) : 0), 0.f),
    reward(N, 0.f),
    terminated(N, 0),
    truncated(N, 0),
    head_x(N, 0.f),
    head_y(N, 0.f),
    dir_angle(N, 0.f),
    snake_len(N, 1)
{
    full_reset();
}

void BatchedEnv::full_reset() {
  std::fill(obs.begin(), obs.end(), 0.f);
  std::fill(reward.begin(), reward.end(), 0.f);
  std::fill(terminated.begin(), terminated.end(), 0);
  std::fill(truncated.begin(), truncated.end(), 0);
  std::fill(head_x.begin(), head_x.end(), 0.f);
  std::fill(head_y.begin(), head_y.end(), 0.f);
  std::fill(dir_angle.begin(), dir_angle.end(), 0.f);
  std::fill(snake_len.begin(), snake_len.end(), 1);
}

void BatchedEnv::reset(const uint8_t* mask) {
    for (int i = 0; i < N; ++i) {
        if (mask[i]) {
            reward[i] = 0.f;
            terminated[i] = 0;
            truncated[i] = 0;
            head_x[i] = 0.f;
            head_y[i] = 0.f;
            dir_angle[i] = 0.f;
            snake_len[i] = 1;
            if (obs_dim >= 1) {
                obs[i * obs_dim + 0] = head_x[i];
            }
        }
    }
}

void BatchedEnv::step(const float* actions) {
  // Minimal placeholder dynamics: move along x by action * dt.
  // Extend to full snake logic later.
  for (int i = 0; i < N; ++i) {
    if (terminated[i] || truncated[i]) continue;
    // Interpret action as angle in radians. Wrap into [0, 2π) defensively.
    float a = actions[i * act_dim + 0];
    if (a >= kTwoPi || a < 0.0f) {
      a = std::fmod(a, kTwoPi);
      if (a < 0.0f) a += kTwoPi;
    }
    // Move with unit speed scaled by dt in the direction of 'a'.
    head_x[i] += std::cos(a) * step_size;
    head_y[i] += std::sin(a) * step_size;
    reward[i] = 0.0f; // placeholder reward
    if (obs_dim >= 1) {
      obs[i * obs_dim + 0] = head_x[i];
    }
  }
}