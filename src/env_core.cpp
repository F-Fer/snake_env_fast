#include "env_core.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cstdint>

// Simple xorshift64* RNG
static inline uint64_t xorshift64star(uint64_t& s) {
  s ^= s >> 12; // a
  s ^= s << 25; // b
  s ^= s >> 27; // c
  return s * 2685821657736338717ULL;
}

static inline float rand_uniform_01(uint64_t& s) {
  // Convert top 24 bits to float in [0,1)
  uint64_t r = xorshift64star(s);
  const uint32_t mant = static_cast<uint32_t>((r >> 40) & 0xFFFFFFu);
  return static_cast<float>(mant) / static_cast<float>(0x1000000u);
}

static inline float wrap_angle_0_2pi(float a) {
  if (a >= kTwoPi || a < 0.0f) {
    a = std::fmod(a, kTwoPi);
    if (a < 0.0f) a += kTwoPi;
  }
  return a;
}

static inline float wrap_coord_0_map(float x, int map_size) {
  float m = static_cast<float>(map_size);
  if (x >= m || x < 0.0f) {
    x = std::fmod(x, m);
    if (x < 0.0f) x += m;
  }
  return x;
}

BatchedEnv::BatchedEnv(int num_envs, RenderMode mode, int map_size, int step_size, int max_steps, float max_turn, float eat_radius, unsigned long long seed)
  : N(num_envs),
    obs_dim(mode == RenderMode::Headless ? static_cast<int>(ObservationSize::Headless) : throw std::runtime_error("RenderMode not supported")),
    act_dim(1),
    map_size(map_size),
    step_size(step_size),
    max_steps(max_steps),
    max_turn(max_turn),
    eat_radius(eat_radius),
    single_observation_space(mode == RenderMode::Headless ? BoxSpace(-INFINITY, INFINITY, {static_cast<int>(ObservationSize::Headless)}, "float32") : BoxSpace(0, 0, {0}, "float32")),
    single_action_space(-max_turn, max_turn, {1}, "float32"),
    render_mode(mode),
    obs(N * (mode == RenderMode::Headless ? static_cast<int>(ObservationSize::Headless) : 0), 0.f),
    reward(N, 0.f),
    terminated(N, 0),
    truncated(N, 0),
    head_x(N, 0.f),
    head_y(N, 0.f),
    dir_angle(N, 0.f),
    snake_len(N, 1),
    food_x(N, 0.f),
    food_y(N, 0.f),
    steps_since_reset(N, 0),
    rng_state(N, 0ULL)
{
    set_seed(seed);
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
  std::fill(food_x.begin(), food_x.end(), 0.f);
  std::fill(food_y.begin(), food_y.end(), 0.f);
  std::fill(steps_since_reset.begin(), steps_since_reset.end(), 0);

  // Initialize each env with randomized head, angle, and food
  for (int i = 0; i < N; ++i) {
    float rx = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i]));
    float ry = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i]));
    float ra = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i]));
    head_x[i] = rx * static_cast<float>(map_size);
    head_y[i] = ry * static_cast<float>(map_size);
    dir_angle[i] = ra * kTwoPi;
    // Place food somewhere else
    float fx = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i]));
    float fy = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i]));
    food_x[i] = fx * static_cast<float>(map_size);
    food_y[i] = fy * static_cast<float>(map_size);

    // Fill observation (7D)
    if (obs_dim >= 1) {
      const int base = i * obs_dim;
      const float dx = food_x[i] - head_x[i];
      const float dy = food_y[i] - head_y[i];
      const float dist = std::sqrt(dx*dx + dy*dy);
      obs[base + 0] = head_x[i];
      obs[base + 1] = head_y[i];
      obs[base + 2] = dir_angle[i];
      obs[base + 3] = static_cast<float>(snake_len[i]);
      obs[base + 4] = food_x[i];
      obs[base + 5] = food_y[i];
      obs[base + 6] = dist;
    }
  }
}

void BatchedEnv::reset(const uint8_t* mask) {
    for (int i = 0; i < N; ++i) {
        if (mask[i]) {
            reward[i] = 0.f;
            terminated[i] = 0;
            truncated[i] = 0;
            steps_since_reset[i] = 0;
            // Randomize head and angle
            float rx = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i]));
            float ry = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i]));
            float ra = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i]));
            head_x[i] = rx * static_cast<float>(map_size);
            head_y[i] = ry * static_cast<float>(map_size);
            dir_angle[i] = ra * kTwoPi;
            snake_len[i] = 1;
            // Respawn food
            float fx = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i]));
            float fy = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i]));
            food_x[i] = fx * static_cast<float>(map_size);
            food_y[i] = fy * static_cast<float>(map_size);

            if (obs_dim >= 1) {
                const int base = i * obs_dim;
                const float dx = food_x[i] - head_x[i];
                const float dy = food_y[i] - head_y[i];
                const float dist = std::sqrt(dx*dx + dy*dy);
                obs[base + 0] = head_x[i];
                obs[base + 1] = head_y[i];
                obs[base + 2] = dir_angle[i];
                obs[base + 3] = static_cast<float>(snake_len[i]);
                obs[base + 4] = food_x[i];
                obs[base + 5] = food_y[i];
                obs[base + 6] = dist;
            }
        }
    }
}

void BatchedEnv::step(const float* actions) {
  for (int i = 0; i < N; ++i) {
    if (terminated[i] || truncated[i]) continue;

    // Steering: clamp delta turn and update heading
    float dtheta = actions[i * act_dim + 0];
    if (dtheta > max_turn) dtheta = max_turn;
    if (dtheta < -max_turn) dtheta = -max_turn;
    dir_angle[i] = wrap_angle_0_2pi(dir_angle[i] + dtheta);

    // Move forward
    head_x[i] = wrap_coord_0_map(head_x[i] + std::cos(dir_angle[i]) * static_cast<float>(step_size), map_size);
    head_y[i] = wrap_coord_0_map(head_y[i] + std::sin(dir_angle[i]) * static_cast<float>(step_size), map_size);

    // Reward and eating logic
    reward[i] = 0.0f;
    float dx = food_x[i] - head_x[i];
    float dy = food_y[i] - head_y[i];
    float dist = std::sqrt(dx*dx + dy*dy);
    if (dist <= eat_radius) {
      reward[i] = 1.0f;
      // Respawn food randomly
      float fx = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i]));
      float fy = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i]));
      food_x[i] = fx * static_cast<float>(map_size);
      food_y[i] = fy * static_cast<float>(map_size);
      // Recompute new distance to new food for obs
      dx = food_x[i] - head_x[i];
      dy = food_y[i] - head_y[i];
      dist = std::sqrt(dx*dx + dy*dy);
    }

    // Steps and truncation
    steps_since_reset[i] += 1;
    if (steps_since_reset[i] >= max_steps) {
      truncated[i] = 1;
    }

    // Write observation
    if (obs_dim >= 1) {
      const int base = i * obs_dim;
      obs[base + 0] = head_x[i];
      obs[base + 1] = head_y[i];
      obs[base + 2] = dir_angle[i];
      obs[base + 3] = static_cast<float>(snake_len[i]);
      obs[base + 4] = food_x[i];
      obs[base + 5] = food_y[i];
      obs[base + 6] = dist;
    }
  }
}

void BatchedEnv::set_seed(unsigned long long seed) {
  // Distinct non-zero seeds per env
  const uint64_t golden = 0x9E3779B97F4A7C15ULL; // splitmix64 constant
  for (int i = 0; i < N; ++i) {
    uint64_t s = static_cast<uint64_t>(seed) + golden * static_cast<uint64_t>(i + 1);
    if (s == 0ULL) s = golden; // avoid zero state
    rng_state[i] = s;
  }
}