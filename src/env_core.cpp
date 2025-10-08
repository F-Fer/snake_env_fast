#include "env_core.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <limits>

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

BatchedEnv::BatchedEnv(int num_envs, RenderMode mode, int map_size, int step_size, int max_steps, float max_turn, float eat_radius, unsigned long long seed, int max_segments, int initial_segments, float segment_radius, float min_segment_distance, float cell_size, int num_bots, int max_bot_segments, int num_food)
  : N(num_envs),
    // Support both Headless and RGB with the same observation layout for now
    obs_dim(static_cast<int>(ObservationSize::Headless)),
    act_dim(1),
    map_size(map_size),
    step_size(step_size),
    max_steps(max_steps),
    max_turn(max_turn),
    eat_radius(eat_radius),
    max_segments(max_segments),
    initial_segments(initial_segments),
    segment_radius(segment_radius),
    min_segment_distance(min_segment_distance),
    cell_size(cell_size),
    grid_w(static_cast<int>(std::ceil(static_cast<float>(map_size) / cell_size))),
    grid_h(static_cast<int>(std::ceil(static_cast<float>(map_size) / cell_size))),
    num_bots(num_bots),
    max_bot_segments(max_bot_segments),
    num_food(std::max(0, num_food)),
    single_observation_space(BoxSpace(-INFINITY, INFINITY, {static_cast<int>(ObservationSize::Headless)}, "float32")),
    single_action_space(-max_turn, max_turn, {1}, "float32"),
    render_mode(mode),
    obs(N * static_cast<int>(ObservationSize::Headless), 0.f),
    reward(N, 0.f),
    terminated(N, 0),
    truncated(N, 0),
    dir_angle(N, 0.f),
    snake_len(N, 1),
    food_x(static_cast<size_t>(N) * static_cast<size_t>(num_food), 0.f),
    food_y(static_cast<size_t>(N) * static_cast<size_t>(num_food), 0.f),
    steps_since_reset(N, 0),
    rgb_image(mode == RenderMode::RGB ? N * 84 * 84 * 3 : 0, 0),
    rng_state(N, 0ULL),
    segments_x(static_cast<size_t>(N) * static_cast<size_t>(max_segments), 0.f),
    segments_y(static_cast<size_t>(N) * static_cast<size_t>(max_segments), 0.f),
    segments_count(N, 0),
    pending_growth(N, 0),
    bot_segments_x(static_cast<size_t>(N) * static_cast<size_t>(num_bots) * static_cast<size_t>(max_bot_segments), 0.f),
    bot_segments_y(static_cast<size_t>(N) * static_cast<size_t>(num_bots) * static_cast<size_t>(max_bot_segments), 0.f),
    bot_segments_count(static_cast<size_t>(N) * static_cast<size_t>(num_bots), 0),
    bot_pending_growth(static_cast<size_t>(N) * static_cast<size_t>(num_bots), 0),
    bot_dir_angle(static_cast<size_t>(N) * static_cast<size_t>(num_bots), 0.f),
    bot_alive(static_cast<size_t>(N) * static_cast<size_t>(num_bots), 1),
    grid(static_cast<size_t>(N) * static_cast<size_t>(grid_w) * static_cast<size_t>(grid_h), -1)
{
    set_seed(seed);
    full_reset();
}

void BatchedEnv::place_food(int env_idx, int food_slot) {
  if (num_food <= 0) {
      return;
  }
  const int cell_base = env_idx * grid_w * grid_h;
  const int food_base = env_idx * num_food;

  float fx = 0.f, fy = 0.f;
  int cx = 0, cy = 0;
  bool placed = false;
  for (int tries = 0; tries < 128 && !placed; ++tries) {
      fx = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[env_idx])) * static_cast<float>(map_size);
      fy = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[env_idx])) * static_cast<float>(map_size);
      cx = static_cast<int>(fx / cell_size);
      cy = static_cast<int>(fy / cell_size);
      if (cx < 0) cx = 0; else if (cx >= grid_w) cx = grid_w - 1;
      if (cy < 0) cy = 0; else if (cy >= grid_h) cy = grid_h - 1;
      if (grid[cell_base + cy * grid_w + cx] != -1) continue;
      placed = true;
  }
  if (!placed) {
      fx = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[env_idx])) * static_cast<float>(map_size);
      fy = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[env_idx])) * static_cast<float>(map_size);
      cx = static_cast<int>(fx / cell_size);
      cy = static_cast<int>(fy / cell_size);
      if (cx < 0) cx = 0; else if (cx >= grid_w) cx = grid_w - 1;
      if (cy < 0) cy = 0; else if (cy >= grid_h) cy = grid_h - 1;
  }
  food_x[food_base + food_slot] = fx;
  food_y[food_base + food_slot] = fy;
  grid[cell_base + cy * grid_w + cx] = 0;
}

void BatchedEnv::respawn_bot(int env_idx, int bot_idx) {
  const int global_bot_idx = env_idx * num_bots + bot_idx;
  const int bot_base_seg = global_bot_idx * max_bot_segments;
  bot_alive[global_bot_idx] = 1;
  bot_pending_growth[global_bot_idx] = 0;

  float brx = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[env_idx]));
  float bry = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[env_idx]));
  float bra = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[env_idx]));
  float bhx = brx * static_cast<float>(map_size);
  float bhy = bry * static_cast<float>(map_size);
  bot_dir_angle[global_bot_idx] = bra * kTwoPi;

  const int bot_initial_segs = std::min(3, max_bot_segments);
  bot_segments_count[global_bot_idx] = bot_initial_segs;
  bot_segments_x[bot_base_seg + 0] = bhx;
  bot_segments_y[bot_base_seg + 0] = bhy;
  for (int s = 1; s < bot_initial_segs; ++s) {
    float offset = static_cast<float>(s) * min_segment_distance;
    float ax = std::cos(bot_dir_angle[global_bot_idx] + kPi) * offset;
    float ay = std::sin(bot_dir_angle[global_bot_idx] + kPi) * offset;
    bot_segments_x[bot_base_seg + s] = bhx + ax;
    bot_segments_y[bot_base_seg + s] = bhy + ay;
  }
}

void BatchedEnv::debug_set_player_state(int env_idx, const std::vector<float>& xs, const std::vector<float>& ys, float angle) {
  if (env_idx < 0 || env_idx >= N) {
    throw std::out_of_range("debug_set_player_state env_idx out of range");
  }
  if (xs.size() != ys.size() || xs.empty()) {
    throw std::invalid_argument("debug_set_player_state requires matching non-empty coordinate arrays");
  }
  dir_angle[env_idx] = angle;
  segments_count[env_idx] = static_cast<int>(std::min(xs.size(), static_cast<size_t>(max_segments)));
  const int base_seg = env_idx * max_segments;
  for (int i = 0; i < segments_count[env_idx]; ++i) {
    segments_x[base_seg + i] = xs[i];
    segments_y[base_seg + i] = ys[i];
  }
}

void BatchedEnv::debug_set_bot_state(int env_idx, int bot_idx, const std::vector<float>& xs, const std::vector<float>& ys, float angle, bool alive) {
  if (env_idx < 0 || env_idx >= N) {
    throw std::out_of_range("debug_set_bot_state env_idx out of range");
  }
  if (bot_idx < 0 || bot_idx >= num_bots) {
    throw std::out_of_range("debug_set_bot_state bot_idx out of range");
  }
  if (xs.size() != ys.size() || xs.empty()) {
    throw std::invalid_argument("debug_set_bot_state requires matching non-empty coordinate arrays");
  }
  const int global_bot_idx = env_idx * num_bots + bot_idx;
  bot_alive[global_bot_idx] = alive ? 1 : 0;
  bot_dir_angle[global_bot_idx] = angle;
  bot_segments_count[global_bot_idx] = static_cast<int>(std::min(xs.size(), static_cast<size_t>(max_bot_segments)));
  const int base = global_bot_idx * max_bot_segments;
  for (int i = 0; i < bot_segments_count[global_bot_idx]; ++i) {
    bot_segments_x[base + i] = xs[i];
    bot_segments_y[base + i] = ys[i];
  }
}

void BatchedEnv::debug_rebuild_spatial_hash(int env_idx) {
    if (env_idx < 0 || env_idx >= N) {
        throw std::out_of_range("debug_rebuild_spatial_hash env_idx out of range");
    }
    const int cell_base = env_idx * grid_w * grid_h;
    const int base_seg = env_idx * max_segments;
    std::fill(grid.begin() + cell_base, grid.begin() + cell_base + grid_w * grid_h, -1);
    for (int s = 0; s < segments_count[env_idx]; ++s) {
        int cx = static_cast<int>(segments_x[base_seg + s] / cell_size);
        int cy = static_cast<int>(segments_y[base_seg + s] / cell_size);
        if (cx < 0) cx = 0; else if (cx >= grid_w) cx = grid_w - 1;
        if (cy < 0) cy = 0; else if (cy >= grid_h) cy = grid_h - 1;
        grid[cell_base + cy * grid_w + cx] = 1;
    }
    for (int b = 0; b < num_bots; ++b) {
        const int global_bot_idx = env_idx * num_bots + b;
        if (!bot_alive[global_bot_idx]) continue;
        const int bot_base = global_bot_idx * max_bot_segments;
        for (int s = 0; s < bot_segments_count[global_bot_idx]; ++s) {
            int cx = static_cast<int>(bot_segments_x[bot_base + s] / cell_size);
            int cy = static_cast<int>(bot_segments_y[bot_base + s] / cell_size);
            if (cx < 0) cx = 0; else if (cx >= grid_w) cx = grid_w - 1;
            if (cy < 0) cy = 0; else if (cy >= grid_h) cy = grid_h - 1;
            grid[cell_base + cy * grid_w + cx] = 2 + b;
        }
    }
}

void BatchedEnv::full_reset() {
  std::fill(obs.begin(), obs.end(), 0.f);
  std::fill(reward.begin(), reward.end(), 0.f);
  std::fill(terminated.begin(), terminated.end(), 0);
  std::fill(truncated.begin(), truncated.end(), 0);
  std::fill(dir_angle.begin(), dir_angle.end(), 0.f);
  std::fill(snake_len.begin(), snake_len.end(), initial_segments);
  std::fill(food_x.begin(), food_x.end(), 0.f);
  std::fill(food_y.begin(), food_y.end(), 0.f);
  std::fill(steps_since_reset.begin(), steps_since_reset.end(), 0);
  std::fill(segments_count.begin(), segments_count.end(), 0);
  std::fill(pending_growth.begin(), pending_growth.end(), 0);
  std::fill(grid.begin(), grid.end(), -1);
  std::fill(bot_segments_count.begin(), bot_segments_count.end(), 0);
  std::fill(bot_pending_growth.begin(), bot_pending_growth.end(), 0);
  std::fill(bot_dir_angle.begin(), bot_dir_angle.end(), 0.f);
  std::fill(bot_alive.begin(), bot_alive.end(), 1);

  // Initialize each env with randomized head, bots, and food
  for (int i = 0; i < N; ++i) {
    float rx = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i]));
    float ry = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i]));
    float ra = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i]));
    float hx = rx * static_cast<float>(map_size);
    float hy = ry * static_cast<float>(map_size);
    dir_angle[i] = ra * kTwoPi;

    // Initialize segments behind head
    const int base_seg = i * max_segments;
    segments_count[i] = std::min(initial_segments, max_segments);
    segments_x[base_seg + 0] = hx;
    segments_y[base_seg + 0] = hy;
    for (int s = 1; s < segments_count[i]; ++s) {
      float offset = static_cast<float>(s) * min_segment_distance;
      float ax = std::cos(dir_angle[i] + kPi) * offset;
      float ay = std::sin(dir_angle[i] + kPi) * offset;
      segments_x[base_seg + s] = hx + ax;
      segments_y[base_seg + s] = hy + ay;
    }

    // Clear and populate occupancy grid for this env
    const int cell_base = i * grid_w * grid_h;
    std::fill(grid.begin() + cell_base, grid.begin() + cell_base + grid_w * grid_h, -1);
    for (int s = 0; s < segments_count[i]; ++s) {
      int cx = static_cast<int>(segments_x[base_seg + s] / cell_size);
      int cy = static_cast<int>(segments_y[base_seg + s] / cell_size);
      if (cx < 0) cx = 0; if (cx >= grid_w) cx = grid_w - 1;
      if (cy < 0) cy = 0; if (cy >= grid_h) cy = grid_h - 1;
      grid[cell_base + cy * grid_w + cx] = 1;
    }
    for (int b = 0; b < num_bots; ++b) {
      respawn_bot(i, b);
      const int bot_idx = i * num_bots + b;
      const int bot_base_seg = bot_idx * max_bot_segments;
      const int bot_cell_id = 2 + b;
      for (int s = 0; s < bot_segments_count[bot_idx]; ++s) {
        int bcx = static_cast<int>(bot_segments_x[bot_base_seg + s] / cell_size);
        int bcy = static_cast<int>(bot_segments_y[bot_base_seg + s] / cell_size);
        if (bcx < 0) bcx = 0; if (bcx >= grid_w) bcx = grid_w - 1;
        if (bcy < 0) bcy = 0; if (bcy >= grid_h) bcy = grid_h - 1;
        grid[cell_base + bcy * grid_w + bcx] = bot_cell_id;
      }
    }
    for (int f = 0; f < num_food; ++f) {
      place_food(i, f);
    }

    // Fill observation (7D)
    if (obs_dim >= 1) {
      const int base = i * obs_dim;
      const int base_seg = i * max_segments;
      const float hx = segments_x[base_seg + 0];
      const float hy = segments_y[base_seg + 0];
      float nearest_fx = 0.f;
      float nearest_fy = 0.f;
      float nearest_dist = 0.f;
      if (num_food > 0) {
        nearest_dist = std::numeric_limits<float>::infinity();
        const int food_base = i * num_food;
        for (int f = 0; f < num_food; ++f) {
          const float fx = food_x[food_base + f];
          const float fy = food_y[food_base + f];
          const float dfx = fx - hx;
          const float dfy = fy - hy;
          const float d = std::sqrt(dfx*dfx + dfy*dfy);
          if (d < nearest_dist) {
            nearest_dist = d;
            nearest_fx = fx;
            nearest_fy = fy;
          }
        }
        if (!std::isfinite(nearest_dist)) {
          nearest_dist = 0.f;
          nearest_fx = hx;
          nearest_fy = hy;
        }
      }
      obs[base + 0] = hx;
      obs[base + 1] = hy;
      obs[base + 2] = dir_angle[i];
      obs[base + 3] = static_cast<float>(segments_count[i]);
      obs[base + 4] = nearest_fx;
      obs[base + 5] = nearest_fy;
      obs[base + 6] = nearest_dist;
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
      float hx = rx * static_cast<float>(map_size);
      float hy = ry * static_cast<float>(map_size);
      dir_angle[i] = ra * kTwoPi;
      pending_growth[i] = 0;
      // Reset player occupancy
      const int base_seg = i * max_segments;
      segments_count[i] = std::min(initial_segments, max_segments);
      segments_x[base_seg + 0] = hx;
      segments_y[base_seg + 0] = hy;
      for (int s = 1; s < segments_count[i]; ++s) {
        float offset = static_cast<float>(s) * min_segment_distance;
        float ax = (s == 0 ? 0.0f : std::cos(dir_angle[i] + kPi) * offset);
        float ay = (s == 0 ? 0.0f : std::sin(dir_angle[i] + kPi) * offset);
        segments_x[base_seg + s] = hx + ax;
        segments_y[base_seg + s] = hy + ay;
      }
      const int cell_base = i * grid_w * grid_h;
      std::fill(grid.begin() + cell_base, grid.begin() + cell_base + grid_w * grid_h, -1);
      for (int s = 0; s < segments_count[i]; ++s) {
        int cx = static_cast<int>(segments_x[base_seg + s] / cell_size);
        int cy = static_cast<int>(segments_y[base_seg + s] / cell_size);
        if (cx < 0) cx = 0; if (cx >= grid_w) cx = grid_w - 1;
        if (cy < 0) cy = 0; if (cy >= grid_h) cy = grid_h - 1;
        grid[cell_base + cy * grid_w + cx] = 1;
      }
      for (int b = 0; b < num_bots; ++b) {
        respawn_bot(i, b);
        const int bot_idx = i * num_bots + b;
        const int bot_base_seg = bot_idx * max_bot_segments;
        const int bot_cell_id = 2 + b;
        for (int s = 0; s < bot_segments_count[bot_idx]; ++s) {
          int bcx = static_cast<int>(bot_segments_x[bot_base_seg + s] / cell_size);
          int bcy = static_cast<int>(bot_segments_y[bot_base_seg + s] / cell_size);
          if (bcx < 0) bcx = 0; if (bcx >= grid_w) bcx = grid_w - 1;
          if (bcy < 0) bcy = 0; if (bcy >= grid_h) bcy = grid_h - 1;
          grid[cell_base + bcy * grid_w + bcx] = bot_cell_id;
        }
      }

      if (obs_dim >= 1) {
          const int base = i * obs_dim;
          const float hx = segments_x[base_seg + 0];
          const float hy = segments_y[base_seg + 0];
          float nearest_fx = hx;
          float nearest_fy = hy;
          float nearest_dist = 0.f;
          if (num_food > 0) {
              nearest_dist = std::numeric_limits<float>::infinity();
              const int food_base = i * num_food;
              for (int f = 0; f < num_food; ++f) {
                  const float fx = food_x[food_base + f];
                  const float fy = food_y[food_base + f];
                  const float dfx = fx - hx;
                  const float dfy = fy - hy;
                  const float d = std::sqrt(dfx*dfx + dfy*dfy);
                  if (d < nearest_dist) {
                      nearest_dist = d;
                      nearest_fx = fx;
                      nearest_fy = fy;
                  }
              }
              if (!std::isfinite(nearest_dist)) {
                  nearest_dist = 0.f;
                  nearest_fx = hx;
                  nearest_fy = hy;
              }
          }
          obs[base + 0] = hx;
          obs[base + 1] = hy;
          obs[base + 2] = dir_angle[i];
          obs[base + 3] = static_cast<float>(segments_count[i]);
          obs[base + 4] = nearest_fx;
          obs[base + 5] = nearest_fy;
          obs[base + 6] = nearest_dist;
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

    // Move head forward (segments[0])
    const int base_seg = i * max_segments;
    float hx = segments_x[base_seg + 0];
    float hy = segments_y[base_seg + 0];
    hx = hx + std::cos(dir_angle[i]) * static_cast<float>(step_size);
    hy = hy + std::sin(dir_angle[i]) * static_cast<float>(step_size);
    // Check if head is out of bounds
    if (hx < 0 || hx >= map_size || hy < 0 || hy >= map_size) {
      terminated[i] = 1;
      truncated[i] = 1;
      continue;
    }
    // Update segments[0]
    segments_x[base_seg + 0] = hx;
    segments_y[base_seg + 0] = hy;

    // Follow pass for body segments
    const int segs = segments_count[i]; // :)
    for (int s = 1; s < segs; ++s) {
      float tx = segments_x[base_seg + (s - 1)];
      float ty = segments_y[base_seg + (s - 1)];
      float cx = segments_x[base_seg + s];
      float cy = segments_y[base_seg + s];
      float dxs = tx - cx;
      float dys = ty - cy;
      float d = std::sqrt(dxs*dxs + dys*dys);
      if (d > min_segment_distance && d > 0.0f) {
        float move_ratio = static_cast<float>(step_size) / d;
        segments_x[base_seg + s] = cx + dxs * move_ratio;
        segments_y[base_seg + s] = cy + dys * move_ratio;
      }
    }

    // Update player occupancy in grid
    const int cell_base = i * grid_w * grid_h;
    for (int s = 0; s < segs; ++s) {
      int cx = static_cast<int>(segments_x[base_seg + s] / cell_size);
      int cy = static_cast<int>(segments_y[base_seg + s] / cell_size);
      if (cx < 0) cx = 0; if (cx >= grid_w) cx = grid_w - 1;
      if (cy < 0) cy = 0; if (cy >= grid_h) cy = grid_h - 1;
      grid[cell_base + cy * grid_w + cx] = 1;
    }

    // Reward and eating logic
    reward[i] = 0.0f;
    const int food_base = i * num_food;
    int nearest_food_idx = -1;
    float nearest_fx = hx;
    float nearest_fy = hy;
    float nearest_dist = 0.f;
    if (num_food > 0) {
      float best = std::numeric_limits<float>::infinity();
      for (int f = 0; f < num_food; ++f) {
      const float fx = food_x[food_base + f];
      const float fy = food_y[food_base + f];
      const float dfx = fx - hx;
      const float dfy = fy - hy;
      const float d = std::sqrt(dfx*dfx + dfy*dfy);
      if (d < best) {
        best = d;
        nearest_food_idx = f;
        nearest_fx = fx;
        nearest_fy = fy;
        nearest_dist = d;
      }
      }
    }
    if (nearest_food_idx >= 0 && nearest_dist <= eat_radius) {
      reward[i] = 1.0f;
      pending_growth[i] += 1;
      place_food(i, nearest_food_idx);
      nearest_dist = 0.f;
    }

    // Apply growth one segment per step
    if (pending_growth[i] > 0 && segments_count[i] < max_segments) {
      int sc = segments_count[i];
      float lx = segments_x[base_seg + (sc - 1)];
      float ly = segments_y[base_seg + (sc - 1)];
      float lx2 = (sc >= 2) ? segments_x[base_seg + (sc - 2)] : (lx - std::cos(dir_angle[i]) * min_segment_distance);
      float ly2 = (sc >= 2) ? segments_y[base_seg + (sc - 2)] : (ly - std::sin(dir_angle[i]) * min_segment_distance);
      float tx = lx - lx2;
      float ty = ly - ly2;
      float td = std::sqrt(tx*tx + ty*ty);
      if (td > 0.0f) { tx /= td; ty /= td; }
      segments_x[base_seg + sc] = lx + tx * min_segment_distance;
      segments_y[base_seg + sc] = ly + ty * min_segment_distance;
      segments_count[i] = sc + 1;
      pending_growth[i] -= 1;
    }

    // Bot snakes: AI and movement
    for (int s = 0; s < segments_count[i]; ++s) {
      int cx = static_cast<int>(segments_x[base_seg + s] / cell_size);
      int cy = static_cast<int>(segments_y[base_seg + s] / cell_size);
      if (cx < 0) cx = 0; if (cx >= grid_w) cx = grid_w - 1;
      if (cy < 0) cy = 0; if (cy >= grid_h) cy = grid_h - 1;
      grid[cell_base + cy * grid_w + cx] = 1;
    }
    
    // For each bot snake
    for (int b = 0; b < num_bots; ++b) {
      const int bot_idx = i * num_bots + b;
      if (!bot_alive[bot_idx]) {
        respawn_bot(i, b);
      }

      const int bot_base_seg = bot_idx * max_bot_segments;
      float bot_hx = bot_segments_x[bot_base_seg + 0];
      float bot_hy = bot_segments_y[bot_base_seg + 0];
      
      // Simple AI: turn towards nearest food
      int bot_target_idx = -1;
      float bot_target_fx = bot_hx;
      float bot_target_fy = bot_hy;
      if (num_food > 0) {
        float best = std::numeric_limits<float>::infinity();
        const int food_base = i * num_food;
        for (int f = 0; f < num_food; ++f) {
          const float fx = food_x[food_base + f];
          const float fy = food_y[food_base + f];
          const float dfx = fx - bot_hx;
          const float dfy = fy - bot_hy;
          const float d = std::sqrt(dfx*dfx + dfy*dfy);
          if (d < best) {
            best = d;
            bot_target_idx = f;
            bot_target_fx = fx;
            bot_target_fy = fy;
          }
        }
      }
      float bot_dx = bot_target_fx - bot_hx;
      float bot_dy = bot_target_fy - bot_hy;
      float target_angle = (bot_target_idx >= 0) ? std::atan2(bot_dy, bot_dx) : bot_dir_angle[bot_idx];
      float angle_diff = target_angle - bot_dir_angle[bot_idx];
      // Normalize angle difference to [-pi, pi]
      while (angle_diff > kPi) angle_diff -= kTwoPi;
      while (angle_diff < -kPi) angle_diff += kTwoPi;
      // Clamp turn
      float bot_turn = angle_diff;
      if (bot_turn > max_turn * 0.7f) bot_turn = max_turn * 0.7f;
      if (bot_turn < -max_turn * 0.7f) bot_turn = -max_turn * 0.7f;
      bot_dir_angle[bot_idx] = wrap_angle_0_2pi(bot_dir_angle[bot_idx] + bot_turn);
      
      // Move bot head
      bot_hx = bot_hx + std::cos(bot_dir_angle[bot_idx]) * static_cast<float>(step_size);
      bot_hy = bot_hy + std::sin(bot_dir_angle[bot_idx]) * static_cast<float>(step_size);
      
      const int bot_segs = bot_segments_count[bot_idx];
      // Check bounds
      if (bot_hx < 0 || bot_hx >= map_size || bot_hy < 0 || bot_hy >= map_size) {
        // Clear old cells for this bot
        for (int s = 0; s < bot_segs; ++s) {
          int bcx = static_cast<int>(bot_segments_x[bot_base_seg + s] / cell_size);
          int bcy = static_cast<int>(bot_segments_y[bot_base_seg + s] / cell_size);
          if (bcx < 0) bcx = 0; if (bcx >= grid_w) bcx = grid_w - 1;
          if (bcy < 0) bcy = 0; if (bcy >= grid_h) bcy = grid_h - 1;
          grid[cell_base + bcy * grid_w + bcx] = -1;
        }
        bot_alive[bot_idx] = 0;
        continue;
      }
      
      bot_segments_x[bot_base_seg + 0] = bot_hx;
      bot_segments_y[bot_base_seg + 0] = bot_hy;
      
      // Clear old bot cells
      for (int s = 0; s < bot_segs; ++s) {
        int bcx = static_cast<int>(bot_segments_x[bot_base_seg + s] / cell_size);
        int bcy = static_cast<int>(bot_segments_y[bot_base_seg + s] / cell_size);
        if (bcx < 0) bcx = 0; if (bcx >= grid_w) bcx = grid_w - 1;
        if (bcy < 0) bcy = 0; if (bcy >= grid_h) bcy = grid_h - 1;
        grid[cell_base + bcy * grid_w + bcx] = -1;
      }

      // Follow segments
      for (int s = 1; s < bot_segs; ++s) {
        float tx = bot_segments_x[bot_base_seg + (s - 1)];
        float ty = bot_segments_y[bot_base_seg + (s - 1)];
        float cx = bot_segments_x[bot_base_seg + s];
        float cy = bot_segments_y[bot_base_seg + s];
        float dxs = tx - cx;
        float dys = ty - cy;
        float d = std::sqrt(dxs*dxs + dys*dys);
        if (d > min_segment_distance && d > 0.0f) {
          float move_ratio = static_cast<float>(step_size) / d;
          bot_segments_x[bot_base_seg + s] = cx + dxs * move_ratio;
          bot_segments_y[bot_base_seg + s] = cy + dys * move_ratio;
        }
      }

      // Bot vs bot collision detection via grid
      int bot_head_cx = static_cast<int>(bot_hx / cell_size);
      int bot_head_cy = static_cast<int>(bot_hy / cell_size);
      if (bot_head_cx < 0) bot_head_cx = 0; if (bot_head_cx >= grid_w) bot_head_cx = grid_w - 1;
      if (bot_head_cy < 0) bot_head_cy = 0; if (bot_head_cy >= grid_h) bot_head_cy = grid_h - 1;
      int occupant = grid[cell_base + bot_head_cy * grid_w + bot_head_cx];
      if (occupant == 1) {
        bot_alive[bot_idx] = 0;
        continue;
      } else if (occupant >= 2 && occupant != (2 + b)) {
        int other_bot_idx = (occupant - 2) + i * num_bots;
        bot_alive[bot_idx] = 0;
        bot_alive[other_bot_idx] = 0;
        continue;
      }
      
      // Occupancy grid update for bot
      for (int s = 0; s < bot_segs; ++s) {
        int bcx = static_cast<int>(bot_segments_x[bot_base_seg + s] / cell_size);
        int bcy = static_cast<int>(bot_segments_y[bot_base_seg + s] / cell_size);
        if (bcx < 0) bcx = 0; if (bcx >= grid_w) bcx = grid_w - 1;
        if (bcy < 0) bcy = 0; if (bcy >= grid_h) bcy = grid_h - 1;
        grid[cell_base + bcy * grid_w + bcx] = 2 + b;
      }

      // Bot eating logic (can steal player's food)
      if (num_food > 0) {
        bool ate = false;
        for (int f = 0; f < num_food && !ate; ++f) {
          const float dfx = food_x[food_base + f] - bot_hx;
          const float dfy = food_y[food_base + f] - bot_hy;
          if (std::sqrt(dfx*dfx + dfy*dfy) <= eat_radius) {
            bot_pending_growth[bot_idx] += 1;
            place_food(i, f);
            ate = true;
          }
        }
      }
      
      // Bot growth
      if (bot_pending_growth[bot_idx] > 0 && bot_segments_count[bot_idx] < max_bot_segments) {
        int sc = bot_segments_count[bot_idx];
        float lx = bot_segments_x[bot_base_seg + (sc - 1)];
        float ly = bot_segments_y[bot_base_seg + (sc - 1)];
        float lx2 = (sc >= 2) ? bot_segments_x[bot_base_seg + (sc - 2)] : (lx - std::cos(bot_dir_angle[bot_idx]) * min_segment_distance);
        float ly2 = (sc >= 2) ? bot_segments_y[bot_base_seg + (sc - 2)] : (ly - std::sin(bot_dir_angle[bot_idx]) * min_segment_distance);
        float tx = lx - lx2;
        float ty = ly - ly2;
        float td = std::sqrt(tx*tx + ty*ty);
        if (td > 0.0f) { tx /= td; ty /= td; }
        bot_segments_x[bot_base_seg + sc] = lx + tx * min_segment_distance;
        bot_segments_y[bot_base_seg + sc] = ly + ty * min_segment_distance;
        bot_segments_count[bot_idx] = sc + 1;
        bot_pending_growth[bot_idx] -= 1;
      }
    }
    
    // Check collision between player head and bot snakes
    int head_cx = static_cast<int>(hx / cell_size);
    int head_cy = static_cast<int>(hy / cell_size);
    for (int oy = -1; oy <= 1 && !terminated[i]; ++oy) {
      for (int ox = -1; ox <= 1 && !terminated[i]; ++ox) {
        int ncx = head_cx + ox, ncy = head_cy + oy;
        if (ncx < 0 || ncx >= grid_w || ncy < 0 || ncy >= grid_h) continue;
        int occupant = grid[cell_base + ncy * grid_w + ncx];
        if (occupant >= 2) {
          terminated[i] = 1;
          int hit_bot_local = occupant - 2;
          if (hit_bot_local >= 0 && hit_bot_local < num_bots) {
            bot_alive[i * num_bots + hit_bot_local] = 0;
          }
          break;
        }
      }
    }

    // Steps and truncation
    steps_since_reset[i] += 1;
    if (steps_since_reset[i] >= max_steps) {
      truncated[i] = 1;
    }

    // Write observation
    if (obs_dim >= 1) {
      const int base = i * obs_dim;
      float final_fx = hx;
      float final_fy = hy;
      float final_dist = 0.f;
      if (num_food > 0) {
        float best = std::numeric_limits<float>::infinity();
        const int food_base_obs = i * num_food;
        const float phx = segments_x[base_seg + 0];
        const float phy = segments_y[base_seg + 0];
        for (int f = 0; f < num_food; ++f) {
          const float fx = food_x[food_base_obs + f];
          const float fy = food_y[food_base_obs + f];
          const float dfx = fx - phx;
          const float dfy = fy - phy;
          const float d = std::sqrt(dfx*dfx + dfy*dfy);
          if (d < best) {
            best = d;
            final_fx = fx;
            final_fy = fy;
            final_dist = d;
          }
        }
        if (!std::isfinite(final_dist)) {
          final_dist = 0.f;
          final_fx = phx;
          final_fy = phy;
        }
      }
      obs[base + 0] = segments_x[base_seg + 0];
      obs[base + 1] = segments_y[base_seg + 0];
      obs[base + 2] = dir_angle[i];
      obs[base + 3] = static_cast<float>(segments_count[i]);
      obs[base + 4] = final_fx;
      obs[base + 5] = final_fy;
      obs[base + 6] = final_dist;
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

// Very simple software renderer: draw head, body, and food as filled circles into NHWC uint8 buffer.
// Resolution: 84x84, per-env slice is contiguous.
void BatchedEnv::render_rgb() {
  if (render_mode != RenderMode::RGB) return;
  const int H = 84, W = 84, C = 3;
  const float scale = 84.0f / static_cast<float>(map_size);
  // Clear
  std::fill(rgb_image.begin(), rgb_image.end(), static_cast<uint8_t>(0));
  // For each env
  for (int i = 0; i < N; ++i) {
    const int base_img = i * H * W * C;
    // Helper functions for drawing pixels
    auto put_px = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b){
      if (x < 0 || x >= W || y < 0 || y >= H) return;
      const int idx = base_img + (y * W + x) * C;
      rgb_image[idx + 0] = r;
      rgb_image[idx + 1] = g;
      rgb_image[idx + 2] = b;
    };
    // Helper function for drawing a disk
    auto draw_disk = [&](float cx, float cy, float rad, uint8_t r, uint8_t g, uint8_t b){
      int sx = static_cast<int>(cx * scale);
      int sy = static_cast<int>(cy * scale);
      int rr = std::max(1, static_cast<int>(rad * scale));
      int xmin = std::max(0, sx - rr), xmax = std::min(W - 1, sx + rr);
      int ymin = std::max(0, sy - rr), ymax = std::min(H - 1, sy + rr);
      int rr2 = rr * rr;
      for (int y = ymin; y <= ymax; ++y) {
        int dy = y - sy;
        for (int x = xmin; x <= xmax; ++x) {
          int dx = x - sx;
          if (dx*dx + dy*dy <= rr2) put_px(x, y, r, g, b);
        }
      }
    };
    // Draw bot snakes first (behind player)
    for (int b = 0; b < num_bots; ++b) {
      const int bot_idx = i * num_bots + b;
      if (!bot_alive[bot_idx]) continue;
      const int bot_base_seg = bot_idx * max_bot_segments;
      // Draw bot body (orange/yellow)
      for (int s = 0; s < bot_segments_count[bot_idx]; ++s) {
        draw_disk(bot_segments_x[bot_base_seg + s], bot_segments_y[bot_base_seg + s], segment_radius, 200, 150, 60);
      }
      // Draw bot head brighter
      draw_disk(bot_segments_x[bot_base_seg + 0], bot_segments_y[bot_base_seg + 0], segment_radius * 1.1f, 255, 180, 40);
    }
    
    // Draw player body segments (green)
    const int base_seg = i * max_segments;
    for (int s = 0; s < segments_count[i]; ++s) {
      draw_disk(segments_x[base_seg + s], segments_y[base_seg + s], segment_radius, 80, 200, 80);
    }
    // Draw player head brighter
    draw_disk(segments_x[base_seg + 0], segments_y[base_seg + 0], segment_radius * 1.1f, 40, 255, 40);
    // Draw food items (red)
    const int food_base = i * num_food;
    for (int f = 0; f < num_food; ++f) {
      draw_disk(food_x[food_base + f], food_y[food_base + f], eat_radius, 200, 60, 60);
    }
  }
}