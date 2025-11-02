#include "env_core.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <limits>
#if defined(_OPENMP)
#include <omp.h>
#endif

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

namespace {

inline int fast_floor(float x) {
  int i = static_cast<int>(x);
  return (x < static_cast<float>(i)) ? (i - 1) : i;
}

inline int wrap_index(int value, int size) {
  int m = value % size;
  return (m < 0) ? (m + size) : m;
}

struct HexTileCache {
  static constexpr float kHexRadius = 2.8f;
  static constexpr float kSqrt3 = 1.73205080757f;
  static constexpr float kHexWidth = kSqrt3 * kHexRadius;         // world width of one hex
  static constexpr float kHexVertical = 1.5f * kHexRadius;        // vertical spacing
  static constexpr float kTileWorldW = kHexWidth * 2.0f;          // repeat every 2 columns
  static constexpr float kTileWorldH = kHexVertical * 2.0f;       // repeat every 2 rows
  static constexpr int kTileW = 96;
  static constexpr int kTileH = 128;

  std::array<uint8_t, kTileW * kTileH * 3> data;

  HexTileCache() { build(); }

  void build() {
    const float edge_width = kHexRadius * 0.25f;
    for (int y = 0; y < kTileH; ++y) {
      const float wy = (static_cast<float>(y) + 0.5f) / static_cast<float>(kTileH) * kTileWorldH;
      const int row = fast_floor(wy / kHexVertical);
      const bool row_odd = (row & 1) != 0;
      for (int x = 0; x < kTileW; ++x) {
        const float wx = (static_cast<float>(x) + 0.5f) / static_cast<float>(kTileW) * kTileWorldW;
        const float col_shift = row_odd ? 0.5f : 0.0f;
        const int col = fast_floor(wx / kHexWidth + col_shift);

        const float center_x = (static_cast<float>(col) - col_shift + 0.5f) * kHexWidth;
        const float center_y = (static_cast<float>(row) + 0.5f) * kHexVertical;
        const float lx = wx - center_x;
        const float ly = wy - center_y;
        const float ax = std::fabs(lx);
        const float ay = std::fabs(ly);

        const float edge_primary = ax - kHexWidth * 0.5f;
        const float edge_diagonal = (kSqrt3 * ax + ay) - kSqrt3 * kHexRadius;
        const float edge_value = std::max(edge_primary, edge_diagonal);
        const float dist_to_edge = std::max(-edge_value, 0.0f);
        float edge_factor = 1.0f - std::min(dist_to_edge / edge_width, 1.0f);
        if (edge_factor < 0.0f) edge_factor = 0.0f;

        const bool alternate = ((row + col) & 1) != 0;
        const float base_dark = alternate ? 28.0f : 22.0f;
        const float base_light = alternate ? 38.0f : 34.0f;
        float brightness = base_dark + edge_factor * (base_light - base_dark);

        const float vignette = 0.12f;
        brightness += vignette * base_light;

        const int idx = (y * kTileW + x) * 3;
        data[idx + 0] = static_cast<uint8_t>(std::clamp(brightness, 0.0f, 255.0f));
        data[idx + 1] = static_cast<uint8_t>(std::clamp(brightness * 0.92f, 0.0f, 255.0f));
        data[idx + 2] = static_cast<uint8_t>(std::clamp(brightness * 0.75f, 0.0f, 255.0f));
      }
    }
  }
};

const HexTileCache& GetHexTile() {
  static HexTileCache cache;
  return cache;
}

} // namespace

SnakeGymCore::SnakeGymCore(
  int num_envs,
  int map_size, 
  int step_size, 
  int max_steps, 
  float max_turn, 
  float eat_radius, 
  unsigned long long seed, 
  int max_segments, 
  int initial_segments, 
  float segment_radius, 
  float min_segment_distance, 
  float cell_size, 
  int num_bots, 
  int max_bot_segments, 
  int num_food, 
  float food_reward, 
  float kill_reward, 
  float death_reward,
  bool bot_ai_enabled)
  : N(num_envs),
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
    food_reward(food_reward),
    kill_reward(kill_reward),
    death_reward(death_reward),
    bot_ai_enabled(bot_ai_enabled),
    obs(N * static_cast<int>(ObservationSize::Headless), 0.f),
    reward(N, 0.f),
    terminated(N, 0),
    truncated(N, 0),
    dir_angle(N, 0.f),
    snake_len(N, 1),
    food_x(static_cast<size_t>(N) * static_cast<size_t>(num_food), 0.f),
    food_y(static_cast<size_t>(N) * static_cast<size_t>(num_food), 0.f),
    steps_since_reset(N, 0),
    rgb_image(N * 84 * 84 * 3, 0),
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

void SnakeGymCore::place_food(int env_idx, int food_slot) {
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

void SnakeGymCore::respawn_bot(int env_idx, int bot_idx) {
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

void SnakeGymCore::debug_set_player_state(int env_idx, const std::vector<float>& xs, const std::vector<float>& ys, float angle) {
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

void SnakeGymCore::debug_set_bot_state(int env_idx, int bot_idx, const std::vector<float>& xs, const std::vector<float>& ys, float angle, bool alive) {
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

void SnakeGymCore::debug_rebuild_spatial_hash(int env_idx) {
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

void SnakeGymCore::full_reset() {
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
    const int base = i * static_cast<int>(ObservationSize::Headless);
    hx = segments_x[base_seg + 0];
    hy = segments_y[base_seg + 0];
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

void SnakeGymCore::reset(const uint8_t* mask) {
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

      // Fill observation (7D)
      const int base = i * static_cast<int>(ObservationSize::Headless);
      hx = segments_x[base_seg + 0];
      hy = segments_y[base_seg + 0];
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

void SnakeGymCore::step(const float* actions) {
  for (int i = 0; i < N; ++i) {
    if (terminated[i] || truncated[i]) continue;

    // Steering: clamp delta turn and update heading
    float dtheta = actions[i * act_dim + 0];
    if (dtheta > max_turn) dtheta = max_turn;
    if (dtheta < -max_turn) dtheta = -max_turn;
    dir_angle[i] = wrap_angle_0_2pi(dir_angle[i] + dtheta);

    // Move head forward (segments[0])
    const int base_seg = i * max_segments;
    const int cell_base = i * grid_w * grid_h;
    auto clamp_cell = [&](int& cx, int& cy) {
      if (cx < 0) cx = 0; else if (cx >= grid_w) cx = grid_w - 1;
      if (cy < 0) cy = 0; else if (cy >= grid_h) cy = grid_h - 1;
    };
    float new_hx = segments_x[base_seg + 0] + std::cos(dir_angle[i]) * static_cast<float>(step_size);
    float new_hy = segments_y[base_seg + 0] + std::sin(dir_angle[i]) * static_cast<float>(step_size);

    // Check if head is out of bounds
    if (new_hx < 0 || new_hx >= map_size || new_hy < 0 || new_hy >= map_size) {
      terminated[i] = 1;
      truncated[i] = 1;
      continue;
    }
    int head_cx = static_cast<int>(new_hx / cell_size);
    int head_cy = static_cast<int>(new_hy / cell_size);
    clamp_cell(head_cx, head_cy);

    // Check if head is in a bot snake
    if (grid[cell_base + head_cy * grid_w + head_cx] >= 2) {
      terminated[i] = 1;
      truncated[i] = 1;
      reward[i] += death_reward;
      continue;
    }

    // Clear previous player occupancy
    for (int s = 0; s < segments_count[i]; ++s) {
      int cx = static_cast<int>(segments_x[base_seg + s] / cell_size);
      int cy = static_cast<int>(segments_y[base_seg + s] / cell_size);
      clamp_cell(cx, cy);
      grid[cell_base + cy * grid_w + cx] = -1;
    }

    // Update segments[0]
    segments_x[base_seg + 0] = new_hx;
    segments_y[base_seg + 0] = new_hy;
    float hx = new_hx;
    float hy = new_hy;

    // Follow pass for body segments
    const int segs = segments_count[i]; // :)
    for (int s = 1; s < segs; ++s) {
      float target_x = segments_x[base_seg + (s - 1)];
      float target_y = segments_y[base_seg + (s - 1)];
      float curr_x = segments_x[base_seg + s];
      float curr_y = segments_y[base_seg + s];
      float dxs = target_x - curr_x;
      float dys = target_y - curr_y;
      float d = std::sqrt(dxs*dxs + dys*dys);
      if (d > 0.0f) {
        float excess = d - min_segment_distance;
        if (excess > 0.0f) {
          float move = std::min(static_cast<float>(step_size), excess);
          float move_ratio = move / d;
          segments_x[base_seg + s] = curr_x + dxs * move_ratio;
          segments_y[base_seg + s] = curr_y + dys * move_ratio;
        }
      }
    }

    // Repaint player occupancy
    for (int s = 0; s < segments_count[i]; ++s) {
      int cx = static_cast<int>(segments_x[base_seg + s] / cell_size);
      int cy = static_cast<int>(segments_y[base_seg + s] / cell_size);
      clamp_cell(cx, cy);
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
      reward[i] += food_reward;
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
      // Update occupancy grid
      int cx = static_cast<int>(segments_x[base_seg + sc] / cell_size);
      int cy = static_cast<int>(segments_y[base_seg + sc] / cell_size);
      clamp_cell(cx, cy);
      grid[cell_base + cy * grid_w + cx] = 1;
    }

    // Bot snakes: AI and movement
    for (int b = 0; b < num_bots; ++b) {
      const int bot_idx = i * num_bots + b;
      if (!bot_alive[bot_idx]) {
        respawn_bot(i, b);
      }

      const int bot_base_seg = bot_idx * max_bot_segments;
      const int bot_segs = bot_segments_count[bot_idx];
      float bot_hx = bot_segments_x[bot_base_seg + 0];
      float bot_hy = bot_segments_y[bot_base_seg + 0];
      // Remove previous occupancy for this bot before movement
      for (int s = 0; s < bot_segs; ++s) {
        int bcx = static_cast<int>(bot_segments_x[bot_base_seg + s] / cell_size);
        int bcy = static_cast<int>(bot_segments_y[bot_base_seg + s] / cell_size);
        clamp_cell(bcx, bcy);
        grid[cell_base + bcy * grid_w + bcx] = -1;
      }
      
      // Simple AI: turn towards nearest food
      int bot_target_idx = -1;
      float bot_target_fx = bot_hx;
      float bot_target_fy = bot_hy;
      if (num_food > 0 && bot_ai_enabled) {
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
      
      // Check bounds
      if (bot_hx < 0 || bot_hx >= map_size || bot_hy < 0 || bot_hy >= map_size) {
        bot_alive[bot_idx] = 0;
        continue;
      }
      
      bot_segments_x[bot_base_seg + 0] = bot_hx;
      bot_segments_y[bot_base_seg + 0] = bot_hy;

      // Follow segments
      for (int s = 1; s < bot_segs; ++s) {
        float tx = bot_segments_x[bot_base_seg + (s - 1)];
        float ty = bot_segments_y[bot_base_seg + (s - 1)];
        float cx = bot_segments_x[bot_base_seg + s];
        float cy = bot_segments_y[bot_base_seg + s];
        float dxs = tx - cx;
        float dys = ty - cy;
        float d = std::sqrt(dxs*dxs + dys*dys);
        if (d > 0.0f) {
          float excess = d - min_segment_distance;
          if (excess > 0.0f) {
            float move = std::min(static_cast<float>(step_size), excess);
            float move_ratio = move / d;
            bot_segments_x[bot_base_seg + s] = cx + dxs * move_ratio;
            bot_segments_y[bot_base_seg + s] = cy + dys * move_ratio;
          }
        }
      }

      // Bot vs bot & bot vs player collision detection via grid
      int bot_head_cx = static_cast<int>(bot_hx / cell_size);
      int bot_head_cy = static_cast<int>(bot_hy / cell_size);
      clamp_cell(bot_head_cx, bot_head_cy);
      int occupant = grid[cell_base + bot_head_cy * grid_w + bot_head_cx];
      if (occupant == 1) {
        // Check if this is the player head
        if (bot_hx == segments_x[base_seg + 0] && bot_hy == segments_y[base_seg + 0]) {
          bot_alive[bot_idx] = 0;
          terminated[i] = 1;
          // Reward 
          reward[i] += death_reward;
          continue;
        }
        bot_alive[bot_idx] = 0;
        // Reward 
        reward[i] += kill_reward;
        continue;
      } else if (occupant >= 2 && occupant != (2 + b)) {
        int other_bot_idx = (occupant - 2) + i * num_bots;
        const int other_base_seg = other_bot_idx * max_bot_segments;
        const int other_segs = bot_segments_count[other_bot_idx];
        for (int s = 0; s < other_segs; ++s) {
          int ocx = static_cast<int>(bot_segments_x[other_base_seg + s] / cell_size);
          int ocy = static_cast<int>(bot_segments_y[other_base_seg + s] / cell_size);
          clamp_cell(ocx, ocy);
          grid[cell_base + ocy * grid_w + ocx] = -1;
        }
        bot_alive[bot_idx] = 0;
        bot_alive[other_bot_idx] = 0;
        continue;
      }
      
      // Occupancy grid update for bot
      for (int s = 0; s < bot_segs; ++s) {
        int bcx = static_cast<int>(bot_segments_x[bot_base_seg + s] / cell_size);
        int bcy = static_cast<int>(bot_segments_y[bot_base_seg + s] / cell_size);
        clamp_cell(bcx, bcy);
        grid[cell_base + bcy * grid_w + bcx] = 2 + b;
      }

      // Bot eating logic
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
        int bcx = static_cast<int>(bot_segments_x[bot_base_seg + sc] / cell_size);
        int bcy = static_cast<int>(bot_segments_y[bot_base_seg + sc] / cell_size);
        clamp_cell(bcx, bcy);
        grid[cell_base + bcy * grid_w + bcx] = 2 + b;
      }
    }
    
    // Check collision between player head and bot snakes
    head_cx = static_cast<int>(hx / cell_size);
    head_cy = static_cast<int>(hy / cell_size);
    clamp_cell(head_cx, head_cy);
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
          reward[i] += kill_reward;
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
    const int base = i * static_cast<int>(ObservationSize::Headless);
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

void SnakeGymCore::set_seed(unsigned long long seed) {
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
void SnakeGymCore::render_rgb() {
  const int H = 84, W = 84, C = 3;
  const float view_span_world = std::min(static_cast<float>(map_size), 60.0f);
  const float half_view = view_span_world * 0.5f;
  const float scale = static_cast<float>(W) / view_span_world;
  const float inv_scale = 1.0f / scale;
  const float map_f = static_cast<float>(map_size);
  const auto& tile = GetHexTile();
  const int tile_stride = HexTileCache::kTileW * 3;
  // For each env
  #pragma omp parallel for if(N > 1) schedule(static)
  for (int i = 0; i < N; ++i) {
    const int base_img = i * H * W * C;
    const int base_seg = i * max_segments;
    const float cam_x = segments_x[base_seg + 0];
    const float cam_y = segments_y[base_seg + 0];
    const float world_x0 = cam_x - half_view + 0.5f * inv_scale;
    const float world_y0 = cam_y - half_view + 0.5f * inv_scale;

    // Blit background using cached tile
    for (int py = 0; py < H; ++py) {
      const float wy = world_y0 + static_cast<float>(py) * inv_scale;
      const int tile_y = wrap_index(fast_floor((wy / HexTileCache::kTileWorldH) * HexTileCache::kTileH), HexTileCache::kTileH);
      const bool inside_y = (wy >= 0.0f && wy < map_f);
      float wx = world_x0;
      for (int px = 0; px < W; ++px, wx += inv_scale) {
        const bool inside = inside_y && wx >= 0.0f && wx < map_f;
        uint8_t r = 8, g = 8, b = 12;
        if (inside) {
          const int tile_x = wrap_index(fast_floor((wx / HexTileCache::kTileWorldW) * HexTileCache::kTileW), HexTileCache::kTileW);
          const int t_idx = tile_y * tile_stride + tile_x * 3;
          r = tile.data[t_idx + 0];
          g = tile.data[t_idx + 1];
          b = tile.data[t_idx + 2];
        }
        const int idx = base_img + (py * W + px) * C;
        rgb_image[idx + 0] = r;
        rgb_image[idx + 1] = g;
        rgb_image[idx + 2] = b;
      }
    }

    // Helper functions for drawing pixels
    auto put_px = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b){
      if (x < 0 || x >= W || y < 0 || y >= H) return;
      const int idx = base_img + (y * W + x) * C;
      rgb_image[idx + 0] = r;
      rgb_image[idx + 1] = g;
      rgb_image[idx + 2] = b;
    };
    // Helper function for drawing a disk centered on the camera
    auto draw_disk = [&](float wx, float wy, float rad, uint8_t r, uint8_t g, uint8_t b){
      const float dx_world = wx - cam_x;
      const float dy_world = wy - cam_y;
      if (std::fabs(dx_world) > (half_view + rad) || std::fabs(dy_world) > (half_view + rad)) return;
      int sx = static_cast<int>(std::round(dx_world * scale + static_cast<float>(W) * 0.5f));
      int sy = static_cast<int>(std::round(dy_world * scale + static_cast<float>(H) * 0.5f));
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
