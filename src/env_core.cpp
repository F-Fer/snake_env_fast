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

BatchedEnv::BatchedEnv(int num_envs, RenderMode mode, int map_size, int step_size, int max_steps, float max_turn, float eat_radius, unsigned long long seed, int max_segments, int initial_segments, float segment_radius, float min_segment_distance, float cell_size)
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
    single_observation_space(BoxSpace(-INFINITY, INFINITY, {static_cast<int>(ObservationSize::Headless)}, "float32")),
    single_action_space(-max_turn, max_turn, {1}, "float32"),
    render_mode(mode),
    obs(N * static_cast<int>(ObservationSize::Headless), 0.f),
    reward(N, 0.f),
    terminated(N, 0),
    truncated(N, 0),
    dir_angle(N, 0.f),
    snake_len(N, 1),
    food_x(N, 0.f),
    food_y(N, 0.f),
    steps_since_reset(N, 0),
    rgb_image(mode == RenderMode::RGB ? N * 84 * 84 * 3 : 0, 0),
    rng_state(N, 0ULL),
    segments_x(static_cast<size_t>(N) * static_cast<size_t>(max_segments), 0.f),
    segments_y(static_cast<size_t>(N) * static_cast<size_t>(max_segments), 0.f),
    segments_count(N, 0),
    pending_growth(N, 0),
    cell_head(static_cast<size_t>(N) * static_cast<size_t>(grid_w) * static_cast<size_t>(grid_h), -1),
    next_in_cell(static_cast<size_t>(N) * static_cast<size_t>(max_segments), -1)
{
    set_seed(seed);
    full_reset();
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
  std::fill(cell_head.begin(), cell_head.end(), -1);
  std::fill(next_in_cell.begin(), next_in_cell.end(), -1);

  // Initialize each env with randomized head, angle, and food
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

    // Build spatial hash for this env
    const int cell_base = i * grid_w * grid_h;
    for (int c = 0; c < grid_w * grid_h; ++c) cell_head[cell_base + c] = -1;
    for (int s = 0; s < segments_count[i]; ++s) {
      int cx = static_cast<int>(segments_x[base_seg + s] / cell_size);
      int cy = static_cast<int>(segments_y[base_seg + s] / cell_size);
      if (cx < 0) cx = 0; if (cx >= grid_w) cx = grid_w - 1;
      if (cy < 0) cy = 0; if (cy >= grid_h) cy = grid_h - 1;
      int cell_idx = cell_base + cy * grid_w + cx;
      next_in_cell[base_seg + s] = cell_head[cell_idx];
      cell_head[cell_idx] = base_seg + s;
    }

    // Spawn food away from snake (rejection sampling)
    float fx = 0.f, fy = 0.f;
    bool placed = false;
    for (int tries = 0; tries < 64 && !placed; ++tries) {
      fx = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i])) * static_cast<float>(map_size);
      fy = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i])) * static_cast<float>(map_size);
      int cx = static_cast<int>(fx / cell_size);
      int cy = static_cast<int>(fy / cell_size);
      if (cx < 0) cx = 0; if (cx >= grid_w) cx = grid_w - 1;
      if (cy < 0) cy = 0; if (cy >= grid_h) cy = grid_h - 1;
      bool ok = true;
      for (int oy = -1; oy <= 1 && ok; ++oy) {
        for (int ox = -1; ox <= 1 && ok; ++ox) {
          int ncx = cx + ox, ncy = cy + oy;
          if (ncx < 0 || ncx >= grid_w || ncy < 0 || ncy >= grid_h) continue;
          int head_idx = cell_head[cell_base + ncy * grid_w + ncx];
          while (head_idx != -1) {
            int sidx = head_idx - base_seg;
            float dx = segments_x[base_seg + sidx] - fx;
            float dy = segments_y[base_seg + sidx] - fy;
            if (std::sqrt(dx*dx + dy*dy) < (segment_radius + eat_radius)) { ok = false; break; }
            head_idx = next_in_cell[head_idx];
          }
        }
      }
      if (ok) placed = true;
    }
    food_x[i] = placed ? fx : rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i])) * static_cast<float>(map_size);
    food_y[i] = placed ? fy : rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i])) * static_cast<float>(map_size);

    // Fill observation (7D)
    if (obs_dim >= 1) {
      const int base = i * obs_dim;
      const int base_seg = i * max_segments;
      const float hx = segments_x[base_seg + 0];
      const float hy = segments_y[base_seg + 0];
      const float dx = food_x[i] - hx;
      const float dy = food_y[i] - hy;
      const float dist = std::sqrt(dx*dx + dy*dy);
      obs[base + 0] = hx;
      obs[base + 1] = hy;
      obs[base + 2] = dir_angle[i];
      obs[base + 3] = static_cast<float>(segments_count[i]);
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
      float hx = rx * static_cast<float>(map_size);
      float hy = ry * static_cast<float>(map_size);
      dir_angle[i] = ra * kTwoPi;
      pending_growth[i] = 0;
      const int base_seg = i * max_segments;
      segments_count[i] = std::min(initial_segments, max_segments);
      segments_x[base_seg + 0] = hx;
      segments_y[base_seg + 0] = hy;
      for (int s = 0; s < segments_count[i]; ++s) {
        float offset = static_cast<float>(s) * min_segment_distance;
        float ax = (s == 0 ? 0.0f : std::cos(dir_angle[i] + kPi) * offset);
        float ay = (s == 0 ? 0.0f : std::sin(dir_angle[i] + kPi) * offset);
        segments_x[base_seg + s] = hx + ax;
        segments_y[base_seg + s] = hy + ay;
      }
      // Rebuild hash for this env
      const int cell_base = i * grid_w * grid_h;
      for (int c = 0; c < grid_w * grid_h; ++c) cell_head[cell_base + c] = -1;
      for (int s = 0; s < segments_count[i]; ++s) {
        int cx = static_cast<int>(segments_x[base_seg + s] / cell_size);
        int cy = static_cast<int>(segments_y[base_seg + s] / cell_size);
        if (cx < 0) cx = 0; if (cx >= grid_w) cx = grid_w - 1;
        if (cy < 0) cy = 0; if (cy >= grid_h) cy = grid_h - 1;
        int cell_idx = cell_base + cy * grid_w + cx;
        next_in_cell[base_seg + s] = cell_head[cell_idx];
        cell_head[cell_idx] = base_seg + s;
      }
      // Respawn food with spacing
      float fx = 0.f, fy = 0.f;
      bool placed = false;
      for (int tries = 0; tries < 64 && !placed; ++tries) {
        fx = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i])) * static_cast<float>(map_size);
        fy = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i])) * static_cast<float>(map_size);
        int cx = static_cast<int>(fx / cell_size);
        int cy = static_cast<int>(fy / cell_size);
        if (cx < 0) cx = 0; if (cx >= grid_w) cx = grid_w - 1;
        if (cy < 0) cy = 0; if (cy >= grid_h) cy = grid_h - 1;
        bool ok = true;
        for (int oy = -1; oy <= 1 && ok; ++oy) {
          for (int ox = -1; ox <= 1 && ok; ++ox) {
            int ncx = cx + ox, ncy = cy + oy;
            if (ncx < 0 || ncx >= grid_w || ncy < 0 || ncy >= grid_h) continue;
            int head_idx = cell_head[cell_base + ncy * grid_w + ncx];
            while (head_idx != -1) {
                int sidx = head_idx - base_seg;
                float dx = segments_x[base_seg + sidx] - fx;
                float dy = segments_y[base_seg + sidx] - fy;
                if (std::sqrt(dx*dx + dy*dy) < (segment_radius + eat_radius)) { ok = false; break; }
                head_idx = next_in_cell[head_idx];
            }
          }
        }
        if (ok) placed = true;
      }
      food_x[i] = placed ? fx : rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i])) * static_cast<float>(map_size);
      food_y[i] = placed ? fy : rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i])) * static_cast<float>(map_size);

      if (obs_dim >= 1) {
          const int base = i * obs_dim;
          const float hx = segments_x[base_seg + 0];
          const float hy = segments_y[base_seg + 0];
          const float dx = food_x[i] - hx;
          const float dy = food_y[i] - hy;
          const float dist = std::sqrt(dx*dx + dy*dy);
          obs[base + 0] = hx;
          obs[base + 1] = hy;
          obs[base + 2] = dir_angle[i];
          obs[base + 3] = static_cast<float>(segments_count[i]);
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
    const int segs = segments_count[i];
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

    // Rebuild spatial hash for this env
    const int cell_base = i * grid_w * grid_h;
    for (int c = 0; c < grid_w * grid_h; ++c) cell_head[cell_base + c] = -1;
    for (int s = 0; s < segs; ++s) {
      int cx = static_cast<int>(segments_x[base_seg + s] / cell_size);
      int cy = static_cast<int>(segments_y[base_seg + s] / cell_size);
      if (cx < 0) cx = 0; if (cx >= grid_w) cx = grid_w - 1;
      if (cy < 0) cy = 0; if (cy >= grid_h) cy = grid_h - 1;
      int cell_idx = cell_base + cy * grid_w + cx;
      next_in_cell[base_seg + s] = cell_head[cell_idx];
      cell_head[cell_idx] = base_seg + s;
    }

    // Reward and eating logic
    reward[i] = 0.0f;
    float dx = food_x[i] - hx;
    float dy = food_y[i] - hy;
    float dist = std::sqrt(dx*dx + dy*dy);
    if (dist <= eat_radius) {
      reward[i] = 1.0f;
      pending_growth[i] += 1;
      // Respawn food away from snake
      float fx = 0.f, fy = 0.f; bool placed = false;
      for (int tries = 0; tries < 64 && !placed; ++tries) {
        fx = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i])) * static_cast<float>(map_size);
        fy = rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i])) * static_cast<float>(map_size);
        int cx = static_cast<int>(fx / cell_size);
        int cy = static_cast<int>(fy / cell_size);
        if (cx < 0) cx = 0; if (cx >= grid_w) cx = grid_w - 1;
        if (cy < 0) cy = 0; if (cy >= grid_h) cy = grid_h - 1;
        bool ok = true;
        for (int oy = -1; oy <= 1 && ok; ++oy) {
          for (int ox = -1; ox <= 1 && ok; ++ox) {
            int ncx = cx + ox, ncy = cy + oy;
            if (ncx < 0 || ncx >= grid_w || ncy < 0 || ncy >= grid_h) continue;
            int head_idx = cell_head[cell_base + ncy * grid_w + ncx];
            while (head_idx != -1) {
              int sidx = head_idx - base_seg;
              float ddx = segments_x[base_seg + sidx] - fx;
              float ddy = segments_y[base_seg + sidx] - fy;
              if (std::sqrt(ddx*ddx + ddy*ddy) < (segment_radius + eat_radius)) { ok = false; break; }
              head_idx = next_in_cell[head_idx];
            }
          }
        }
        if (ok) placed = true;
      }
      food_x[i] = placed ? fx : rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i])) * static_cast<float>(map_size);
      food_y[i] = placed ? fy : rand_uniform_01(reinterpret_cast<uint64_t&>(rng_state[i])) * static_cast<float>(map_size);
      // Recompute dist to new food
      dx = food_x[i] - hx; dy = food_y[i] - hy; dist = std::sqrt(dx*dx + dy*dy);
    }

    // Apply growth one segment per step
    if (pending_growth[i] > 0 && segments_count[i] < max_segments) {
      int sc = segments_count[i];
      float lx = segments_x[base_seg + (sc - 1)];
      float ly = segments_y[base_seg + (sc - 1)];
      float lx2 = (sc >= 2) ? segments_x[base_seg + (sc - 2)] : (lx - std::cos(dir_angle[i]) * 1.0f);
      float ly2 = (sc >= 2) ? segments_y[base_seg + (sc - 2)] : (ly - std::sin(dir_angle[i]) * 1.0f);
      float tx = lx - lx2;
      float ty = ly - ly2;
      float td = std::sqrt(tx*tx + ty*ty);
      if (td > 0.0f) { tx /= td; ty /= td; }
      segments_x[base_seg + sc] = lx + tx * min_segment_distance;
      segments_y[base_seg + sc] = ly + ty * min_segment_distance;
      segments_count[i] = sc + 1;
      pending_growth[i] -= 1;
    }

    // Self-collision: head vs body (skip close indices)
    bool self_hit = false;
    int head_cx = static_cast<int>(hx / cell_size);
    int head_cy = static_cast<int>(hy / cell_size);
    const float collide_r = segment_radius * 1.2f;
    for (int oy = -1; oy <= 1 && !self_hit; ++oy) {
      for (int ox = -1; ox <= 1 && !self_hit; ++ox) {
        int ncx = head_cx + ox, ncy = head_cy + oy;
        if (ncx < 0 || ncx >= grid_w || ncy < 0 || ncy >= grid_h) continue;
        int ptr = cell_head[cell_base + ncy * grid_w + ncx];
        while (ptr != -1) {
          int sidx = ptr - base_seg;
          if (sidx >= 3) {
            float ddx = segments_x[ptr] - hx;
            float ddy = segments_y[ptr] - hy;
            if (std::sqrt(ddx*ddx + ddy*ddy) < collide_r) { self_hit = true; break; }
          }
          ptr = next_in_cell[ptr];
        }
      }
    }
    if (self_hit) {
      terminated[i] = 1;
    }

    // Steps and truncation
    steps_since_reset[i] += 1;
    if (steps_since_reset[i] >= max_steps) {
      truncated[i] = 1;
    }

    // Write observation
    if (obs_dim >= 1) {
      const int base = i * obs_dim;
      obs[base + 0] = segments_x[base_seg + 0];
      obs[base + 1] = segments_y[base_seg + 0];
      obs[base + 2] = dir_angle[i];
      obs[base + 3] = static_cast<float>(segments_count[i]);
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
    // Draw body segments
    const int base_seg = i * max_segments;
    for (int s = 0; s < segments_count[i]; ++s) {
      draw_disk(segments_x[base_seg + s], segments_y[base_seg + s], segment_radius, 80, 200, 80);
    }
    // Draw head brighter
    draw_disk(segments_x[base_seg + 0], segments_y[base_seg + 0], segment_radius * 1.1f, 40, 255, 40);
    // Draw food
    draw_disk(food_x[i], food_y[i], eat_radius, 200, 60, 60);
  }
}