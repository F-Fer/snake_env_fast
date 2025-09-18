#pragma once
#include <vector>
#include <cstdint>

struct BatchedEnv {
  int N;
  int obs_dim = 1;
  int act_dim = 1;

  std::vector<float> x;
  std::vector<float> obs;
  std::vector<float> reward;
  std::vector<uint8_t> terminated;
  std::vector<uint8_t> truncated;

  explicit BatchedEnv(int num_envs);
  void full_reset();
  void reset(const uint8_t* mask);
  void step(const float* actions, float dt);
};