#include "env_core.hpp"
#include <algorithm>
#include <cmath>

BatchedEnv::BatchedEnv(int num_envs) : N(num_envs),
  x(N, 0.f), obs(N*obs_dim, 0.f), reward(N, 0.f),
  terminated(N, 0), truncated(N, 0) { full_reset(); }

void BatchedEnv::full_reset() {
  std::fill(x.begin(), x.end(), 0.f);
  std::fill(obs.begin(), obs.end(), 0.f);
  std::fill(reward.begin(), reward.end(), 0.f); 
  std::fill(terminated.begin(), terminated.end(), 0);
  std::fill(truncated.begin(), truncated.end(), 0);
}

void BatchedEnv::reset(const uint8_t* mask) {
  for (int i=0;i<N;++i) if (mask[i]) {
    x[i]=0.f; obs[i]=0.f; reward[i]=0.f; terminated[i]=0; truncated[i]=0;
  }
}

void BatchedEnv::step(const float* actions, float dt) {
  const float bound = 10.f;
  for (int i=0;i<N;++i) {
    if (terminated[i] | truncated[i]) continue;
    float a = actions[i*act_dim + 0];
    x[i] += dt * a;
    obs[i] = x[i];
    reward[i] = -std::abs(x[i]);
    if (std::abs(x[i]) > bound) terminated[i] = 1;
  }
}