#pragma once
#include <vector>
#include <cstdint>
#include <utility>

// Constant used for angle wrapping; action is an angle in [0, 2π]
constexpr float kTwoPi = 6.283185307179586f;
constexpr float kPi    = 3.141592653589793f;

// Render modes
enum class RenderMode { Headless };

// Define the observation size for each render mode
enum class ObservationSize {
    // head_x, head_y, dir_angle, snake_len, nearest_food_x, nearest_food_y, nearest_food_dist
    Headless = 7
};

// Minimal "Box" space description for introspection via bindings.
// This is *not* used in hot loops; it's metadata.
struct BoxSpace {
    float low;
    float high;
    std::vector<int> shape;
    const char* dtype;

    BoxSpace(
        float low_, 
        float high_, 
        std::vector<int> shape_, 
        const char* dtype_ = "float32"
    ) : 
        low(low_), 
        high(high_), 
        shape(std::move(shape_)), 
        dtype(dtype_) 
    {}
};

// Batched environment: steps N envs at once.
class BatchedEnv {
public:
    // Sizes
    const int N;
    const int obs_dim;
    const int act_dim;

    // Game parameters
    const int map_size; // square map size
    const int step_size;
    const int max_steps; // truncation after this many steps since reset
    const float max_turn; // per-step max turn in radians (delta heading)
    const float eat_radius; // distance threshold to eat food

    // Space metadata (single env)
    BoxSpace single_observation_space;
    BoxSpace single_action_space;

    // Mode for rendering
    RenderMode render_mode;

    // Output buffers (returned to Python as zero-copy views)
    std::vector<float> obs;        // [N * obs_dim]
    std::vector<float> reward;     // [N]
    std::vector<uint8_t> terminated; // [N]
    std::vector<uint8_t> truncated;  // [N]

    // Minimal internal state 
    std::vector<float> head_x;     // [N]
    std::vector<float> head_y;     // [N]
    std::vector<float> dir_angle;  // [N] in range [0, 2*pi]
    std::vector<int>   snake_len;  // [N] starts from 3
    std::vector<float> food_x;     // [N]
    std::vector<float> food_y;     // [N]
    std::vector<int>   steps_since_reset; // [N]

    // RNG per env for deterministic spawning
    std::vector<unsigned long long> rng_state; // simple LCG or seed storage

    // Actions are a single angle per env; act_dim is fixed to 1.
    explicit BatchedEnv(
        int num_envs,
        RenderMode mode = RenderMode::Headless,
        int map_size = 100,
        int step_size = 1,
        int max_steps = 1000,
        float max_turn = kPi / 4.0f,
        float eat_radius = 1.0f,
        unsigned long long seed = 0ULL
    );

    // Disallow copying (large buffers); allow moving if needed (default okay).
    BatchedEnv(const BatchedEnv&) = delete;
    BatchedEnv& operator=(const BatchedEnv&) = delete;

    // Reset only the sub-envs where mask[i] == 1. Mask length must be N.
    void reset(const uint8_t* mask);

    // Reset all sub-envs.
    void full_reset();

    // Step all sub-envs with a batch of actions [N * act_dim], dt in "world units".
    void step(const float* actions);

    // Set base seed for all envs (env i uses seed + i)
    void set_seed(unsigned long long seed);
};