#pragma once
#include <vector>
#include <cstdint>
#include <utility>

// Constant used for angle wrapping
constexpr float kTwoPi = 6.283185307179586f;
constexpr float kPi    = 3.141592653589793f;

// Render modes
enum class RenderMode { Headless, RGB };

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
    const int step_size; // head speed in world units per step
    const int max_steps; // truncation after this many steps since reset
    const float max_turn; // per-step max turn in radians (delta heading)
    const float eat_radius; // distance threshold to eat food
    const int max_segments; // capacity of segments per env (player)
    const int initial_segments; // starting number of segments (player)
    const float segment_radius; // for collisions and rendering
    const float min_segment_distance; // target spacing between segments
    const float cell_size; // spatial hash cell size
    const int grid_w; // spatial hash width
    const int grid_h; // spatial hash height
    const int num_bots; // number of bot snakes per env
    const int max_bot_segments; // capacity of segments per bot (smaller than player)

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
    std::vector<float> dir_angle;  // [N] in range [0, 2*pi]
    std::vector<int>   snake_len;  // [N] starts from 3
    std::vector<float> food_x;     // [N]
    std::vector<float> food_y;     // [N]
    std::vector<int>   steps_since_reset; // [N]

    std::vector<uint8_t> rgb_image; // [N * H * W * 3] if render_mode is RenderMode::RGB

    // RNG per env for deterministic spawning
    std::vector<unsigned long long> rng_state; // simple LCG or seed storage

    // Continuous snake body (beads), SoA
    std::vector<float> segments_x; // [N * max_segments]
    std::vector<float> segments_y; // [N * max_segments]
    std::vector<int>   segments_count; // [N]
    std::vector<int>   pending_growth; // [N]

    // Bot snakes (multiple per env)
    std::vector<float> bot_segments_x; // [N * num_bots * max_bot_segments]
    std::vector<float> bot_segments_y; // [N * num_bots * max_bot_segments]
    std::vector<int>   bot_segments_count; // [N * num_bots]
    std::vector<int>   bot_pending_growth; // [N * num_bots]
    std::vector<float> bot_dir_angle; // [N * num_bots]
    std::vector<uint8_t> bot_alive; // [N * num_bots], 0 if dead, 1 if alive

    // Spatial hash for collision/queries
    std::vector<int> cell_head;   // [N * grid_w * grid_h], -1 for empty
    std::vector<int> next_in_cell; // [N * max_segments], next index or -1
    std::vector<int> bot_cell_head; // [N * grid_w * grid_h], -1 for empty (separate for bots)
    std::vector<int> bot_next_in_cell; // [N * num_bots * max_bot_segments], next index or -1

    // Actions are a single angle per env; act_dim is fixed to 1.
    explicit BatchedEnv(
        int num_envs,
        RenderMode mode = RenderMode::Headless,
        int map_size = 100,
        int step_size = 1,
        int max_steps = 1000,
        float max_turn = kPi / 4.0f,
        float eat_radius = 1.0f,
        unsigned long long seed = 0ULL,
        int max_segments = 64,
        int initial_segments = 4,
        float segment_radius = 2.0f,
        float min_segment_distance = 3.0f,
        float cell_size = 3.0f,
        int num_bots = 3,
        int max_bot_segments = 12
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

    // Render RGB frames into rgb_image if render_mode == RenderMode::RGB
    void render_rgb();
};