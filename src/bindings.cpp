#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "env_core.hpp"

namespace py = pybind11;

template <typename T>
py::array make_view(std::vector<T>& v, std::vector<ssize_t> shape) {
    std::vector<ssize_t> strides(shape.size());
    ssize_t item = sizeof(T);
    for (ssize_t k = (ssize_t)shape.size()-1; k >= 0; --k) {
        strides[k] = item;
        item *= shape[k];
    }
    return py::array(py::buffer_info(
        v.data(), sizeof(T), py::format_descriptor<T>::format(),
        shape.size(), shape, strides
    ));
}

PYBIND11_MODULE(snake_gym_core, m) {
    py::enum_<RenderMode>(m, "RenderMode")
        .value("Headless", RenderMode::Headless)
        .value("RGB", RenderMode::RGB)
        .export_values();

    py::class_<SnakeGymCore>(m, "SnakeGymCore")
        .def(py::init<int, int, int, int, float, float, unsigned long long, int, int, float, float, float, int, int, int, float, float, float, bool>(),
             py::arg("num_envs"),
             py::arg("map_size") = 100,
             py::arg("step_size") = 1,
             py::arg("max_steps") = 1000,
             py::arg("max_turn") = kPi / 4.0f,
             py::arg("eat_radius") = 1.0f,
             py::arg("seed") = 0ULL,
             py::arg("max_segments") = 64,
             py::arg("initial_segments") = 4,
             py::arg("segment_radius") = 2.0f,
             py::arg("min_segment_distance") = 3.0f,
             py::arg("cell_size") = 3.0f,
             py::arg("num_bots") = 3,
             py::arg("max_bot_segments") = 12,
             py::arg("num_food") = 5,
             py::arg("food_reward") = 1.0f,
             py::arg("kill_reward") = 5.0f,
             py::arg("death_reward") = -1.0f,
             py::arg("bot_ai_enabled") = true)
        .def("reset", [](SnakeGymCore& e, py::array_t<uint8_t, py::array::c_style|py::array::forcecast> mask){
            if (mask.ndim()!=1 || mask.shape(0)!=e.N) throw std::runtime_error("mask shape mismatch");
            py::gil_scoped_release release;
            e.reset(mask.data());
        })
        .def("step", [](SnakeGymCore& e, py::array_t<float, py::array::c_style|py::array::forcecast> actions){
            if (actions.ndim()!=2 || actions.shape(0)!=e.N || actions.shape(1)!=e.act_dim)
                throw std::runtime_error("actions shape mismatch");
            py::gil_scoped_release release;
            e.step(actions.data());
        }, py::arg("actions"))
        .def("set_seed", &SnakeGymCore::set_seed, py::arg("seed"))
        .def("render_rgb", &SnakeGymCore::render_rgb)
        .def("debug_set_player_state", [](SnakeGymCore& e, int env_idx, py::array_t<float> xs, py::array_t<float> ys, float angle){
            if (xs.ndim() != 1 || ys.ndim() != 1) throw std::runtime_error("debug_set_player_state expects 1D arrays");
            if (xs.shape(0) != ys.shape(0)) throw std::runtime_error("debug_set_player_state arrays must match");
            std::vector<float> vx(xs.shape(0));
            std::vector<float> vy(ys.shape(0));
            std::memcpy(vx.data(), xs.data(), vx.size() * sizeof(float));
            std::memcpy(vy.data(), ys.data(), vy.size() * sizeof(float));
            e.debug_set_player_state(env_idx, vx, vy, angle);
        }, py::arg("env_idx"), py::arg("xs"), py::arg("ys"), py::arg("angle"))
        .def("debug_set_bot_state", [](SnakeGymCore& e, int env_idx, int bot_idx, py::array_t<float> xs, py::array_t<float> ys, float angle, bool alive){
            if (xs.ndim() != 1 || ys.ndim() != 1) throw std::runtime_error("debug_set_bot_state expects 1D arrays");
            if (xs.shape(0) != ys.shape(0)) throw std::runtime_error("debug_set_bot_state arrays must match");
            std::vector<float> vx(xs.shape(0));
            std::vector<float> vy(ys.shape(0));
            std::memcpy(vx.data(), xs.data(), vx.size() * sizeof(float));
            std::memcpy(vy.data(), ys.data(), vy.size() * sizeof(float));
            e.debug_set_bot_state(env_idx, bot_idx, vx, vy, angle, alive);
        }, py::arg("env_idx"), py::arg("bot_idx"), py::arg("xs"), py::arg("ys"), py::arg("angle"), py::arg("alive"))
        .def("debug_rebuild_spatial_hash", &SnakeGymCore::debug_rebuild_spatial_hash, py::arg("env_idx"))
        .def_property_readonly("obs", [](SnakeGymCore& e){ return make_view(e.obs, {e.N, static_cast<int>(ObservationSize::Headless)}); })
        .def_property_readonly("rgb", [](SnakeGymCore& e){ return py::array(py::buffer_info(
            e.rgb_image.data(), sizeof(uint8_t), py::format_descriptor<uint8_t>::format(),
            4, {e.N, 84, 84, 3}, {static_cast<ssize_t>(84*84*3), static_cast<ssize_t>(84*3), static_cast<ssize_t>(3), static_cast<ssize_t>(1)}
        )); })
        .def_property_readonly("reward", [](SnakeGymCore& e){ return make_view(e.reward, {e.N}); })
        .def_property_readonly("terminated", [](SnakeGymCore& e){ return make_view(e.terminated, {e.N}); })
        .def_property_readonly("truncated", [](SnakeGymCore& e){ return make_view(e.truncated, {e.N}); })
        .def_property_readonly("grid", [](SnakeGymCore& e){ return make_view(e.grid, {e.N, e.grid_w, e.grid_h}); })
        .def_property_readonly("grid_w", [](SnakeGymCore& e){ return e.grid_w; })
        .def_property_readonly("grid_h", [](SnakeGymCore& e){ return e.grid_h; })
        .def_property_readonly("bot_alive", [](SnakeGymCore& e){ return make_view(e.bot_alive, {e.N, e.num_bots}); })
        .def_readonly("N", &SnakeGymCore::N)
        .def_readonly("act_dim", &SnakeGymCore::act_dim)
        .def_readonly("max_turn", &SnakeGymCore::max_turn);
}