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

PYBIND11_MODULE(_fastenv, m) {
    py::enum_<RenderMode>(m, "RenderMode")
        .value("Headless", RenderMode::Headless)
        .export_values();

    py::class_<BatchedEnv>(m, "BatchedEnv")
        .def(py::init<int, RenderMode, int, int, int, float, float, unsigned long long>(),
             py::arg("num_envs"),
             py::arg("mode") = RenderMode::Headless,
             py::arg("map_size") = 100,
             py::arg("step_size") = 1,
             py::arg("max_steps") = 1000,
             py::arg("max_turn") = kPi / 4.0f,
             py::arg("eat_radius") = 1.0f,
             py::arg("seed") = 0ULL)
        .def_property_readonly("single_observation_space", [](const BatchedEnv& e){
            py::dict d;
            d["low"] = e.single_observation_space.low;
            d["high"] = e.single_observation_space.high;
            py::list shape;
            for (int s : e.single_observation_space.shape) shape.append(s);
            d["shape"] = shape;
            d["dtype"] = e.single_observation_space.dtype;
            return d;
        })
        .def_property_readonly("single_action_space", [](const BatchedEnv& e){
            py::dict d;
            d["low"] = e.single_action_space.low;
            d["high"] = e.single_action_space.high;
            py::list shape;
            for (int s : e.single_action_space.shape) shape.append(s);
            d["shape"] = shape;
            d["dtype"] = e.single_action_space.dtype;
            return d;
        })
        .def("reset", [](BatchedEnv& e, py::array_t<uint8_t, py::array::c_style|py::array::forcecast> mask){
            if (mask.ndim()!=1 || mask.shape(0)!=e.N) throw std::runtime_error("mask shape mismatch");
            py::gil_scoped_release release;
            e.reset(mask.data());
        })
        .def("step", [](BatchedEnv& e, py::array_t<float, py::array::c_style|py::array::forcecast> actions){
            if (actions.ndim()!=2 || actions.shape(0)!=e.N || actions.shape(1)!=e.act_dim)
                throw std::runtime_error("actions shape mismatch");
            py::gil_scoped_release release;
            e.step(actions.data());
        }, py::arg("actions"))
        .def("set_seed", &BatchedEnv::set_seed, py::arg("seed"))
        .def_property_readonly("obs", [](BatchedEnv& e){ return make_view(e.obs, {e.N, e.obs_dim}); })
        .def_property_readonly("reward", [](BatchedEnv& e){ return make_view(e.reward, {e.N}); })
        .def_property_readonly("terminated", [](BatchedEnv& e){ return make_view(e.terminated, {e.N}); })
        .def_property_readonly("truncated", [](BatchedEnv& e){ return make_view(e.truncated, {e.N}); })
        .def_readonly("N", &BatchedEnv::N)
        .def_readonly("obs_dim", &BatchedEnv::obs_dim)
        .def_readonly("act_dim", &BatchedEnv::act_dim);
}