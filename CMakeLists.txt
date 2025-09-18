cmake_minimum_required(VERSION 3.18)
project(snake_env_fast LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_fastenv src/bindings.cpp src/env_core.cpp)
target_compile_features(_fastenv PRIVATE cxx_std_17)
target_compile_options(_fastenv PRIVATE -O3 -march=native)

install(TARGETS _fastenv DESTINATION snake_env_fast)