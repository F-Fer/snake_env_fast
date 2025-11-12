cmake_minimum_required(VERSION 3.18)
project(snake_gym LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

pybind11_add_module(snake_gym_core src/bindings.cpp src/env_core.cpp)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(snake_gym_core PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(snake_gym_core PRIVATE SNAKEGYM_USE_OPENMP=1)
endif()
target_compile_features(snake_gym_core PRIVATE cxx_std_17)
target_compile_options(snake_gym_core PRIVATE -O3 -march=native)
target_include_directories(snake_gym_core PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

install(TARGETS snake_gym_core DESTINATION snake_gym_core)