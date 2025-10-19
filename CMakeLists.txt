cmake_minimum_required(VERSION 3.18)
project(snake_env_fast LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)


pybind11_add_module(_fastenv src/bindings.cpp src/env_core.cpp)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_fastenv PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_features(_fastenv PRIVATE cxx_std_17)
target_compile_options(_fastenv PRIVATE -O3 -march=native)
target_include_directories(_fastenv PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

install(TARGETS _fastenv DESTINATION snake_env_fast)