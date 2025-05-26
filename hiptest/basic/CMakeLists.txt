cmake_minimum_required(VERSION 3.10)
project(vecadd LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_COMPILER "/opt/rocm/bin/hipcc")
set(CMAKE_CXX_COMPILER "/opt/rocm/bin/hipcc")

find_package(Python3 REQUIRED COMPONENTS Development)
find_package(pybind11 REQUIRED)

add_library(vecadd MODULE vec_add.hip.cpp)
target_link_libraries(vecadd PRIVATE pybind11::module)

set_target_properties(vecadd PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)
