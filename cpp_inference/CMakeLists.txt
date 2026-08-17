cmake_minimum_required(VERSION 3.18)
project(load_prototypes CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(load_prototypes load_prototypes.cpp)
target_link_libraries(load_prototypes "${TORCH_LIBRARIES}")
