cmake_minimum_required(VERSION 3.21)

project(PARTE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


option(BUILD_EXECUTABLES "Build the Open3D-backed command-line executables" ON)
option(BUILD_PYTHON_MODULE "Build the nanobind Python extension" OFF)

include(FetchContent)
include(cmake/EigenSetup.cmake)

find_package(OpenMP REQUIRED)

add_subdirectory(common)
add_subdirectory(segmentation)
add_subdirectory(registration)
add_subdirectory(wpmc)

if(BUILD_EXECUTABLES)
  include(cmake/Open3DSetup.cmake)
  add_executable(parte parte.cpp)
  add_executable(parte-benchmark benchmark.cpp)

  target_link_libraries(parte PRIVATE parte-segmentation parte-registration wpmc OpenMP::OpenMP_CXX Open3D::Open3D)
  target_link_libraries(parte-benchmark PRIVATE parte-segmentation parte-registration wpmc OpenMP::OpenMP_CXX Open3D::Open3D)
endif()


if(BUILD_PYTHON_MODULE)
  add_subdirectory(python)
endif()
