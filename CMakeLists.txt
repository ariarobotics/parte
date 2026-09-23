cmake_minimum_required(VERSION 3.21)

project(PARTE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


option(BUILD_EXECUTABLES "Build the Open3D-backed command-line executables" ON)
option(BUILD_CACHE_EXPORTER "Build the website cache exporter" OFF)
option(BUILD_PYTHON_MODULE "Build the nanobind Python extension" OFF)

include(FetchContent)
include(cmake/EigenSetup.cmake)

find_package(OpenMP REQUIRED)

add_library(clipperplus STATIC
  clipperp/src/clique_optimization.cpp
  clipperp/src/clipperplus_clique.cpp
  clipperp/src/clipperplus_graph.cpp
  clipperp/src/clipperplus_heuristic.cpp
  clipperp/src/utils.cpp
)
target_include_directories(clipperplus PUBLIC clipperp/include)
target_link_libraries(clipperplus PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)

add_subdirectory(common)
add_subdirectory(segmentation)
add_subdirectory(registration)

if(BUILD_EXECUTABLES)
  include(cmake/Open3DSetup.cmake)
  add_executable(parte parte.cpp)
  add_executable(parte-benchmark benchmark.cpp)

  target_link_libraries(parte PRIVATE parte-segmentation parte-registration OpenMP::OpenMP_CXX Open3D::Open3D)
  target_link_libraries(parte-benchmark PRIVATE parte-segmentation parte-registration OpenMP::OpenMP_CXX Open3D::Open3D)
endif()


if(BUILD_PYTHON_MODULE)
  add_subdirectory(python)
endif()

if(BUILD_CACHE_EXPORTER)
  add_subdirectory(website/cache)
endif()
