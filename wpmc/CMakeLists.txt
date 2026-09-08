cmake_minimum_required(VERSION 3.21)

project(wpmc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(WPMC_SOURCE_FILES
    pmc_cores.cpp
    pmcx_maxclique.cpp
)

add_library(wpmc ${WPMC_SOURCE_FILES})

target_include_directories(wpmc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenMP REQUIRED)
target_link_libraries(wpmc PUBLIC OpenMP::OpenMP_CXX)
