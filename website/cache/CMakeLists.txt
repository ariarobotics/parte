add_executable(parte-cache export.cpp)
target_link_libraries(parte-cache PRIVATE parte-segmentation parte-registration OpenMP::OpenMP_CXX)
set_target_properties(parte-cache PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
