set_target_properties(
  parte-common parte-segmentation parte-registration clipperplus
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)

add_library(parte-cache SHARED export.cpp)
target_link_libraries(parte-cache PRIVATE parte-registration Open3D::Open3D)
set_target_properties(parte-cache PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  WINDOWS_EXPORT_ALL_SYMBOLS ON
)
