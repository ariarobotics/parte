find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NANOBIND_CMAKE_ROOT
)
find_package(
  nanobind CONFIG REQUIRED
  PATHS "${NANOBIND_CMAKE_ROOT}"
  NO_DEFAULT_PATH
)

set_target_properties(
  parte-common parte-segmentation parte-registration wpmc
  PROPERTIES POSITION_INDEPENDENT_CODE ON
)

nanobind_add_module(pyparte bindings.cpp)
set_target_properties(pyparte PROPERTIES OUTPUT_NAME parte)
target_link_libraries(pyparte PRIVATE
  parte::common
  parte-segmentation
  parte-registration
  OpenMP::OpenMP_CXX
)
