add_library(parte-common STATIC
  logger.cpp
  preprocessing.cpp
)
add_library(parte::common ALIAS parte-common)

target_compile_features(parte-common PUBLIC cxx_std_20)

target_include_directories(parte-common PUBLIC
  "${PROJECT_SOURCE_DIR}"
)

target_link_libraries(parte-common
  PUBLIC
    Eigen3::Eigen
  PRIVATE
    OpenMP::OpenMP_CXX
)
