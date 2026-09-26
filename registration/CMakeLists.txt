add_library(parte-registration STATIC
  fpfh.cpp
  matching.cpp
  pch.cpp
  pipeline.cpp
  outliers.cpp
  transformation.cpp
)

target_link_libraries(parte-registration
  PUBLIC
    parte::common
    clipperplus
  PRIVATE
    parte-segmentation
    OpenMP::OpenMP_CXX
)
