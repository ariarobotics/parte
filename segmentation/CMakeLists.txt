find_package(OpenMP REQUIRED)

add_library(parte-segmentation STATIC
  segmentation.cpp
  tgs.cpp
)

target_link_libraries(parte-segmentation
  PUBLIC
    parte::common
  PRIVATE
    OpenMP::OpenMP_CXX
)
