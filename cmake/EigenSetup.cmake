option(USE_SYSTEM_EIGEN "Use a system Eigen installation" OFF)

if(TARGET Eigen3::Eigen)
  return()
endif()

if(USE_SYSTEM_EIGEN)
  find_package(Eigen3 3.3 REQUIRED NO_MODULE)
  return()
endif()

FetchContent_Declare(
  parte_eigen
  URL "https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.gz"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

message(STATUS "Downloading Eigen 5.0.0")
FetchContent_MakeAvailable(parte_eigen)
if(NOT TARGET Eigen3::Eigen)
  message(FATAL_ERROR "The downloaded Eigen archive did not define Eigen3::Eigen")
endif()
