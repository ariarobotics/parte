option(USE_SYSTEM_OPEN3D "Use a system Open3D installation" OFF)

if(TARGET Open3D::Open3D)
  return()
endif()

if(USE_SYSTEM_OPEN3D)
  find_package(Open3D REQUIRED)
  return()
endif()

set(PARTE_OPEN3D_VERSION "0.19.0")

if(WIN32)
    set(_open3d_archive
        "open3d-devel-windows-amd64-${PARTE_OPEN3D_VERSION}.zip"
    )

elseif(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        set(_open3d_archive
            "open3d-devel-darwin-arm64-${PARTE_OPEN3D_VERSION}.tar.xz"
        )
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
        set(_open3d_archive
            "open3d-devel-darwin-x86_64-${PARTE_OPEN3D_VERSION}.tar.xz"
        )
    else()
        message(FATAL_ERROR "Unsupported macOS architecture")
    endif()

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        set(_open3d_archive
            "open3d-devel-linux-x86_64-cxx11-abi-${PARTE_OPEN3D_VERSION}.tar.xz"
        )
    else()
        message(FATAL_ERROR "Unsupported Linux architecture")
    endif()

else()
    message(FATAL_ERROR "Unsupported platform")
endif()

set(_open3d_url
    "https://github.com/isl-org/Open3D/releases/download/v${PARTE_OPEN3D_VERSION}/${_open3d_archive}"
)

FetchContent_Declare(
    parte_open3d
    URL "${_open3d_url}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
message(STATUS "Downloading prebuilt Open3D ${PARTE_OPEN3D_VERSION}")

FetchContent_MakeAvailable(parte_open3d)

find_package(
    Open3D REQUIRED CONFIG
    PATHS "${parte_open3d_SOURCE_DIR}"
    NO_DEFAULT_PATH
)
