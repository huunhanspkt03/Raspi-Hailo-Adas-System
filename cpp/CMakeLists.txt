cmake_minimum_required(VERSION 3.16)
project(adas_cpp LANGUAGES CXX)

# ----------------------------------------------------------------------------
# Parallel C++ runtime for the ADAS pipeline on Raspberry Pi 5 + Hailo-8L.
# Python remains the reference / fallback implementation under ../src.
#
# This milestone implements the lane path end to end:
#   capture -> Hailo STDC1 -> lane decode -> centerline -> overlay -> display.
#
# YOLO26 raw-head decode and lane/object fusion are present. Object tracking,
# behavior smoothing, and RC control remain lightweight/TODO ports that point
# back to their Python source-of-truth modules.
# ----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()
option(ADAS_NATIVE_OPTIMIZE "Enable native CPU tuning flags when supported" ON)

# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
find_package(Threads REQUIRED)

# HailoRT C++ headers/lib. The Pi `hailo-all` package installs
#   /usr/include/hailo/        (hailort.hpp, ...)
#   /usr/lib/aarch64-linux-gnu/libhailort.so
# Optional: enable HAILO via -DADAS_USE_HAILO=ON (default ON on Linux aarch64).
option(ADAS_USE_HAILO "Link against HailoRT for on-device inference" ON)

set(ADAS_HAILO_AVAILABLE OFF)
if(ADAS_USE_HAILO)
    find_path(HAILORT_INCLUDE_DIR hailo/hailort.hpp
        HINTS /usr/include /usr/local/include
        PATH_SUFFIXES "")
    find_library(HAILORT_LIBRARY NAMES hailort
        HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib)
    if(HAILORT_INCLUDE_DIR AND HAILORT_LIBRARY)
        set(ADAS_HAILO_AVAILABLE ON)
        message(STATUS "HailoRT found: ${HAILORT_LIBRARY}")
        message(STATUS "HailoRT include: ${HAILORT_INCLUDE_DIR}")
    else()
        message(WARNING "HailoRT not found. Build will fail unless -DADAS_USE_HAILO=OFF "
                        "is also set. On Pi 5, install with: sudo apt install hailo-all.")
    endif()
endif()

# ----------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------
set(ADAS_SOURCES
    src/main.cpp
    src/cli.cpp
    src/capture.cpp
    src/hailo_inference.cpp
    src/lane_decode.cpp
    src/lane_geometry.cpp
    src/overlay.cpp
    src/runtime.cpp
    src/object_detection.cpp
    src/object_distance.cpp
    src/object_distance_smooth.cpp
    src/yolo26_raw_head.cpp
    src/behavior.cpp
    src/front_speed.cpp
    src/fusion.cpp
    src/lane_parity_components.cpp
    src/lane_parity_tracker.cpp
    src/lane_parity_render.cpp
    src/gps.cpp
    src/rc_actuator.cpp
)

add_executable(adas_cpp ${ADAS_SOURCES})

target_include_directories(adas_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(adas_cpp PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)

if(ADAS_NATIVE_OPTIMIZE AND CMAKE_BUILD_TYPE MATCHES "^[Rr]elease$")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mcpu=native" ADAS_CXX_HAS_MCPU_NATIVE)
    if(ADAS_CXX_HAS_MCPU_NATIVE)
        target_compile_options(adas_cpp PRIVATE -mcpu=native)
        message(STATUS "Native CPU tuning enabled: -mcpu=native")
    else()
        check_cxx_compiler_flag("-mtune=native" ADAS_CXX_HAS_MTUNE_NATIVE)
        if(ADAS_CXX_HAS_MTUNE_NATIVE)
            target_compile_options(adas_cpp PRIVATE -mtune=native)
            message(STATUS "Native CPU tuning enabled: -mtune=native")
        endif()
    endif()
endif()

if(ADAS_HAILO_AVAILABLE)
    target_include_directories(adas_cpp PRIVATE ${HAILORT_INCLUDE_DIR})
    target_link_libraries(adas_cpp PRIVATE ${HAILORT_LIBRARY})
    target_compile_definitions(adas_cpp PRIVATE ADAS_HAS_HAILO=1)
else()
    target_compile_definitions(adas_cpp PRIVATE ADAS_HAS_HAILO=0)
    message(STATUS "Building adas_cpp WITHOUT Hailo (stub inference for dev). "
                   "On Pi 5 production builds, install hailo-all.")
endif()

install(TARGETS adas_cpp RUNTIME DESTINATION bin)
