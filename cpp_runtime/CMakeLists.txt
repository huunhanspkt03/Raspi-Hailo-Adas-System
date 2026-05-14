cmake_minimum_required(VERSION 3.16)
project(adas_cpp_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(USE_HAILO "Link against HailoRT for on-device inference" OFF)

find_package(OpenCV CONFIG QUIET COMPONENTS core imgproc imgcodecs videoio highgui)
if(NOT OpenCV_FOUND)
    find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs videoio highgui)
endif()
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR
        "OpenCV C++ development package was not found.\n"
        "This runtime needs OpenCV headers/libs plus OpenCVConfig.cmake.\n\n"
        "Windows options:\n"
        "  1) vcpkg:\n"
        "     vcpkg install opencv4:x64-windows\n"
        "     cmake -S cpp_runtime -B cpp_runtime/build -DUSE_HAILO=OFF "
        "-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake\n"
        "  2) Prebuilt OpenCV:\n"
        "     cmake -S cpp_runtime -B cpp_runtime/build -DUSE_HAILO=OFF "
        "-DOpenCV_DIR=C:/path/to/opencv/build/x64/vc16/lib\n\n"
        "Raspberry Pi / Debian:\n"
        "     sudo apt install libopencv-dev\n")
endif()
message(STATUS "OpenCV: ${OpenCV_VERSION}")

set(ADAS_SOURCES
    src/main.cpp
    src/camera_source.cpp
    src/preprocess.cpp
    src/deeplab_postprocess.cpp
    src/lane_center_path.cpp
    src/yolo26s_postprocess.cpp
    src/object_localizer.cpp
    src/controller.cpp
    src/visualization.cpp
    src/hailo_backend.cpp
)

add_executable(adas_runtime ${ADAS_SOURCES})
target_include_directories(adas_runtime PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(adas_runtime PRIVATE ${OpenCV_LIBS})
if(MSVC)
    target_compile_options(adas_runtime PRIVATE /W3)
else()
    target_compile_options(adas_runtime PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

if(USE_HAILO)
    find_path(HAILORT_INCLUDE_DIR hailo/hailort.hpp
        HINTS /usr/include /usr/local/include)
    find_library(HAILORT_LIBRARY hailort
        HINTS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu)
    if(NOT HAILORT_INCLUDE_DIR OR NOT HAILORT_LIBRARY)
        message(FATAL_ERROR "USE_HAILO=ON but HailoRT headers/libs not found. "
            "Install hailort-dev package on the Pi.")
    endif()
    message(STATUS "HailoRT: ${HAILORT_LIBRARY}")
    target_include_directories(adas_runtime PRIVATE ${HAILORT_INCLUDE_DIR})
    target_link_libraries(adas_runtime PRIVATE ${HAILORT_LIBRARY})
    target_compile_definitions(adas_runtime PRIVATE ADAS_USE_HAILO=1)
else()
    message(STATUS "Building without HailoRT — stub backend will produce empty inference results.")
    target_compile_definitions(adas_runtime PRIVATE ADAS_USE_HAILO=0)
endif()
