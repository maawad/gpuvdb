cmake_minimum_required(VERSION 3.21)
project(gpuvdb LANGUAGES CXX HIP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# Find HIP
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to HIP install")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to HIP install")
    endif()
endif()

list(APPEND CMAKE_PREFIX_PATH ${HIP_PATH})
find_package(hip REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Header-only library
add_library(gpuvdb INTERFACE)
target_include_directories(gpuvdb INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Example: simple_sphere
add_executable(simple_sphere examples/simple_sphere.hip)
target_link_libraries(simple_sphere PRIVATE gpuvdb hip::device)
set_target_properties(simple_sphere PROPERTIES
  HIP_ARCHITECTURES "gfx942"
)

# Install headers
install(DIRECTORY include/ DESTINATION include)

