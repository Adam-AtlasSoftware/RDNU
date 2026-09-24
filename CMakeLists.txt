# RDNU: an open neural temporal upscaler for RDNA2/3, built on Arm's Neural Super Sampling.
#
# The Python half (ml/, scripts/) is not built here. This project builds the native runtime in
# runtime/: see docs/build-windows.md and docs/build-linux.md.

cmake_minimum_required(VERSION 3.21)

project(RDNU
    VERSION 1.0.0
    DESCRIPTION "RDNU: Arm NSS on D3D12 behind the FidelityFX API"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

enable_testing()
add_subdirectory(runtime)
