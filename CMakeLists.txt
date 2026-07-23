# RDNU — Radeon Decoupled Neural Upscaler
#
# This repository has two halves:
#   * Python training half  — repo root (RDG/, scripts/); not built by CMake.
#   * Native runtime half    — runtime/ (DX12/HLSL today; Vulkan/CUDA planned).
#
# This CMake project drives the *native runtime* half so the whole thing can be
# built and edited from VS Code without opening the Visual Studio IDE.
# See docs/build-windows.md and docs/build-linux.md.

cmake_minimum_required(VERSION 3.21)

project(RDNU
    VERSION 0.1.0
    DESCRIPTION "Radeon Decoupled Neural Upscaler — native runtime"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Keep the module path available for helper scripts under cmake/.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_subdirectory(runtime)
