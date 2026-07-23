# RDNU native runtime
#
# What this builds today:
#   * validate_shaders  — best-effort DXC validation of the HLSL compute shaders
#   * rdnu_test_dxc / rdnu_test_compile — tiny Windows toolchain smoke tests (opt-in)
#
# What it does NOT build (yet):
#   The production DX12 backend (src/rdg_dx12_backend.cpp + src/ffx_provider_rdg.h)
#   currently compiles *inside* the FidelityFX FSR sample, because it needs the full
#   Cauldron/FidelityFX include+link tree. That sample is built via MSBuild from VS Code
#   (see ../.vscode/tasks.json "Build FSR Sample"). A standalone, cross-platform backend
#   for Linux/Nvidia will be added here behind RDNU_BACKEND_VULKAN. See docs/build-linux.md.

option(RDNU_BUILD_TESTS      "Build the small DXC/d3dcompiler toolchain smoke tests (Windows only)" ON)
option(RDNU_VALIDATE_SHADERS "Add a best-effort HLSL validation target (needs dxc on PATH)"          ON)
option(RDNU_BACKEND_VULKAN   "Build the (not-yet-implemented) cross-platform Vulkan backend"          OFF)

set(RDNU_ROOT        "${CMAKE_CURRENT_SOURCE_DIR}")
set(RDNU_SHADER_DIR  "${RDNU_ROOT}/shaders")
set(RDNU_SDK_DIR     "${RDNU_ROOT}/external/FidelityFX-SDK_WithFSR4")
set(RDNU_AMD_HLSL_INC "${RDNU_SDK_DIR}/Kits/FidelityFX/api/internal/dx12/AmdExtD3D")

# ---------------------------------------------------------------------------
# Windows toolchain smoke tests (opt-in, EXCLUDE_FROM_ALL so they never break
# a default build if the SDK libs aren't on the linker search path).
# ---------------------------------------------------------------------------
if(RDNU_BUILD_TESTS AND WIN32)
    # test_dxc.cpp pulls in dxcompiler.lib and test_compile.cpp pulls in
    # d3dcompiler.lib via #pragma comment(lib, ...); both ship with the Windows SDK.
    add_executable(rdnu_test_dxc     EXCLUDE_FROM_ALL tests/test_dxc.cpp)
    add_executable(rdnu_test_compile EXCLUDE_FROM_ALL tests/test_compile.cpp)
    message(STATUS "RDNU: smoke tests available — build with target rdnu_test_dxc / rdnu_test_compile")
endif()

# ---------------------------------------------------------------------------
# HLSL shader validation (best-effort). Runs DXC over each compute shader and
# reports pass/fail without failing the build (the WMMA shader is WIP and needs
# AMD wave-matrix intrinsics, so it is expected to fail until finished).
# ---------------------------------------------------------------------------
if(RDNU_VALIDATE_SHADERS)
    find_program(RDNU_DXC
        NAMES dxc dxc.exe
        HINTS "${RDNU_SDK_DIR}/Kits/FidelityFX/tools/ffx_sc/bin"
        PATHS ENV PATH)
    if(RDNU_DXC)
        add_custom_target(validate_shaders
            COMMAND ${CMAKE_COMMAND}
                    -DDXC=${RDNU_DXC}
                    -DSHADER_DIR=${RDNU_SHADER_DIR}
                    -DAMD_INC=${RDNU_AMD_HLSL_INC}
                    -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}
                    -P "${CMAKE_SOURCE_DIR}/cmake/validate_shaders.cmake"
            COMMENT "Validating HLSL shaders with DXC (best-effort; WIP shaders may fail)"
            VERBATIM)
        message(STATUS "RDNU: shader validation available via 'dxc' at ${RDNU_DXC} (target: validate_shaders)")
    else()
        message(STATUS "RDNU: dxc not found — 'validate_shaders' target skipped. "
                       "Install the DirectX Shader Compiler and re-run CMake to enable it.")
    endif()
endif()

# ---------------------------------------------------------------------------
# Placeholder for the future cross-platform (Linux/Nvidia) inference backend.
# ---------------------------------------------------------------------------
if(RDNU_BACKEND_VULKAN)
    message(WARNING "RDNU_BACKEND_VULKAN is set, but the Vulkan backend is not implemented yet. "
                    "See docs/build-linux.md for the porting plan.")
endif()
