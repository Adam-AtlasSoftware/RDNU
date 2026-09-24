# RDNU native runtime: Arm's NSS on D3D12 behind the FidelityFX API.
#
#   rdnu_core      platform-free network planner (engine core)
#   rdnu_shaderc   offline compiler: every shader to DXIL, embedded with the model
#   rdnu_dx12      network recorder, FidelityFX backend and Arm's NSS component
#   rdnu_ffx       the FidelityFX API DLL (Windows), installed under three names:
#                    amd_fidelityfx_dx12.dll            FSR 3.1 games, OptiScaler
#                    amd_fidelityfx_upscaler_dx12.dll   FidelityFX SDK 2 games (AMD's loader)
#                    rdnu_dx12.dll                      hosts that load RDNU explicitly
#   tests          ctest: core tests everywhere, D3D12 tests on Linux through vkd3d-proton
#
# See docs/build-windows.md and docs/build-linux.md.

set(RDNU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

set(RDNU_SDK_DIR "${RDNU_ROOT}/external/FidelityFX-SDK_WithFSR4" CACHE PATH
    "FidelityFX SDK (the fork): FidelityFX API and AMD extension headers")
set(RDNU_NSS_SDK_DIR "${RDNU_ROOT}/external/neural-graphics-sdk-for-game-engines/sdk" CACHE PATH
    "Arm neural graphics SDK: the NSS component")
set(RDNU_MODEL_DIR "${RDNU_ROOT}/models/NSS_INT8" CACHE PATH "nss.rdnm and nss_w8.bin")
set(RDNU_GOLDEN_DIR "${RDNU_ROOT}/tools/golden" CACHE PATH "nss_export.py and nss_pass_frames.py output, for the tests")
set(RDNU_VKD3D_DIR "" CACHE PATH "vkd3d-proton install prefix (Linux D3D12 tests; runtime/tools/setup_vkd3d_proton.sh)")
set(RDNU_SHADERC "" CACHE FILEPATH "rdnu_shaderc for the build machine, when cross compiling")

set(RDNU_FFX_API_INC "${RDNU_SDK_DIR}/Kits/FidelityFX/api/include")
set(RDNU_FFX_UPSCALE_INC "${RDNU_SDK_DIR}/Kits/FidelityFX/upscalers/include")
set(RDNU_AMDEXT_INC "${RDNU_SDK_DIR}/Kits/FidelityFX/api/internal/dx12/AmdExtD3D")
set(RDNU_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen")

if(NOT EXISTS "${RDNU_NSS_SDK_DIR}/src/components/nss/ffx_nss.cpp")
    message(FATAL_ERROR "RDNU: Arm's SDK not found at ${RDNU_NSS_SDK_DIR} (git submodule update --init)")
endif()
set(RDNU_HAVE_FFX_API OFF)
if(EXISTS "${RDNU_FFX_API_INC}/ffx_api.h")
    set(RDNU_HAVE_FFX_API ON)
else()
    message(STATUS "RDNU: no FidelityFX SDK at ${RDNU_SDK_DIR}; the FidelityFX API DLL and its tests are skipped")
endif()
set(RDNU_HAVE_AMDEXT OFF)
if(EXISTS "${RDNU_AMDEXT_INC}/AmdExtD3D.h")
    set(RDNU_HAVE_AMDEXT ON)
else()
    message(STATUS "RDNU: no AMD extension headers; RDNA3 runs the DP4a kernels")
endif()

if(MINGW)
    add_link_options(-static)  # no MinGW runtime DLLs to ship
endif()
if(NOT WIN32)
    set(CMAKE_CXX_EXTENSIONS ON)  # vkd3d-proton's headers
endif()
if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
endif()

# ---------------------------------------------------------------------------- core and shaders

add_library(rdnu_core STATIC src/engine/rdnu_engine_core.cpp)
target_include_directories(rdnu_core PUBLIC src/engine)

if(CMAKE_CROSSCOMPILING)
    if(NOT RDNU_SHADERC)
        message(FATAL_ERROR "RDNU: cross builds need RDNU_SHADERC, rdnu_shaderc built for this machine")
    endif()
    set(RDNU_SHADERC_CMD "${RDNU_SHADERC}")
else()
    add_executable(rdnu_shaderc tools/rdnu_shaderc.cpp)
    target_link_libraries(rdnu_shaderc PRIVATE rdnu_core)
    set(RDNU_SHADERC_CMD $<TARGET_FILE:rdnu_shaderc>)
endif()

find_program(RDNU_DXC NAMES dxc
    HINTS ENV DXC_DIR "${RDNU_SDK_DIR}/Kits/FidelityFX/tools/ffx_sc/bin"
    PATHS ENV PATH)
if(NOT RDNU_DXC AND WIN32)
    file(GLOB _kits "$ENV{ProgramFiles\(x86\)}/Windows Kits/10/bin/10.*/x64/dxc.exe")
    list(SORT _kits COMPARE NATURAL ORDER DESCENDING)
    list(GET _kits 0 RDNU_DXC)
endif()

set(RDNU_HAVE_DX12 OFF)
if(WIN32 OR RDNU_VKD3D_DIR)
    if(RDNU_DXC)
        set(RDNU_HAVE_DX12 ON)
    else()
        message(STATUS "RDNU: dxc not found (set RDNU_DXC); the D3D12 targets are skipped")
    endif()
endif()

if(RDNU_HAVE_DX12)
    file(GLOB RDNU_SHADER_SOURCES CONFIGURE_DEPENDS
        shaders/net/*.hlsl shaders/net/*.hlsli shaders/nss/*.hlsl shaders/nss/*.h shaders/nss/generated/*.h)
    set(_amdext_arg --no-wmma)
    if(RDNU_HAVE_AMDEXT)
        set(_amdext_arg --amd-ext "${RDNU_AMDEXT_INC}")
    endif()
    add_custom_command(
        OUTPUT "${RDNU_GEN_DIR}/rdnu_embedded.cpp"
        COMMAND "${RDNU_SHADERC_CMD}" --dxc "${RDNU_DXC}" --shaders "${RDNU_ROOT}/shaders" --model "${RDNU_MODEL_DIR}"
                ${_amdext_arg} --out "${RDNU_GEN_DIR}/rdnu_embedded.cpp"
        DEPENDS ${RDNU_SHADER_SOURCES} "${RDNU_MODEL_DIR}/nss.rdnm" "${RDNU_MODEL_DIR}/nss_w8.bin"
                $<$<NOT:$<BOOL:${CMAKE_CROSSCOMPILING}>>:rdnu_shaderc>
        COMMENT "Compiling the NSS shaders to DXIL"
        VERBATIM)
endif()

# ------------------------------------------------------------------------------------ D3D12

if(RDNU_HAVE_DX12)
    set(RDNU_ARM_SOURCES
        "${RDNU_NSS_SDK_DIR}/src/components/nss/ffx_nss.cpp"
        "${RDNU_NSS_SDK_DIR}/src/shared/ffx_object_management.cpp"
        "${RDNU_NSS_SDK_DIR}/src/shared/ffx_assert.cpp")
    set_source_files_properties(${RDNU_ARM_SOURCES} PROPERTIES COMPILE_OPTIONS $<IF:$<CXX_COMPILER_ID:MSVC>,/W0,-w>)

    add_library(rdnu_dx12 STATIC
        src/engine/rdnu_engine_dx12.cpp
        src/engine/rdnu_amd_ext.cpp
        src/backend_dx12/ffx_nss_dx12.cpp
        "${RDNU_GEN_DIR}/rdnu_embedded.cpp"
        ${RDNU_ARM_SOURCES})
    target_include_directories(rdnu_dx12 SYSTEM PUBLIC
        "${RDNU_NSS_SDK_DIR}/include" "${RDNU_NSS_SDK_DIR}/src/components/nss" "${RDNU_NSS_SDK_DIR}/src/shared")
    target_link_libraries(rdnu_dx12 PUBLIC rdnu_core)
    set_target_properties(rdnu_dx12 PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(RDNU_HAVE_AMDEXT AND WIN32)
        target_include_directories(rdnu_dx12 SYSTEM PRIVATE "${RDNU_AMDEXT_INC}")
        target_compile_definitions(rdnu_dx12 PRIVATE RDNU_HAVE_AMDEXT)
    endif()
    if(WIN32)
        target_compile_definitions(rdnu_dx12 PUBLIC WIN32_LEAN_AND_MEAN NOMINMAX)
        target_link_libraries(rdnu_dx12 PUBLIC d3d12)
    else()
        # vkd3d-proton's D3D12 headers and library behind the Windows header names
        target_include_directories(rdnu_dx12 PUBLIC tests/vkd3d)
        target_include_directories(rdnu_dx12 SYSTEM PUBLIC "${RDNU_VKD3D_DIR}/include/vkd3d-proton")
        target_link_directories(rdnu_dx12 PUBLIC "${RDNU_VKD3D_DIR}/lib")
        target_link_libraries(rdnu_dx12 PUBLIC vkd3d-proton-d3d12)
        target_compile_options(rdnu_dx12 PUBLIC "-D__declspec(x)=")  # ffx_api.h exports
    endif()
endif()

if(RDNU_HAVE_DX12 AND RDNU_HAVE_FFX_API)
    add_library(rdnu_provider OBJECT src/provider/rdnu_ffx_api.cpp src/provider/rdnu_upscale_map.cpp src/provider/rdnu_capture.cpp)
    target_include_directories(rdnu_provider SYSTEM PUBLIC "${RDNU_FFX_API_INC}" "${RDNU_FFX_UPSCALE_INC}")
    target_link_libraries(rdnu_provider PUBLIC rdnu_dx12)
    set_target_properties(rdnu_provider PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(TARGET rdnu_provider)
    # a whole upscale through the FidelityFX API, GPU timed
    add_executable(rdnu_bench harness/rdnu_bench.cpp)
    target_link_libraries(rdnu_bench PRIVATE rdnu_provider)
endif()

if(WIN32 AND TARGET rdnu_provider)
    add_library(rdnu_ffx SHARED $<TARGET_OBJECTS:rdnu_provider>)
    target_link_libraries(rdnu_ffx PRIVATE rdnu_dx12)
    set_target_properties(rdnu_ffx PROPERTIES OUTPUT_NAME amd_fidelityfx_dx12 PREFIX "")
    add_custom_command(TARGET rdnu_ffx POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:rdnu_ffx> "$<TARGET_FILE_DIR:rdnu_ffx>/amd_fidelityfx_upscaler_dx12.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:rdnu_ffx> "$<TARGET_FILE_DIR:rdnu_ffx>/rdnu_dx12.dll"
        VERBATIM)
    install(FILES $<TARGET_FILE:rdnu_ffx> "$<TARGET_FILE_DIR:rdnu_ffx>/amd_fidelityfx_upscaler_dx12.dll"
                  "$<TARGET_FILE_DIR:rdnu_ffx>/rdnu_dx12.dll" DESTINATION bin)
endif()

if(TARGET rdnu_dx12)
    # the production network against the exporter's golden, and its timing
    add_executable(rdnu_prod harness/rdnu_prod.cpp)
    target_link_libraries(rdnu_prod PRIVATE rdnu_dx12)
endif()

# ------------------------------------------------------------------------------------ tests


add_executable(test_engine_core tests/core/test_engine_core.cpp)
target_link_libraries(test_engine_core PRIVATE rdnu_core)
if(EXISTS "${RDNU_GOLDEN_DIR}/nss_prod.rdnut")
    add_test(NAME engine_core COMMAND test_engine_core "${RDNU_MODEL_DIR}" "${RDNU_GOLDEN_DIR}")
else()
    add_test(NAME engine_core COMMAND test_engine_core "${RDNU_MODEL_DIR}")
endif()

if(RDNU_HAVE_FFX_API)
    add_executable(test_upscale_map tests/core/test_upscale_map.cpp src/provider/rdnu_upscale_map.cpp)
    target_include_directories(test_upscale_map SYSTEM PRIVATE
        "${RDNU_NSS_SDK_DIR}/include" "${RDNU_FFX_API_INC}" "${RDNU_FFX_UPSCALE_INC}")
    if(NOT WIN32)
        target_compile_options(test_upscale_map PRIVATE "-D__declspec(x)=")
    endif()
    add_test(NAME upscale_map COMMAND test_upscale_map)
endif()

# D3D12: on Windows the GPU, on Linux Mesa lavapipe through vkd3d-proton
if(TARGET rdnu_dx12)
    set(_env "")
    if(NOT WIN32)
        set(_env "LD_LIBRARY_PATH=${RDNU_VKD3D_DIR}/lib" "VKD3D_SHADER_MODEL=6_6" "VKD3D_DEBUG=err")
    endif()
    add_executable(test_nss_dx12 tests/vkd3d/test_nss_dx12.cpp)
    target_link_libraries(test_nss_dx12 PRIVATE rdnu_dx12)
    set(_tests)
    if(TARGET rdnu_provider)
        add_executable(test_fsr_api tests/vkd3d/test_fsr_api.cpp)
        target_link_libraries(test_fsr_api PRIVATE rdnu_provider)
        target_include_directories(test_fsr_api SYSTEM PRIVATE "${RDNU_SDK_DIR}/Kits/FidelityFX/api/internal")
    endif()
    if(EXISTS "${RDNU_GOLDEN_DIR}/nss_prod.rdnut")
        add_test(NAME dx12_network COMMAND rdnu_prod "${RDNU_GOLDEN_DIR}")
        add_test(NAME dx12_network_drs COMMAND rdnu_prod "${RDNU_GOLDEN_DIR}" --drs)
        list(APPEND _tests dx12_network dx12_network_drs)
    endif()
    if(EXISTS "${RDNU_GOLDEN_DIR}/nss_frames.rdnut")
        add_test(NAME dx12_nss COMMAND test_nss_dx12 "${RDNU_GOLDEN_DIR}" --frames 8)
        add_test(NAME dx12_nss_1_5x COMMAND test_nss_dx12 "${RDNU_GOLDEN_DIR}" --frames 4 --scale 1.5)
        list(APPEND _tests dx12_nss dx12_nss_1_5x)
        if(TARGET test_fsr_api)
            add_test(NAME fsr_api COMMAND test_fsr_api "${RDNU_GOLDEN_DIR}" --frames 8)
            list(APPEND _tests fsr_api)
            find_package(Python3 COMPONENTS Interpreter)
            if(Python3_FOUND)
                # dispatches captured by the provider, scored against the ground truth
                set(_capture "${CMAKE_CURRENT_BINARY_DIR}/capture")
                add_test(NAME fsr_capture COMMAND ${CMAKE_COMMAND} -E env "RDNU_CAPTURE=${_capture}" RDNU_CAPTURE_FRAMES=6
                         $<TARGET_FILE:test_fsr_api> "${RDNU_GOLDEN_DIR}" --frames 6)
                add_test(NAME capture_compare COMMAND Python3::Interpreter "${RDNU_ROOT}/tools/eval/compare.py" "${_capture}"
                         --context 0 --truth "${RDNU_GOLDEN_DIR}/nss_frames.rdnut" --min-psnr 27)
                set_tests_properties(fsr_capture PROPERTIES FIXTURES_SETUP capture)
                set_tests_properties(capture_compare PROPERTIES FIXTURES_REQUIRED capture)
                list(APPEND _tests fsr_capture)
            endif()
        endif()
    endif()
    if(_tests AND _env)
        set_tests_properties(${_tests} PROPERTIES ENVIRONMENT "${_env}")
    endif()
endif()

# The HLSL passes and kernels against Arm's GLSL on Vulkan (lavapipe); compiles shaders at run
# time with dxc and glslangValidator.
find_package(Vulkan QUIET)
if(Vulkan_FOUND AND NOT WIN32 AND RDNU_DXC)
    add_library(rdnu_vk STATIC tests/vk/vk_compute.cpp)
    target_link_libraries(rdnu_vk PUBLIC Vulkan::Vulkan rdnu_core)
    target_compile_definitions(rdnu_vk PUBLIC "RDNU_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"")
    target_include_directories(rdnu_vk SYSTEM PUBLIC "${RDNU_NSS_SDK_DIR}/include" "${RDNU_NSS_SDK_DIR}/src/components/nss")
    add_executable(test_net_vk tests/vk/test_net_vk.cpp)
    add_executable(test_passes_vk tests/vk/test_passes_vk.cpp)
    target_link_libraries(test_net_vk PRIVATE rdnu_vk)
    target_link_libraries(test_passes_vk PRIVATE rdnu_vk)
    if(EXISTS "${RDNU_GOLDEN_DIR}/nss_frames.rdnut")
        add_test(NAME vk_network COMMAND test_net_vk "${RDNU_GOLDEN_DIR}")
        add_test(NAME vk_passes COMMAND test_passes_vk "${RDNU_GOLDEN_DIR}" --frames 2 --sharpen 0.5)
        set_tests_properties(vk_network vk_passes PROPERTIES ENVIRONMENT "DXC=${RDNU_DXC}")
    endif()
endif()

if(RDNU_DXC AND NOT CMAKE_CROSSCOMPILING)
    set(_ags "")
    if(RDNU_HAVE_AMDEXT)
        set(_ags "AGS_INC=${RDNU_AMDEXT_INC}")
    endif()
    find_package(Python3 COMPONENTS Interpreter)
    add_custom_target(check_shaders
        COMMAND ${CMAKE_COMMAND} -E env "DXC=${RDNU_DXC}" "RDNU_SHADERC=$<TARGET_FILE:rdnu_shaderc>" ${_ags}
                sh "${RDNU_ROOT}/tools/check_shaders.sh"
        COMMAND Python3::Interpreter "${RDNU_ROOT}/tools/port_nss_shaders.py" --check
        DEPENDS rdnu_shaderc
        USES_TERMINAL
        COMMENT "DXC over every shader; the generated NSS headers are current")
endif()

# --------------------------------------------------------------------------------- FSR sample

# AMD's FSR sample with RDNU as its upscaler DLL (runtime/integration/*.patch applied to the SDK).
if(WIN32 AND MSVC AND TARGET rdnu_ffx)
    set(RDNU_SAMPLE_BIN "${RDNU_SDK_DIR}/Samples/Upscalers/FidelityFX_FSR/dx12/x64/$<CONFIG>")
    add_custom_target(fsr_sample
        COMMAND pwsh -NoProfile -File "${RDNU_ROOT}/tools/build_sample.ps1" -Config $<CONFIG>
        DEPENDS rdnu_ffx
        USES_TERMINAL
        COMMENT "MSBuild: FidelityFX FSR sample ($<CONFIG>)")
    add_custom_target(run_fsr_sample
        COMMAND "${RDNU_SAMPLE_BIN}/FidelityFX_FSR.exe"
        WORKING_DIRECTORY "${RDNU_SAMPLE_BIN}"
        DEPENDS fsr_sample
        USES_TERMINAL)
endif()

# ------------------------------------------------------------------- reference and bring-up

# The fp32/INT8 reference engine and the layer harness (harness/rdnu_dx12.h); Windows only.
if(WIN32 AND MSVC)
    add_subdirectory(harness)
endif()
