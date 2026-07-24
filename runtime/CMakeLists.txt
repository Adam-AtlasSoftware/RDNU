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
# Standalone DX12 validation harness (rdnu_harness): runs a single RDNU layer on
# the GPU and compares it to the PyTorch golden (runtime/tools/golden/*.rdnut).
# A real CMake-built executable (no Cauldron) for layer-by-layer numerical bring-up.
#   * target 'rdnu_harness' -> builds the harness
#   * target 'run_harness'  -> builds it, then runs it on the first_conv golden bundle
# ---------------------------------------------------------------------------
if(WIN32)
    set(RDNU_FFX_BIN "${RDNU_SDK_DIR}/Kits/Cauldron2/dx12/vcpkg_installed/x64-windows/x64-windows/bin")
    add_executable(rdnu_harness harness/rdnu_harness.cpp)
    target_link_libraries(rdnu_harness PRIVATE d3d12 dxgi dxguid)
    # DXC (AMD build, wave-matrix capable) + dxil + the harness shaders, staged next to
    # the exe. conv2d.hlsl is the general direct-conv reference the harness defaults to;
    # first_conv.hlsl is kept for reference (superseded by conv2d.hlsl).
    add_custom_command(TARGET rdnu_harness POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${RDNU_FFX_BIN}/dxcompiler.dll" "$<TARGET_FILE_DIR:rdnu_harness>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${RDNU_FFX_BIN}/dxil.dll"       "$<TARGET_FILE_DIR:rdnu_harness>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/harness/shaders/conv2d.hlsl"     "$<TARGET_FILE_DIR:rdnu_harness>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/harness/shaders/gelu.hlsl"       "$<TARGET_FILE_DIR:rdnu_harness>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/harness/shaders/first_conv.hlsl" "$<TARGET_FILE_DIR:rdnu_harness>"
        VERBATIM)
    # run_harness sweeps every validated primitive against the PyTorch golden. Conv ops use
    # the default conv2d kernel; GELU passes an explicit shader+entry (pointwise, no weights).
    set(RDNU_GOLDEN "${RDNU_ROOT}/tools/golden")
    add_custom_target(run_harness
        COMMAND ${CMAKE_COMMAND} -E echo "== first_conv    (3x3,  3->36,  g1)  =="
        COMMAND "$<TARGET_FILE:rdnu_harness>" "${RDNU_GOLDEN}/first_conv.rdnut"
        COMMAND ${CMAKE_COMMAND} -E echo "== ffn.fn.0      (3x3,  36->72, g1)  =="
        COMMAND "$<TARGET_FILE:rdnu_harness>" "${RDNU_GOLDEN}/ffn_conv3x3.rdnut"
        COMMAND ${CMAKE_COMMAND} -E echo "== ffn.fn.2      (1x1,  72->36, g1)  =="
        COMMAND "$<TARGET_FILE:rdnu_harness>" "${RDNU_GOLDEN}/ffn_conv1x1.rdnut"
        COMMAND ${CMAKE_COMMAND} -E echo "== dfm.proj_in.0 (3x3,  36->72, g36) =="
        COMMAND "$<TARGET_FILE:rdnu_harness>" "${RDNU_GOLDEN}/dfm_dw3x3.rdnut"
        COMMAND ${CMAKE_COMMAND} -E echo "== cyclefc mlp_h (1x7 shift,    g36) =="
        COMMAND "$<TARGET_FILE:rdnu_harness>" "${RDNU_GOLDEN}/cyclefc_shift_1x7.rdnut"
        COMMAND ${CMAKE_COMMAND} -E echo "== cyclefc mlp_w (7x1 shift,    g36) =="
        COMMAND "$<TARGET_FILE:rdnu_harness>" "${RDNU_GOLDEN}/cyclefc_shift_7x1.rdnut"
        COMMAND ${CMAKE_COMMAND} -E echo "== ffn.fn.1      (GELU, erf)         =="
        COMMAND "$<TARGET_FILE:rdnu_harness>" "${RDNU_GOLDEN}/gelu.rdnut" "$<TARGET_FILE_DIR:rdnu_harness>/gelu.hlsl" "gelu_CS"
        DEPENDS rdnu_harness
        USES_TERMINAL
        COMMENT "Validating conv + shift + GELU primitives vs golden")

    # rdnu_engine: chains several kernels on the GPU through ping-pong feature buffers to
    # run a whole block (currently CCM/FFN) and validates the end-to-end result vs golden.
    # Shares the DX12/.rdnut helpers with the harness via harness/rdnu_dx12.h.
    add_executable(rdnu_engine harness/rdnu_engine.cpp)
    target_link_libraries(rdnu_engine PRIVATE d3d12 dxgi dxguid)
    add_custom_command(TARGET rdnu_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${RDNU_FFX_BIN}/dxcompiler.dll" "$<TARGET_FILE_DIR:rdnu_engine>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${RDNU_FFX_BIN}/dxil.dll"       "$<TARGET_FILE_DIR:rdnu_engine>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/harness/shaders/conv2d.hlsl" "$<TARGET_FILE_DIR:rdnu_engine>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/harness/shaders/gelu.hlsl"   "$<TARGET_FILE_DIR:rdnu_engine>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/harness/shaders/mul.hlsl"    "$<TARGET_FILE_DIR:rdnu_engine>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/harness/shaders/add.hlsl"    "$<TARGET_FILE_DIR:rdnu_engine>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/harness/shaders/concat.hlsl" "$<TARGET_FILE_DIR:rdnu_engine>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/harness/shaders/l2norm_spatial.hlsl" "$<TARGET_FILE_DIR:rdnu_engine>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/harness/shaders/ctr_scores.hlsl"     "$<TARGET_FILE_DIR:rdnu_engine>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/harness/shaders/softmax_rows.hlsl"   "$<TARGET_FILE_DIR:rdnu_engine>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/harness/shaders/ctr_apply.hlsl"      "$<TARGET_FILE_DIR:rdnu_engine>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/harness/shaders/resize.hlsl"        "$<TARGET_FILE_DIR:rdnu_engine>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/harness/shaders/pixelshuffle.hlsl" "$<TARGET_FILE_DIR:rdnu_engine>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/harness/shaders/axpy.hlsl"        "$<TARGET_FILE_DIR:rdnu_engine>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/harness/shaders/conv2d_int8.hlsl" "$<TARGET_FILE_DIR:rdnu_engine>"
        VERBATIM)
    add_custom_target(run_engine
        COMMAND ${CMAKE_COMMAND} -E echo "== CCM/FFN block  (conv3x3 -> GELU -> conv1x1) =="
        COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN}/ffn_block.rdnut"
        COMMAND ${CMAKE_COMMAND} -E echo "== DFM block      (proj_in -> chunk -> mlp_h/w -> gate/merge -> proj_out) =="
        COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN}/dfm_block.rdnut"
        COMMAND ${CMAKE_COMMAND} -E echo "== EncodeLayer    (dfm(x)+x -> ffn(x)+x) =="
        COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN}/encode_layer.rdnut"
        COMMAND ${CMAKE_COMMAND} -E echo "== HFB block      (conv_x/conv_g -> cross-GELU-gate -> concat -> proj_out) =="
        COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN}/hfb_block.rdnut"
        COMMAND ${CMAKE_COMMAND} -E echo "== CTR block      (frame0: to_q/to_kv -> cosine attn -> proj_out) =="
        COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN}/ctr_block.rdnut"
        COMMAND ${CMAKE_COMMAND} -E echo "== DecodeLayer    (frame0: hfb(x,g)+x -> ctr(x,d)+x -> ffn(x)+x) =="
        COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN}/decode_layer.rdnut"
        COMMAND ${CMAKE_COMMAND} -E echo "== Upsampling     (interpolate x2 bilinear -> conv3x3) =="
        COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN}/upsample_block.rdnut"
        COMMAND ${CMAKE_COMMAND} -E echo "== UpSample head  (conv 36->12 -> PixelShuffle(2)) =="
        COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN}/upsample_head.rdnut"
        COMMAND ${CMAKE_COMMAND} -E echo "== WHOLE MODEL    (process_forward, frame0) =="
        COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN}/full_model.rdnut"
        COMMAND ${CMAKE_COMMAND} -E echo "== INT8 convs      (W8A8 vs numpy int8 ref: 3x3/1x1/depthwise/strided) =="
        COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN}/int8_first_conv.rdnut"
        COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN}/int8_ffn_3x3.rdnut"
        COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN}/int8_ffn_1x1.rdnut"
        COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN}/int8_dfm_dw3x3.rdnut"
        COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN}/int8_down_s2.rdnut"
        COMMAND ${CMAKE_COMMAND} -E echo "== WHOLE MODEL INT8 (W8A8 process_forward vs torch int8 ref) =="
        COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN}/full_model_int8.rdnut"
        DEPENDS rdnu_engine
        USES_TERMINAL
        COMMENT "Running the RDNU block engine on all composite blocks vs golden")
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
# Windows: build & run the FidelityFX FSR sample (with the RDNU upscaler integrated)
# straight from CMake, so VS Code's CMake Tools drives the whole integration:
#   * target 'fsr_sample'      -> MSBuilds the sample in the active config (Debug/Release)
#   * target 'run_fsr_sample'  -> builds it, then launches FidelityFX_FSR.exe
# The sample is an MSBuild/.vcxproj project; these targets just wrap build_sample.ps1.
# (For source-level debugging, use the F5 "Debug FSR Sample" config in .vscode/launch.json.)
# ---------------------------------------------------------------------------
if(WIN32)
    set(RDNU_SAMPLE_PS1 "${RDNU_ROOT}/tools/build_sample.ps1")
    set(RDNU_SAMPLE_BIN "${RDNU_SDK_DIR}/Samples/Upscalers/FidelityFX_FSR/dx12/x64/$<CONFIG>")

    add_custom_target(fsr_sample
        COMMAND pwsh -NoProfile -File "${RDNU_SAMPLE_PS1}" -Config $<CONFIG>
        USES_TERMINAL
        COMMENT "MSBuild: FidelityFX FSR sample (RDNU upscaler) — config $<CONFIG>")

    add_custom_target(run_fsr_sample
        COMMAND "${RDNU_SAMPLE_BIN}/FidelityFX_FSR.exe"
        WORKING_DIRECTORY "${RDNU_SAMPLE_BIN}"
        USES_TERMINAL
        COMMENT "Running FidelityFX_FSR.exe ($<CONFIG>)")
    add_dependencies(run_fsr_sample fsr_sample)
endif()

# ---------------------------------------------------------------------------
# Placeholder for the future cross-platform (Linux/Nvidia) inference backend.
# ---------------------------------------------------------------------------
if(RDNU_BACKEND_VULKAN)
    message(WARNING "RDNU_BACKEND_VULKAN is set, but the Vulkan backend is not implemented yet. "
                    "See docs/build-linux.md for the porting plan.")
endif()
