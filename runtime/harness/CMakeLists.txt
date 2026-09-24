# Windows bring-up tools on the shared DX12 helpers (rdnu_dx12.h). They compile their shaders at
# run time with the dxcompiler.dll staged next to them.
#
#   rdnu_engine / run_engine   the fp32 and INT8 reference engine: NSS against the exporter golden
#   run_engine_rdg             the same engine over the retired RDG model's goldens (dump_golden.py)
#   rdnu_harness               one layer against its golden
#   rdnu_wmma_probe            a single 16x16x16 INT8 wave-matrix tile on RDNA3

set(RDNU_FFX_BIN "${RDNU_SDK_DIR}/Kits/Cauldron2/dx12/vcpkg_installed/x64-windows/x64-windows/bin")
set(RDNU_HARNESS_SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders")

function(rdnu_bringup name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE d3d12 dxgi dxguid)
    target_compile_definitions(${name} PRIVATE "RDNU_AGS_INC=\"${RDNU_AMDEXT_INC}\"")
    set(_copy
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${RDNU_FFX_BIN}/dxcompiler.dll" "${RDNU_FFX_BIN}/dxil.dll"
                "$<TARGET_FILE_DIR:${name}>")
    foreach(shader IN LISTS ARGN)
        list(APPEND _copy COMMAND ${CMAKE_COMMAND} -E copy_if_different "${RDNU_HARNESS_SHADERS}/${shader}" "$<TARGET_FILE_DIR:${name}>")
    endforeach()
    add_custom_command(TARGET ${name} POST_BUILD ${_copy} VERBATIM)
endfunction()

rdnu_bringup(rdnu_harness rdnu_harness.cpp conv2d.hlsl gelu.hlsl first_conv.hlsl)
rdnu_bringup(rdnu_engine rdnu_engine.cpp
    conv2d.hlsl gelu.hlsl mul.hlsl add.hlsl concat.hlsl l2norm_spatial.hlsl ctr_scores.hlsl softmax_rows.hlsl
    ctr_apply.hlsl resize.hlsl pixelshuffle.hlsl axpy.hlsl conv2d_int8.hlsl relu.hlsl sigmoid.hlsl upsample_nearest.hlsl)
rdnu_bringup(rdnu_wmma_probe rdnu_wmma_probe.cpp wmma_gemm.hlsl)
rdnu_bringup(rdnu_wmma_conv1x1 rdnu_wmma_conv1x1.cpp wmma_conv1x1.hlsl)

add_custom_target(run_engine
    COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN_DIR}/nss_backbone.rdnut"
    COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN_DIR}/nss_backbone_int8.rdnut"
    DEPENDS rdnu_engine
    USES_TERMINAL
    COMMENT "NSS reference engine, fp32 and INT8, against the exporter golden")

set(_rdg ffn_block dfm_block encode_layer hfb_block ctr_block decode_layer upsample_block upsample_head
    full_model int8_first_conv int8_ffn_3x3 int8_ffn_1x1 int8_dfm_dw3x3 int8_down_s2 full_model_int8)
set(_rdg_cmds)
foreach(g IN LISTS _rdg)
    list(APPEND _rdg_cmds COMMAND "$<TARGET_FILE:rdnu_engine>" "${RDNU_GOLDEN_DIR}/${g}.rdnut")
endforeach()
add_custom_target(run_engine_rdg ${_rdg_cmds}
    DEPENDS rdnu_engine
    USES_TERMINAL
    COMMENT "Reference engine over the RDG goldens")

add_custom_target(run_wmma_probe COMMAND "$<TARGET_FILE:rdnu_wmma_probe>" DEPENDS rdnu_wmma_probe USES_TERMINAL)
add_custom_target(run_wmma_conv1x1
    COMMAND "$<TARGET_FILE:rdnu_wmma_conv1x1>" "${RDNU_GOLDEN_DIR}/int8_ffn_1x1.rdnut"
    DEPENDS rdnu_wmma_conv1x1
    USES_TERMINAL)
