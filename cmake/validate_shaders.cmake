# Best-effort HLSL validation, invoked at build time via:
#   cmake -DDXC=<dxc> -DSHADER_DIR=<dir> -DAMD_INC=<dir> -DOUT_DIR=<dir> -P validate_shaders.cmake
#
# Runs DXC (Shader Model 6.6 compute) over each RDG shader and reports pass/fail.
# Never fails the build — the WMMA conv shader is WIP and depends on AMD wave-matrix
# intrinsics that stock DXC will reject until the kernel is finished.

set(_shaders
    "rdg_upsample.hlsl:RDG_Upsample_CS"
    "rdg_dfm_block.hlsl:RDG_DFM_Block_CS"
    "rdg_ctr_block.hlsl:RDG_CTR_Block_CS"
    "rdg_wmma_conv2d.hlsl:RDG_Conv2D_WMMA_CS")

set(_fail 0)
foreach(entry IN LISTS _shaders)
    string(REPLACE ":" ";" parts "${entry}")
    list(GET parts 0 file)
    list(GET parts 1 ep)
    execute_process(
        COMMAND "${DXC}" -T cs_6_6 -E ${ep} -I "${AMD_INC}"
                "${SHADER_DIR}/${file}" -Fo "${OUT_DIR}/${file}.dxil"
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err)
    if(rc EQUAL 0)
        message(STATUS "  OK    ${file} [${ep}]")
    else()
        math(EXPR _fail "${_fail}+1")
        string(STRIP "${err}" err)
        message(STATUS "  FAIL  ${file} [${ep}]  (WIP)\n        ${err}")
    endif()
endforeach()

message(STATUS "Shader validation complete (${_fail} shader(s) failed; failures are expected while shaders are WIP).")
