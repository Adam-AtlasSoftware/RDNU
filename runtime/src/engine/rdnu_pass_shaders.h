// rdnu_pass_shaders.h - the NSS pass shader permutations built into the runtime. Shared by
// rdnu_shaderc (which compiles them) and the DX12 backend (which looks them up by name).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rdnu
{

enum PassBits : uint32_t
{
    kPassReverseZ      = 1u << 0,
    kPassScaleX2       = 1u << 1,
    kPassManageHistory = 1u << 2,
};

struct PassShader
{
    const char* name;   // ffx_nss_<name>.hlsl under shaders/nss
    uint32_t    bits;   // permutation bits the pass depends on
};

inline const std::vector<PassShader>& PassShaders()
{
    static const std::vector<PassShader> passes = {
        {"depth_scatter", kPassReverseZ},
        {"disocclusion_mask_lq", kPassReverseZ},
        {"pre_process", kPassReverseZ},
        {"generate_offset_lut", 0},
        {"post_process", kPassScaleX2 | kPassManageHistory},
        {"debug_view", kPassReverseZ},
    };
    return passes;
}

// Quality modes with a shipped model (0 = high).
inline const std::vector<uint32_t>& PassQualities()
{
    static const std::vector<uint32_t> q = {0};
    return q;
}

// e.g. nss_post_process_q0_sp1_mh0; bits outside the pass's own are ignored.
inline std::string PassShaderName(const PassShader& p, uint32_t quality, uint32_t bits)
{
    bits &= p.bits;
    std::string n = std::string("nss_") + p.name + "_q" + std::to_string(quality);
    if (p.bits & kPassReverseZ)
        n += bits & kPassReverseZ ? "_rz1" : "_rz0";
    if (p.bits & kPassScaleX2)
        n += bits & kPassScaleX2 ? "_sp1" : "_sp0";
    if (p.bits & kPassManageHistory)
        n += bits & kPassManageHistory ? "_mh1" : "_mh0";
    return n;
}

inline std::vector<std::string> PassShaderDefines(const PassShader& p, uint32_t quality, uint32_t bits)
{
    bits &= p.bits;
    return {"NSS_SHADER_QUALITY_MODE=" + std::to_string(quality), std::string("REVERSE_Z=") + (bits & kPassReverseZ ? "1" : "0"),
            std::string("SCALE_PRESET_MODE=") + (bits & kPassScaleX2 ? "1" : "0"),
            std::string("MANAGE_HISTORY=") + (bits & kPassManageHistory ? "1" : "0")};
}

}  // namespace rdnu
