// rdnu_upscale_map.cpp - see rdnu_upscale_map.h
#include "rdnu_upscale_map.h"

#include <FidelityFX/host/ffx_nss.h>
#include <ffx_upscale.h>

namespace rdnu
{

uint32_t NssFlagsFromUpscale(uint32_t f, std::vector<std::string>& warnings)
{
    uint32_t n = FFX_NSS_CONTEXT_FLAG_QUANTIZED | FFX_NSS_CONTEXT_FLAG_ALLOW_16BIT | FFX_NSS_CONTEXT_FLAG_MANAGE_HISTORY;
    if (f & FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE)
        n |= FFX_NSS_CONTEXT_FLAG_HIGH_DYNAMIC_RANGE;
    if (f & FFX_UPSCALE_ENABLE_DEPTH_INVERTED)
        n |= FFX_NSS_CONTEXT_FLAG_DEPTH_INVERTED;
    if (f & FFX_UPSCALE_ENABLE_DEPTH_INFINITE)
        n |= FFX_NSS_CONTEXT_FLAG_DEPTH_INFINITE;
    if (f & FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE)
        warnings.push_back("non-linear colour input is treated as linear");
    if (f & FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION)
        warnings.push_back("each new render size restarts the history");
    return n;
}

float UpscaleRatio(uint32_t mode)
{
    switch (mode)
    {
    case FFX_UPSCALE_QUALITY_MODE_NATIVEAA: return 1.0f;
    case FFX_UPSCALE_QUALITY_MODE_QUALITY: return 1.5f;
    case FFX_UPSCALE_QUALITY_MODE_BALANCED: return 1.7f;
    case FFX_UPSCALE_QUALITY_MODE_PERFORMANCE: return 2.0f;
    case FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE: return 3.0f;
    default: return 0.0f;
    }
}

bool RenderResolution(uint32_t displayWidth, uint32_t displayHeight, uint32_t mode, uint32_t& width, uint32_t& height)
{
    const float r = UpscaleRatio(mode);
    if (r == 0.0f)
        return false;
    width  = uint32_t(float(displayWidth) / r);
    height = uint32_t(float(displayHeight) / r);
    return true;
}

uint32_t NssStateFromApi(uint32_t s)
{
    switch (s)
    {
    case FFX_API_RESOURCE_STATE_UNORDERED_ACCESS: return FFX_RESOURCE_STATE_GENERIC_UAV;
    case FFX_API_RESOURCE_STATE_COMPUTE_READ: return FFX_RESOURCE_STATE_COMPUTE_READ;
    case FFX_API_RESOURCE_STATE_PIXEL_READ: return FFX_RESOURCE_STATE_PIXEL_READ;
    case FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ: return FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ;
    case FFX_API_RESOURCE_STATE_COPY_SRC: return FFX_RESOURCE_STATE_COPY_SRC;
    case FFX_API_RESOURCE_STATE_COPY_DEST: return FFX_RESOURCE_STATE_COPY_DEST;
    case FFX_API_RESOURCE_STATE_GENERIC_READ: return FFX_RESOURCE_STATE_GENERIC_READ;
    case FFX_API_RESOURCE_STATE_INDIRECT_ARGUMENT: return FFX_RESOURCE_STATE_INDIRECT_ARGUMENT;
    case FFX_API_RESOURCE_STATE_RENDER_TARGET: return FFX_RESOURCE_STATE_RENDER_TARGET;
    case FFX_API_RESOURCE_STATE_DEPTH_ATTACHMENT: return FFX_RESOURCE_STATE_DEPTH_ATTACHEMENT;
    default: return FFX_RESOURCE_STATE_COMMON;
    }
}

}  // namespace rdnu
