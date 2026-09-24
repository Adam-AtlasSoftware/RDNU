// test_upscale_map.cpp - FSR 3.1 upscale API to NSS mapping.
#include "../../src/provider/rdnu_upscale_map.h"

#include <FidelityFX/host/ffx_nss.h>
#include <ffx_upscale.h>

#include <cstdio>

namespace
{
int g_failures = 0;

#define CHECK(cond)                                                     \
    do                                                                  \
    {                                                                   \
        if (!(cond))                                                    \
        {                                                               \
            ++g_failures;                                               \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                               \
    } while (0)
}  // namespace

int main()
{
    using namespace rdnu;
    std::vector<std::string> w;
    const uint32_t base = FFX_NSS_CONTEXT_FLAG_QUANTIZED | FFX_NSS_CONTEXT_FLAG_ALLOW_16BIT | FFX_NSS_CONTEXT_FLAG_MANAGE_HISTORY;
    CHECK(NssFlagsFromUpscale(0, w) == base && w.empty());
    CHECK(NssFlagsFromUpscale(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED | FFX_UPSCALE_ENABLE_DEPTH_INFINITE, w) ==
          (base | FFX_NSS_CONTEXT_FLAG_HIGH_DYNAMIC_RANGE | FFX_NSS_CONTEXT_FLAG_DEPTH_INVERTED | FFX_NSS_CONTEXT_FLAG_DEPTH_INFINITE));
    CHECK(w.empty());
    // exposure and motion vectors are the provider's prepare passes, not context flags
    CHECK(NssFlagsFromUpscale(FFX_UPSCALE_ENABLE_AUTO_EXPOSURE | FFX_UPSCALE_ENABLE_DEBUG_CHECKING |
                                  FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION | FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS,
                              w) == base &&
          w.empty());
    NssFlagsFromUpscale(FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE | FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION, w);
    CHECK(w.size() == 2);

    CHECK(UpscaleRatio(FFX_UPSCALE_QUALITY_MODE_NATIVEAA) == 1.0f);
    CHECK(UpscaleRatio(FFX_UPSCALE_QUALITY_MODE_QUALITY) == 1.5f);
    CHECK(UpscaleRatio(FFX_UPSCALE_QUALITY_MODE_BALANCED) == 1.7f);
    CHECK(UpscaleRatio(FFX_UPSCALE_QUALITY_MODE_PERFORMANCE) == 2.0f);
    CHECK(UpscaleRatio(FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE) == 3.0f);
    CHECK(UpscaleRatio(99) == 0.0f);

    uint32_t rw = 0, rh = 0;
    CHECK(RenderResolution(3840, 2160, FFX_UPSCALE_QUALITY_MODE_PERFORMANCE, rw, rh) && rw == 1920 && rh == 1080);
    CHECK(RenderResolution(2560, 1440, FFX_UPSCALE_QUALITY_MODE_QUALITY, rw, rh) && rw == 1706 && rh == 960);
    CHECK(RenderResolution(1920, 1080, FFX_UPSCALE_QUALITY_MODE_BALANCED, rw, rh) && rw == 1129 && rh == 635);
    CHECK(RenderResolution(3840, 2160, FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE, rw, rh) && rw == 1280 && rh == 720);
    CHECK(!RenderResolution(1920, 1080, 7, rw, rh));

    CHECK(NssStateFromApi(FFX_API_RESOURCE_STATE_UNORDERED_ACCESS) == FFX_RESOURCE_STATE_GENERIC_UAV);
    CHECK(NssStateFromApi(FFX_API_RESOURCE_STATE_COMPUTE_READ) == FFX_RESOURCE_STATE_COMPUTE_READ);
    CHECK(NssStateFromApi(FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ) == FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ);
    CHECK(NssStateFromApi(FFX_API_RESOURCE_STATE_GENERIC_READ) == FFX_RESOURCE_STATE_GENERIC_READ);
    CHECK(NssStateFromApi(FFX_API_RESOURCE_STATE_COPY_DEST) == FFX_RESOURCE_STATE_COPY_DEST);
    CHECK(NssStateFromApi(FFX_API_RESOURCE_STATE_RENDER_TARGET) == FFX_RESOURCE_STATE_RENDER_TARGET);
    CHECK(NssStateFromApi(FFX_API_RESOURCE_STATE_PRESENT) == FFX_RESOURCE_STATE_COMMON);

    std::printf("%s (%d failures)\n", g_failures ? "FAIL" : "PASS", g_failures);
    return g_failures ? 2 : 0;
}
