// rdnu_upscale_map.h - FSR 3.1 upscale API to NSS, as pure functions.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rdnu
{

// NSS context flags (FfxNssInitializationFlagBits) for FSR 3.1 create flags
// (FfxApiCreateContextUpscaleFlags). Features NSS does not have are reported in warnings;
// exposure and motion vector flags are the provider's (prepare passes), not context flags.
uint32_t NssFlagsFromUpscale(uint32_t upscaleFlags, std::vector<std::string>& warnings);

// FSR's per-dimension ratio for FfxApiUpscaleQualityMode; 0 for an unknown mode.
float UpscaleRatio(uint32_t qualityMode);

// FSR's render resolution for a display size and quality mode; false for an unknown mode.
bool RenderResolution(uint32_t displayWidth, uint32_t displayHeight, uint32_t qualityMode, uint32_t& width, uint32_t& height);

// FfxApiResourceState to Arm's FfxResourceStates (the two enums number states differently).
uint32_t NssStateFromApi(uint32_t apiState);

}  // namespace rdnu
