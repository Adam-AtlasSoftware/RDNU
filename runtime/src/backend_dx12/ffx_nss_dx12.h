// ffx_nss_dx12.h - D3D12 backend for Arm's FidelityFX interface (sdk/include/FidelityFX/host),
// scoped to what the NSS component uses: compute passes from the embedded DXIL, the network
// as the data graph through rdnu::EngineDx12, clears, and external resource registration.
#pragma once

#include <FidelityFX/host/ffx_interface.h>

#include <d3d12.h>

size_t       ffxGetScratchMemorySizeDX12(size_t maxContexts);
FfxErrorCode ffxGetInterfaceDX12(FfxInterface* backendInterface, ID3D12Device* device, void* scratchBuffer, size_t scratchBufferSize,
                                 size_t maxContexts);
// Destroys what ffxGetInterfaceDX12 built in the scratch buffer; call after the last context.
void         ffxReleaseInterfaceDX12(FfxInterface* backendInterface);

FfxDevice      ffxGetDeviceDX12(ID3D12Device* device);
FfxCommandList ffxGetCommandListDX12(ID3D12GraphicsCommandList* commandList);
// Describes a game resource from its D3D12 description; state is its state at dispatch.
FfxResource    ffxGetResourceDX12(ID3D12Resource* resource, FfxResourceStates state, const char* name);

// Runs the network with the plain DP4a kernels even on RDNA3 (A/B testing).
void ffxNssDx12ForceDp4a(bool force);
// Whether the network runs on the AMD wave-matrix kernels; known once a context exists.
bool ffxNssDx12UsesWmma(FfxInterface* backendInterface);

// RDNU extensions for FSR 3.1 hosts, recorded around ffxNssContextDispatch on the same list.
struct FfxNssDx12Exposure
{
    ID3D12Resource*       texture      = nullptr;  // the game's 1x1 exposure texture, or null
    D3D12_RESOURCE_STATES textureState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    ID3D12Resource*       colour       = nullptr;  // auto exposure from this when there is no texture
    D3D12_RESOURCE_STATES colourState  = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    uint32_t              width        = 0;        // render size, for auto exposure
    uint32_t              height       = 0;
    float                 preExposure  = 1.0f;
};

// Computes this frame's exposure on the GPU (texture / preExposure, auto exposure, or
// 1 / preExposure) and makes the next dispatch's NSS passes use it instead of the constant.
FfxErrorCode ffxNssDx12PrepareExposure(FfxInterface* backendInterface, ID3D12GraphicsCommandList* commandList, const FfxNssDx12Exposure& exposure);

struct FfxNssDx12Motion
{
    ID3D12Resource*       source       = nullptr;  // the game's motion vectors
    D3D12_RESOURCE_STATES sourceState  = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    ID3D12Resource*       target       = nullptr;  // R32G32_FLOAT UAV texture, render size, kept in NON_PIXEL_SHADER_RESOURCE
    uint32_t              renderWidth  = 0;
    uint32_t              renderHeight = 0;
    uint32_t              sourceWidth  = 0;        // display resolution vectors: the upscaled size; else 0
    uint32_t              sourceHeight = 0;
    float                 scale[2]     = {1.0f, 1.0f};
    float                 jitter[2]    = {};
    float                 cancel[2]    = {};       // previous minus current jitter, when the vectors carry it
};

// Writes NSS-ready motion (render pixels, scale 1) for FSR 3.1 display resolution or jittered
// vectors. Hosts without either pass their vectors to NSS directly.
FfxErrorCode ffxNssDx12PrepareMotion(FfxInterface* backendInterface, ID3D12GraphicsCommandList* commandList, const FfxNssDx12Motion& motion);

// RCAS from input to output (both width x height) with the prepared exposure. sharpness 0..1 as in FSR.
FfxErrorCode ffxNssDx12Sharpen(FfxInterface* backendInterface, ID3D12GraphicsCommandList* commandList, ID3D12Resource* input,
                               D3D12_RESOURCE_STATES inputState, ID3D12Resource* output, D3D12_RESOURCE_STATES outputState, uint32_t width,
                               uint32_t height, float sharpness);
