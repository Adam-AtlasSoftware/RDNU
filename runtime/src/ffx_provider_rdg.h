#pragma once

// Resolved via the FidelityFX SDK include root (the sample puts .../Kits on the include path).
#include <FidelityFX/upscalers/include/ffx_upscale.h>

#ifdef __cplusplus
extern "C" {
#endif

// RDNU Context Structure
struct RDGContext {
    void* device;                     // ID3D12Device*
    void* commandList;                // ID3D12GraphicsCommandList*
    struct FfxApiDimensions2D renderSize;
    struct FfxApiDimensions2D upscaleSize;

    // Persistent Hidden States (Ping-Pong buffers for temporal continuity)
    // 3 pairs for the RDG pyramid architecture (FP16 formats)

    // Level 0: Deepest Level (1/4th Render Res, 48 channels)
    void* hiddenState0_A;        // ID3D12Resource*
    void* hiddenState0_B;        // ID3D12Resource*

    // Level 1: Middle Level (1/2 Render Res, 32 channels)
    void* hiddenState1_A;        // ID3D12Resource*
    void* hiddenState1_B;        // ID3D12Resource*

    // Level 2: Surface Level (Full Render Res, 16 channels)
    void* hiddenState2_A;        // ID3D12Resource*
    void* hiddenState2_B;        // ID3D12Resource*

    bool readFromA;

    void* weightsBuffer;              // ID3D12Resource*
    void* descriptorHeap;             // ID3D12DescriptorHeap*

    // Model specific pipeline states
    void* rootSignature;              // ID3D12RootSignature*
    void* psoConv2D_FP16;             // ID3D12PipelineState*
    void* psoPixelShuffle;            // ID3D12PipelineState*
    void* psoCTR_FP16;                // ID3D12PipelineState* Cross-Temporal block
    void* psoUpsample;                // ID3D12PipelineState* Temporary Upsample

    void* pTemporalHistoryUAV;        // ID3D12Resource*
    void* pSkipConnectionsUAV;        // ID3D12Resource*

    uint32_t descriptorHeapIndex;
};

// API Functions mirroring FSR4/DLSS to be used as a drop-in replacement
ffxReturnCode_t ffxCreateContextDescRDG(ffxCreateContextDescUpscale* desc, void* device, RDGContext** outContext);
ffxReturnCode_t ffxDispatchDescRDG(RDGContext* context, const ffxDispatchDescUpscale* desc);
ffxReturnCode_t ffxDestroyContextRDG(RDGContext* context);

#ifdef __cplusplus
}
#endif
