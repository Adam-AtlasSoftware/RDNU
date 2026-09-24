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
