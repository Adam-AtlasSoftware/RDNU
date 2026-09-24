// rdnu_engine_dx12.h - records the NSS INT8 network into a D3D12 command list.
//
// Owns the weights, the activation arena, one root signature and a PSO per kernel of the plan
// (DXIL from rdnu_shaderc, no runtime compilation). The caller owns the network input and KPN
// buffers and the temporal RGBA8_SNORM texture, which the NSS passes share with the network.
#pragma once

#include "rdnu_engine_core.h"

#include <d3d12.h>

#include <string>
#include <vector>

namespace rdnu
{

struct EngineDx12Desc
{
    ID3D12Device* device        = nullptr;
    const void*   manifest      = nullptr;  // nss.rdnm
    size_t        manifestBytes = 0;
    const void*   weights       = nullptr;  // nss_w8.bin
    size_t        weightBytes   = 0;
    uint32_t      maxWidth      = 0;        // network size bound, multiples of 8
    uint32_t      maxHeight     = 0;
    uint32_t      inputPitch    = 0;        // pixels per row of the input buffer (0: maxWidth)
    uint32_t      kpnPitch      = 0;        // pixels per row of the KPN buffer (0: maxWidth / 4)
    bool          useWmma       = false;    // RDNA3 with AMD wave-matrix intrinsics enabled
};

class EngineDx12
{
public:
    EngineDx12() = default;
    EngineDx12(const EngineDx12&) = delete;
    EngineDx12& operator=(const EngineDx12&) = delete;
    ~EngineDx12() { Destroy(); }

    bool Create(const EngineDx12Desc& desc, std::string& error);
    void Destroy();

    // input, kpn: raw buffers in UNORDERED_ACCESS. temporal: UAV of the temporal texture in the
    // shader-visible heap currently bound on cl. width, height: this frame's network size.
    // Leaves the root signature and pipeline of cl changed; ends with a UAV barrier.
    bool Record(ID3D12GraphicsCommandList* cl, D3D12_GPU_VIRTUAL_ADDRESS input, D3D12_GPU_VIRTUAL_ADDRESS kpn,
                D3D12_GPU_DESCRIPTOR_HANDLE temporal, uint32_t width, uint32_t height, std::string& error);

    bool        UsesWmma() const { return wmma_; }
    const Plan& GetPlan() const { return plan_; }
    uint64_t    MemoryBytes() const { return plan_.ArenaBytes() + weightBytes_; }
    ID3D12Resource* Arena() const { return arena_; }  // UNORDERED_ACCESS; for validation readback

private:
    ID3D12Resource* Buffer(uint64_t bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state, std::string& error);
    void            Prepare(ID3D12GraphicsCommandList* cl);

    ID3D12Device*        device_    = nullptr;
    ID3D12RootSignature* rootSig_   = nullptr;
    ID3D12Resource*      weights_   = nullptr;
    ID3D12Resource*      upload_    = nullptr;
    ID3D12Resource*      arena_     = nullptr;
    ID3D12Resource*      mailbox_   = nullptr;  // AMD intrinsics UAV, never really written
    ID3D12PipelineState* fill_      = nullptr;
    Manifest             manifest_;
    Plan                 plan_;
    std::vector<std::pair<KernelKey, ID3D12PipelineState*>> psos_;
    uint64_t             weightBytes_ = 0;
    bool                 wmma_        = false;
    bool                 prepared_    = false;
};

}  // namespace rdnu
