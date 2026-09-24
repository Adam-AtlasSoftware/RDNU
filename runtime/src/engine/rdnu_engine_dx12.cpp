// rdnu_engine_dx12.cpp - see rdnu_engine_dx12.h
#include "rdnu_engine_dx12.h"
#include "rdnu_embedded.h"

#include <cstring>

namespace rdnu
{
namespace
{
// Root signature, shared by every network kernel and the arena fill:
//   b0     NetConstants (root constants)
//   t0     weights           u0  output view         u1  input view
//   u2     temporal texture (descriptor table)
//   u0 in space 2147420894: AMD shader intrinsics mailbox (wave-matrix kernels)
enum RootParam : UINT { kConstants, kWeights, kOutput, kInput, kTemporal, kAmdExt, kRootParamCount };
constexpr UINT kAmdExtSpace = 2147420894u;

constexpr uint32_t kFillWord = 0x80808080u;

template <typename T>
void Release(T*& p)
{
    if (p)
        p->Release();
    p = nullptr;
}

D3D12_RESOURCE_BARRIER UavBarrier()
{
    D3D12_RESOURCE_BARRIER b{};
    b.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = nullptr;
    return b;
}

D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b{};
    b.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource   = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter  = after;
    return b;
}
}  // namespace

ID3D12Resource* EngineDx12::Buffer(uint64_t bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state, std::string& error)
{
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = heap;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = bytes;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = heap == D3D12_HEAP_TYPE_DEFAULT ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    ID3D12Resource* r   = nullptr;
    if (FAILED(device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&r))))
        error = "cannot allocate a " + std::to_string(bytes) + " byte buffer";
    return r;
}

bool EngineDx12::Create(const EngineDx12Desc& desc, std::string& error)
{
    Destroy();
    device_ = desc.device;
    wmma_   = desc.useWmma;
    if (!ParseManifest(desc.manifest, desc.manifestBytes, manifest_, error))
        return false;
    if (desc.weightBytes != manifest_.blobBytes)
        return error = "weight blob does not match the manifest", false;

    PlanConfig cfg;
    cfg.maxWidth   = desc.maxWidth;
    cfg.maxHeight  = desc.maxHeight;
    cfg.inputPitch = desc.inputPitch;
    cfg.kpnPitch   = desc.kpnPitch;
    cfg.packed     = desc.packed;
    cfg.useWmma    = wmma_;
    if (!plan_.Build(manifest_, cfg, error))
        return false;

    D3D12_ROOT_PARAMETER p[kRootParamCount] = {};
    for (D3D12_ROOT_PARAMETER& q : p)
        q.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    p[kConstants].ParameterType            = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p[kConstants].Constants.Num32BitValues = kNetConstantDwords;
    p[kWeights].ParameterType              = D3D12_ROOT_PARAMETER_TYPE_SRV;
    p[kOutput].ParameterType               = D3D12_ROOT_PARAMETER_TYPE_UAV;
    p[kInput].ParameterType                = D3D12_ROOT_PARAMETER_TYPE_UAV;
    p[kInput].Descriptor.ShaderRegister    = 1;
    D3D12_DESCRIPTOR_RANGE temporal{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 2, 0, 0};
    p[kTemporal].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    p[kTemporal].DescriptorTable.NumDescriptorRanges = 1;
    p[kTemporal].DescriptorTable.pDescriptorRanges   = &temporal;
    p[kAmdExt].ParameterType                         = D3D12_ROOT_PARAMETER_TYPE_UAV;
    p[kAmdExt].Descriptor.RegisterSpace              = kAmdExtSpace;
    D3D12_ROOT_SIGNATURE_DESC rs{};
    rs.NumParameters = kRootParamCount;
    rs.pParameters   = p;
    ID3DBlob* blob = nullptr;
    ID3DBlob* msg  = nullptr;
    HRESULT   hr   = D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &msg);
    if (SUCCEEDED(hr))
        hr = device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rootSig_));
    Release(blob);
    Release(msg);
    if (FAILED(hr))
        return error = "cannot create the network root signature", false;

    auto pso = [&](const char* name, ID3D12PipelineState*& out) {
        const ShaderBlob* s = FindShader(name);
        if (!s)
            return error = std::string("shader not compiled: ") + name, false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = rootSig_;
        pd.CS             = {s->data, s->size};
        if (FAILED(device_->CreateComputePipelineState(&pd, IID_PPV_ARGS(&out))))
            return error = std::string("cannot create the pipeline for ") + name, false;
        return true;
    };
    if (!pso("nss_fill", fill_))
        return false;
    for (const KernelKey& k : plan_.RequiredKernels(wmma_))
    {
        ID3D12PipelineState* s = nullptr;
        if (!pso(k.Name().c_str(), s))
            return false;
        psos_.push_back({k, s});
    }

    weightBytes_ = desc.weightBytes;
    upload_      = Buffer(weightBytes_, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, error);
    weights_     = Buffer(weightBytes_, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, error);
    arena_       = Buffer(plan_.ArenaBytes(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, error);
    mailbox_     = Buffer(256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, error);
    if (!upload_ || !weights_ || !arena_ || !mailbox_)
        return false;
    void*       mapped = nullptr;
    D3D12_RANGE none{0, 0};
    if (FAILED(upload_->Map(0, &none, &mapped)))
        return error = "cannot map the weight upload buffer", false;
    std::memcpy(mapped, desc.weights, weightBytes_);
    upload_->Unmap(0, nullptr);
    return true;
}

void EngineDx12::Destroy()
{
    for (auto& p : psos_)
        Release(p.second);
    psos_.clear();
    Release(fill_);
    Release(rootSig_);
    Release(arena_);
    Release(mailbox_);
    Release(weights_);
    Release(upload_);
    prepared_ = false;
}

// First use: weights into device memory, arena to all -128 codes.
void EngineDx12::Prepare(ID3D12GraphicsCommandList* cl)
{
    cl->CopyBufferRegion(weights_, 0, upload_, 0, weightBytes_);
    D3D12_RESOURCE_BARRIER b = Transition(weights_, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cl->ResourceBarrier(1, &b);

    const uint32_t quads   = uint32_t(plan_.ArenaBytes() / 16);
    const uint32_t groups  = DivUp(quads, 256);
    const uint32_t groupsX = groups < 65535 ? groups : 65535;
    const uint32_t fill[3] = {quads, kFillWord, groupsX};
    cl->SetComputeRootSignature(rootSig_);
    cl->SetPipelineState(fill_);
    cl->SetComputeRoot32BitConstants(kConstants, 3, fill, 0);
    cl->SetComputeRootUnorderedAccessView(kOutput, arena_->GetGPUVirtualAddress());
    cl->Dispatch(groupsX, DivUp(groups, groupsX), 1);
    b = UavBarrier();
    cl->ResourceBarrier(1, &b);
    prepared_ = true;
}

bool EngineDx12::Record(ID3D12GraphicsCommandList* cl, D3D12_GPU_VIRTUAL_ADDRESS input, D3D12_GPU_VIRTUAL_ADDRESS kpn,
                        D3D12_GPU_DESCRIPTOR_HANDLE temporal, uint32_t width, uint32_t height, std::string& error)
{
    if (!rootSig_)
        return error = "engine not created", false;
    if ((width != plan_.Width() || height != plan_.Height()) && !plan_.Update(width, height, error))
        return false;
    if (!prepared_)
        Prepare(cl);

    auto address = [&](Resource r) {
        switch (r)
        {
        case Resource::Input: return input;
        case Resource::Kpn: return kpn;
        default: return arena_->GetGPUVirtualAddress();
        }
    };

    cl->SetComputeRootSignature(rootSig_);
    cl->SetComputeRootShaderResourceView(kWeights, weights_->GetGPUVirtualAddress());
    cl->SetComputeRootDescriptorTable(kTemporal, temporal);
    cl->SetComputeRootUnorderedAccessView(kAmdExt, mailbox_->GetGPUVirtualAddress());
    ID3D12PipelineState* bound = nullptr;
    const D3D12_RESOURCE_BARRIER uav = UavBarrier();
    for (const Dispatch& d : plan_.Dispatches())
    {
        ID3D12PipelineState* pso = nullptr;
        for (auto& p : psos_)
            if (p.first == d.key)
                pso = p.second;
        if (!pso)
            return error = "no pipeline for " + d.key.Name(), false;
        if (d.barrierBefore)
            cl->ResourceBarrier(1, &uav);
        if (pso != bound)
            cl->SetPipelineState(bound = pso);
        cl->SetComputeRoot32BitConstants(kConstants, kNetConstantDwords, &d.constants, 0);
        cl->SetComputeRootUnorderedAccessView(kOutput, address(d.out));
        cl->SetComputeRootUnorderedAccessView(kInput, address(d.in));
        cl->Dispatch(d.groups[0], d.groups[1], d.groups[2]);
    }
    cl->ResourceBarrier(1, &uav);
    return true;
}

}  // namespace rdnu
