// ffx_nss_dx12.cpp - see ffx_nss_dx12.h
//
// Resources live in one table indexed by FfxResourceInternal::internalIndex; index 0 is the null
// resource, which the component leaves in every slot it never fills. Registered (game)
// resources are valid for one dispatch and return to their registered state on unregister.
// Every job gets fresh descriptors from a shader-visible ring and its constants from an upload
// ring, both sized for many frames in flight. Compute passes use one root signature each:
// root CBV, an SRV table and a UAV table indexed by register, and static samplers.
#include "ffx_nss_dx12.h"

#include "../engine/rdnu_amd_ext.h"
#include "../engine/rdnu_embedded.h"
#include "../engine/rdnu_engine_dx12.h"
#include "../engine/rdnu_pass_shaders.h"

#include <FidelityFX/host/ffx_nss.h>
#include <ffx_nss_private.h>

#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace
{
constexpr uint32_t kGpuDescriptors = 16384;
constexpr uint32_t kCpuDescriptors = 256;
constexpr uint64_t kUploadBytes    = 4u << 20;
constexpr uint32_t kStagingBytes   = 256u << 10;
constexpr D3D12_RESOURCE_STATES kSrvState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
constexpr D3D12_RESOURCE_STATES kUavState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

bool g_forceDp4a = false;

struct Resource
{
    ID3D12Resource*        res      = nullptr;
    FfxResourceDescription desc     = {};
    D3D12_RESOURCE_STATES  state    = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES  restore  = D3D12_RESOURCE_STATE_COMMON;  // external: state at registration
    uint32_t               context  = 0;
    bool                   used     = false;
    bool                   external = false;
    std::string            name;
};

struct Pipeline
{
    ID3D12RootSignature* rootSig    = nullptr;
    ID3D12PipelineState* pso        = nullptr;
    UINT                 srvParam   = UINT(-1), uavParam = UINT(-1);
    uint32_t             srvSlots   = 0, uavSlots = 0;
    rdnu::EngineDx12*    engine     = nullptr;  // data graph
    uint32_t             width      = 0, height = 0;
};

struct Backend
{
    ID3D12Device*         device   = nullptr;
    FfxBackendMessage     message  = nullptr;
    uint32_t              refCount = 0;
    std::vector<bool>     contexts;
    std::vector<Resource> resources;
    std::vector<FfxGpuJobDescription> jobs;
    std::vector<D3D12_RESOURCE_BARRIER> barriers;

    ID3D12DescriptorHeap* gpuHeap = nullptr;
    ID3D12DescriptorHeap* cpuHeap = nullptr;
    UINT                  descSize = 0;
    uint32_t              gpuHead = 0, cpuHead = 0;
    ID3D12Resource*       upload = nullptr;
    uint8_t*              uploadPtr = nullptr;
    uint64_t              uploadHead = 0;
    std::vector<uint8_t>  staging;
    uint32_t              stagingHead = 0;
    bool                  wmma = false;
};

Backend* Get(FfxInterface* i) { return static_cast<Backend*>(i->scratchBuffer); }

template <typename T>
void Release(T*& p)
{
    if (p)
        p->Release();
    p = nullptr;
}

void Print(Backend* b, const std::string& s)
{
    if (b->message)
        b->message(FFX_MESSAGE_TYPE_ERROR, s.c_str());
}

// ---------------------------------------------------------------------------------------- formats

DXGI_FORMAT Dxgi(FfxSurfaceFormat f)
{
    switch (f)
    {
    case FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_TYPELESS;
    case FFX_SURFACE_FORMAT_R32G32B32A32_UINT: return DXGI_FORMAT_R32G32B32A32_UINT;
    case FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case FFX_SURFACE_FORMAT_R32G32B32_FLOAT: return DXGI_FORMAT_R32G32B32_FLOAT;
    case FFX_SURFACE_FORMAT_R32G32_FLOAT: return DXGI_FORMAT_R32G32_FLOAT;
    case FFX_SURFACE_FORMAT_R8_UINT: return DXGI_FORMAT_R8_UINT;
    case FFX_SURFACE_FORMAT_R8_SINT: return DXGI_FORMAT_R8_SINT;
    case FFX_SURFACE_FORMAT_R32_UINT: return DXGI_FORMAT_R32_UINT;
    case FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_TYPELESS;
    case FFX_SURFACE_FORMAT_R8G8B8A8_UNORM: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case FFX_SURFACE_FORMAT_R8G8B8A8_SNORM: return DXGI_FORMAT_R8G8B8A8_SNORM;
    case FFX_SURFACE_FORMAT_R8G8B8A8_SRGB: return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case FFX_SURFACE_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_TYPELESS;
    case FFX_SURFACE_FORMAT_B8G8R8A8_UNORM: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case FFX_SURFACE_FORMAT_B8G8R8A8_SRGB: return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    case FFX_SURFACE_FORMAT_R11G11B10_FLOAT: return DXGI_FORMAT_R11G11B10_FLOAT;
    case FFX_SURFACE_FORMAT_R10G10B10A2_UNORM: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case FFX_SURFACE_FORMAT_R16G16_FLOAT: return DXGI_FORMAT_R16G16_FLOAT;
    case FFX_SURFACE_FORMAT_R16G16_UINT: return DXGI_FORMAT_R16G16_UINT;
    case FFX_SURFACE_FORMAT_R16G16_SINT: return DXGI_FORMAT_R16G16_SINT;
    case FFX_SURFACE_FORMAT_R16_FLOAT: return DXGI_FORMAT_R16_FLOAT;
    case FFX_SURFACE_FORMAT_R16_UINT: return DXGI_FORMAT_R16_UINT;
    case FFX_SURFACE_FORMAT_R16_UNORM: return DXGI_FORMAT_R16_UNORM;
    case FFX_SURFACE_FORMAT_R16_SNORM: return DXGI_FORMAT_R16_SNORM;
    case FFX_SURFACE_FORMAT_R8_UNORM: return DXGI_FORMAT_R8_UNORM;
    case FFX_SURFACE_FORMAT_R8_SNORM: return DXGI_FORMAT_R8_SNORM;
    case FFX_SURFACE_FORMAT_R8G8_UNORM: return DXGI_FORMAT_R8G8_UNORM;
    case FFX_SURFACE_FORMAT_R8G8_UINT: return DXGI_FORMAT_R8G8_UINT;
    case FFX_SURFACE_FORMAT_R8G8_SINT: return DXGI_FORMAT_R8G8_SINT;
    case FFX_SURFACE_FORMAT_R32_FLOAT: return DXGI_FORMAT_R32_FLOAT;
    case FFX_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP: return DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    case FFX_SURFACE_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_TYPELESS;
    case FFX_SURFACE_FORMAT_R32G32_TYPELESS: return DXGI_FORMAT_R32G32_TYPELESS;
    case FFX_SURFACE_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_TYPELESS;
    case FFX_SURFACE_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_TYPELESS;
    case FFX_SURFACE_FORMAT_R16_TYPELESS: return DXGI_FORMAT_R16_TYPELESS;
    case FFX_SURFACE_FORMAT_R8_TYPELESS: return DXGI_FORMAT_R8_TYPELESS;
    case FFX_SURFACE_FORMAT_R8G8_TYPELESS: return DXGI_FORMAT_R8G8_TYPELESS;
    case FFX_SURFACE_FORMAT_R32_TYPELESS: return DXGI_FORMAT_R32_TYPELESS;
    default: return DXGI_FORMAT_UNKNOWN;
    }
}

FfxSurfaceFormat Ffx(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return FFX_SURFACE_FORMAT_R32G32B32A32_TYPELESS;
    case DXGI_FORMAT_R32G32B32A32_UINT: return FFX_SURFACE_FORMAT_R32G32B32A32_UINT;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return FFX_SURFACE_FORMAT_R16G16B16A16_TYPELESS;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R32G32B32_FLOAT: return FFX_SURFACE_FORMAT_R32G32B32_FLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS: return FFX_SURFACE_FORMAT_R32G32_TYPELESS;
    case DXGI_FORMAT_R32G32_FLOAT: return FFX_SURFACE_FORMAT_R32G32_FLOAT;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return FFX_SURFACE_FORMAT_R10G10B10A2_TYPELESS;
    case DXGI_FORMAT_R10G10B10A2_UNORM: return FFX_SURFACE_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R11G11B10_FLOAT: return FFX_SURFACE_FORMAT_R11G11B10_FLOAT;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return FFX_SURFACE_FORMAT_R8G8B8A8_TYPELESS;
    case DXGI_FORMAT_R8G8B8A8_UNORM: return FFX_SURFACE_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return FFX_SURFACE_FORMAT_R8G8B8A8_SRGB;
    case DXGI_FORMAT_R8G8B8A8_SNORM: return FFX_SURFACE_FORMAT_R8G8B8A8_SNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return FFX_SURFACE_FORMAT_B8G8R8A8_TYPELESS;
    case DXGI_FORMAT_B8G8R8A8_UNORM: return FFX_SURFACE_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return FFX_SURFACE_FORMAT_B8G8R8A8_SRGB;
    case DXGI_FORMAT_R16G16_TYPELESS: return FFX_SURFACE_FORMAT_R16G16_TYPELESS;
    case DXGI_FORMAT_R16G16_FLOAT: return FFX_SURFACE_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R16G16_UINT: return FFX_SURFACE_FORMAT_R16G16_UINT;
    case DXGI_FORMAT_R16G16_SINT: return FFX_SURFACE_FORMAT_R16G16_SINT;
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT: return FFX_SURFACE_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R32_UINT: return FFX_SURFACE_FORMAT_R32_UINT;
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: return FFX_SURFACE_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: return FFX_SURFACE_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R16_TYPELESS: return FFX_SURFACE_FORMAT_R16_TYPELESS;
    case DXGI_FORMAT_D16_UNORM:
    case DXGI_FORMAT_R16_UNORM: return FFX_SURFACE_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R16_FLOAT: return FFX_SURFACE_FORMAT_R16_FLOAT;
    case DXGI_FORMAT_R16_UINT: return FFX_SURFACE_FORMAT_R16_UINT;
    case DXGI_FORMAT_R16_SNORM: return FFX_SURFACE_FORMAT_R16_SNORM;
    case DXGI_FORMAT_R8_TYPELESS: return FFX_SURFACE_FORMAT_R8_TYPELESS;
    case DXGI_FORMAT_R8_UNORM: return FFX_SURFACE_FORMAT_R8_UNORM;
    case DXGI_FORMAT_R8_UINT: return FFX_SURFACE_FORMAT_R8_UINT;
    case DXGI_FORMAT_R8_SINT: return FFX_SURFACE_FORMAT_R8_SINT;
    case DXGI_FORMAT_R8_SNORM: return FFX_SURFACE_FORMAT_R8_SNORM;
    case DXGI_FORMAT_R8G8_TYPELESS: return FFX_SURFACE_FORMAT_R8G8_TYPELESS;
    case DXGI_FORMAT_R8G8_UNORM: return FFX_SURFACE_FORMAT_R8G8_UNORM;
    case DXGI_FORMAT_R8G8_UINT: return FFX_SURFACE_FORMAT_R8G8_UINT;
    case DXGI_FORMAT_R8G8_SINT: return FFX_SURFACE_FORMAT_R8G8_SINT;
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP: return FFX_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP;
    default: return FFX_SURFACE_FORMAT_UNKNOWN;
    }
}

// Typed format a shader can sample (typeless and depth resources resolve to their data).
DXGI_FORMAT SrvFormat(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT: return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_D16_UNORM: return DXGI_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R8_TYPELESS: return DXGI_FORMAT_R8_UNORM;
    case DXGI_FORMAT_R8G8_TYPELESS: return DXGI_FORMAT_R8G8_UNORM;
    case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS: return DXGI_FORMAT_R32G32_FLOAT;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    default: return f;
    }
}

// Typed format a shader can store to (no sRGB UAVs).
DXGI_FORMAT UavFormat(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8A8_UNORM;
    default: return SrvFormat(f);
    }
}

D3D12_RESOURCE_STATES D3dState(FfxResourceStates s)
{
    if (s & (FFX_RESOURCE_STATE_GENERIC_UAV | FFX_RESOURCE_STATE_DATA_GRAPH_READ | FFX_RESOURCE_STATE_DATA_GRAPH_WRITE))
        return kUavState;
    switch (s)
    {
    case FFX_RESOURCE_STATE_COMPUTE_READ: return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    case FFX_RESOURCE_STATE_PIXEL_READ: return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    case FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ:
        return D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    case FFX_RESOURCE_STATE_COPY_SRC: return D3D12_RESOURCE_STATE_COPY_SOURCE;
    case FFX_RESOURCE_STATE_COPY_DEST: return D3D12_RESOURCE_STATE_COPY_DEST;
    case FFX_RESOURCE_STATE_GENERIC_READ: return D3D12_RESOURCE_STATE_GENERIC_READ;
    case FFX_RESOURCE_STATE_INDIRECT_ARGUMENT: return D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    case FFX_RESOURCE_STATE_RENDER_TARGET: return D3D12_RESOURCE_STATE_RENDER_TARGET;
    case FFX_RESOURCE_STATE_DEPTH_ATTACHEMENT: return D3D12_RESOURCE_STATE_DEPTH_WRITE;
    default: return D3D12_RESOURCE_STATE_COMMON;
    }
}

FfxResourceStates FfxState(D3D12_RESOURCE_STATES s)
{
    const D3D12_RESOURCE_STATES anyRead = D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | kSrvState);
    if (s == kUavState)
        return FFX_RESOURCE_STATE_GENERIC_UAV;
    if (s == anyRead)
        return FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ;
    if (s == kSrvState)
        return FFX_RESOURCE_STATE_COMPUTE_READ;
    if (s == D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
        return FFX_RESOURCE_STATE_PIXEL_READ;
    if (s == D3D12_RESOURCE_STATE_COPY_SOURCE)
        return FFX_RESOURCE_STATE_COPY_SRC;
    if (s == D3D12_RESOURCE_STATE_COPY_DEST)
        return FFX_RESOURCE_STATE_COPY_DEST;
    if (s == D3D12_RESOURCE_STATE_GENERIC_READ)
        return FFX_RESOURCE_STATE_GENERIC_READ;
    if (s == D3D12_RESOURCE_STATE_RENDER_TARGET)
        return FFX_RESOURCE_STATE_RENDER_TARGET;
    return FFX_RESOURCE_STATE_COMMON;
}

bool IsBuffer(const Resource& r) { return r.res && r.res->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER; }

// ------------------------------------------------------------------------------- descriptors

D3D12_CPU_DESCRIPTOR_HANDLE Cpu(ID3D12DescriptorHeap* h, UINT size, uint32_t i)
{
    D3D12_CPU_DESCRIPTOR_HANDLE c = h->GetCPUDescriptorHandleForHeapStart();
    c.ptr += SIZE_T(i) * size;
    return c;
}

D3D12_GPU_DESCRIPTOR_HANDLE Gpu(ID3D12DescriptorHeap* h, UINT size, uint32_t i)
{
    D3D12_GPU_DESCRIPTOR_HANDLE g = h->GetGPUDescriptorHandleForHeapStart();
    g.ptr += UINT64(i) * size;
    return g;
}

uint32_t AllocGpu(Backend* b, uint32_t n)
{
    if (b->gpuHead + n > kGpuDescriptors)
        b->gpuHead = 0;
    uint32_t i = b->gpuHead;
    b->gpuHead += n;
    return i;
}

void SrvView(Backend* b, const Resource* r, bool buffer, D3D12_CPU_DESCRIPTOR_HANDLE h)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    ID3D12Resource* res       = r ? r->res : nullptr;
    if (buffer)
    {
        v.Format             = DXGI_FORMAT_R32_TYPELESS;
        v.ViewDimension      = D3D12_SRV_DIMENSION_BUFFER;
        v.Buffer.Flags       = D3D12_BUFFER_SRV_FLAG_RAW;
        v.Buffer.NumElements = res ? UINT(res->GetDesc().Width / 4) : 0;
    }
    else
    {
        D3D12_RESOURCE_DESC d = res ? res->GetDesc() : D3D12_RESOURCE_DESC{};
        v.Format              = res ? SrvFormat(d.Format) : DXGI_FORMAT_R8G8B8A8_UNORM;
        v.ViewDimension       = D3D12_SRV_DIMENSION_TEXTURE2D;
        v.Texture2D.MipLevels = res ? d.MipLevels : 1;
    }
    b->device->CreateShaderResourceView(res, &v, h);
}

void UavView(Backend* b, const Resource* r, bool buffer, uint32_t mip, D3D12_CPU_DESCRIPTOR_HANDLE h)
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
    ID3D12Resource*                  res = r ? r->res : nullptr;
    if (buffer)
    {
        v.Format             = DXGI_FORMAT_R32_TYPELESS;
        v.ViewDimension      = D3D12_UAV_DIMENSION_BUFFER;
        v.Buffer.Flags       = D3D12_BUFFER_UAV_FLAG_RAW;
        v.Buffer.NumElements = res ? UINT(res->GetDesc().Width / 4) : 0;
    }
    else
    {
        v.Format             = res ? UavFormat(res->GetDesc().Format) : DXGI_FORMAT_R8G8B8A8_UNORM;
        v.ViewDimension      = D3D12_UAV_DIMENSION_TEXTURE2D;
        v.Texture2D.MipSlice = mip;
    }
    b->device->CreateUnorderedAccessView(res, nullptr, &v, h);
}

// ---------------------------------------------------------------------------------- barriers

Resource* Res(Backend* b, FfxResourceInternal r)
{
    if (r.internalIndex <= 0 || size_t(r.internalIndex) >= b->resources.size() || !b->resources[r.internalIndex].used)
        return nullptr;
    return &b->resources[r.internalIndex];
}

void Transition(Backend* b, Resource* r, D3D12_RESOURCE_STATES s)
{
    if (!r || !r->res || r->state == s)
        return;
    // a combined read state already covers its parts
    if (s != kUavState && r->state != kUavState && (r->state & s) == s)
        return;
    D3D12_RESOURCE_BARRIER x{};
    x.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    x.Transition.pResource   = r->res;
    x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    x.Transition.StateBefore = r->state;
    x.Transition.StateAfter  = s;
    b->barriers.push_back(x);
    r->state = s;
}

void Flush(Backend* b, ID3D12GraphicsCommandList* cl)
{
    if (!b->barriers.empty())
        cl->ResourceBarrier(UINT(b->barriers.size()), b->barriers.data());
    b->barriers.clear();
}

void UavBarrier(ID3D12GraphicsCommandList* cl)
{
    D3D12_RESOURCE_BARRIER x{};
    x.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    cl->ResourceBarrier(1, &x);
}

uint32_t NewSlot(Backend* b)
{
    for (uint32_t i = 1; i < b->resources.size(); ++i)
        if (!b->resources[i].used)
            return i;
    b->resources.emplace_back();
    return uint32_t(b->resources.size() - 1);
}

// ---------------------------------------------------------------------------------- interface

FfxErrorCode SetMessageCallback(FfxInterface* i, FfxBackendMessage cb)
{
    Get(i)->message = cb;
    i->fpMessage    = cb;
    return FFX_OK;
}

FfxVersionNumber GetSdkVersion(FfxInterface*)
{
    return FFX_SDK_MAKE_VERSION(FFX_SDK_VERSION_MAJOR, FFX_SDK_VERSION_MINOR, FFX_SDK_VERSION_PATCH);
}

FfxErrorCode GetDeviceCapabilities(FfxInterface* i, FfxDeviceCapabilities* caps)
{
    Backend* b = Get(i);
    *caps      = {};
    D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_SHADER_MODEL_6_6};
    if (FAILED(b->device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))))
        sm.HighestShaderModel = D3D_SHADER_MODEL_5_1;
    switch (sm.HighestShaderModel)
    {
    case D3D_SHADER_MODEL_6_0: caps->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_0; break;
    case D3D_SHADER_MODEL_6_1: caps->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_1; break;
    case D3D_SHADER_MODEL_6_2: caps->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_2; break;
    case D3D_SHADER_MODEL_6_3: caps->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_3; break;
    case D3D_SHADER_MODEL_6_4: caps->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_4; break;
    case D3D_SHADER_MODEL_6_5: caps->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_5; break;
    case D3D_SHADER_MODEL_6_6: caps->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_6; break;
    default: caps->maximumSupportedShaderModel = FFX_SHADER_MODEL_5_1; break;
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 o1{};
    if (SUCCEEDED(b->device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &o1, sizeof(o1))))
        caps->waveLaneCountMin = o1.WaveLaneCountMin, caps->waveLaneCountMax = o1.WaveLaneCountMax;
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 o4{};
    if (SUCCEEDED(b->device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &o4, sizeof(o4))))
        caps->fp16Supported = o4.Native16BitShaderOpsSupported;
    // The passes need SM 6.2 with native 16-bit types, the network SM 6.4 (packed int8 dot).
    const bool ok               = caps->maximumSupportedShaderModel >= FFX_SHADER_MODEL_6_4 && caps->fp16Supported;
    caps->tensorSupported       = ok;
    caps->dataGraphSupported    = ok;
    caps->computeSupportTensor  = false;  // tensors are plain buffers and textures
    caps->fragmentSupportTensor = false;
    i->deviceCapabilities       = *caps;
    i->devCapInitialized        = true;
    return FFX_OK;
}

FfxErrorCode CreateBackendContext(FfxInterface* i, FfxEffect, FfxEffectBindlessConfig*, FfxUInt32* id)
{
    Backend* b = Get(i);
    if (b->refCount == 0)
    {
        D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kGpuDescriptors, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        D3D12_DESCRIPTOR_HEAP_DESC cd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kCpuDescriptors, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        D3D12_HEAP_PROPERTIES      hp{};
        hp.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width            = kUploadBytes;
        rd.Height           = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels        = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_RANGE none{0, 0};
        if (FAILED(b->device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&b->gpuHeap))) ||
            FAILED(b->device->CreateDescriptorHeap(&cd, IID_PPV_ARGS(&b->cpuHeap))) ||
            FAILED(b->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                      IID_PPV_ARGS(&b->upload))) ||
            FAILED(b->upload->Map(0, &none, reinterpret_cast<void**>(&b->uploadPtr))))
            return FFX_ERROR_BACKEND_API_ERROR;
        b->descSize = b->device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        b->staging.resize(kStagingBytes);
        b->resources.resize(1);  // index 0: the null resource
        b->wmma = !g_forceDp4a && rdnu::EnableAmdWaveMatrixInt8(b->device);
        FfxDeviceCapabilities caps;
        GetDeviceCapabilities(i, &caps);
    }
    uint32_t c = 0;
    while (c < b->contexts.size() && b->contexts[c])
        ++c;
    if (c == b->contexts.size())
        b->contexts.push_back(false);
    b->contexts[c] = true;
    *id            = c;
    ++b->refCount;
    return FFX_OK;
}

FfxErrorCode DestroyBackendContext(FfxInterface* i, FfxUInt32 id)
{
    Backend* b = Get(i);
    if (id >= b->contexts.size() || !b->contexts[id])
        return FFX_ERROR_INVALID_ARGUMENT;
    for (Resource& r : b->resources)
        if (r.used && r.context == id)
        {
            if (!r.external)
                Release(r.res);
            r = Resource();
        }
    b->contexts[id] = false;
    if (--b->refCount == 0)
    {
        b->upload->Unmap(0, nullptr);
        Release(b->upload);
        Release(b->gpuHeap);
        Release(b->cpuHeap);
        b->jobs.clear();
        b->resources.clear();
    }
    return FFX_OK;
}

FfxErrorCode GetEffectGpuMemoryUsage(FfxInterface* i, FfxUInt32 id, FfxEffectMemoryUsage* out)
{
    Backend* b = Get(i);
    *out       = {};
    for (Resource& r : b->resources)
        if (r.used && !r.external && r.context == id && r.res)
        {
            D3D12_RESOURCE_DESC d = r.res->GetDesc();
            out->totalUsageInBytes += b->device->GetResourceAllocationInfo(0, 1, &d).SizeInBytes;
        }
    return FFX_OK;
}

FfxErrorCode CreateResource(FfxInterface* i, const FfxCreateResourceDescription* cd, FfxUInt32 id, FfxResourceInternal* out)
{
    Backend*                      b = Get(i);
    const FfxResourceDescription& d = cd->resourceDescription;
    D3D12_RESOURCE_DESC           rd{};
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = UINT16(d.mipCount ? d.mipCount : 1);
    rd.SampleDesc.Count = 1;
    if (d.usage & FFX_RESOURCE_USAGE_UAV)
        rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    const bool imageTensor = d.type == FFX_RESOURCE_TYPE_TENSOR && (d.flags & FFX_RESOURCE_FLAGS_IMAGE_ALIASED);
    if (d.type == FFX_RESOURCE_TYPE_BUFFER || (d.type == FFX_RESOURCE_TYPE_TENSOR && !imageTensor))
    {
        uint64_t bytes = d.type == FFX_RESOURCE_TYPE_BUFFER ? d.size : uint64_t(d.width) * d.height * d.channel * (d.batchSize ? d.batchSize : 1);
        if (d.type == FFX_RESOURCE_TYPE_TENSOR && d.format != FFX_SURFACE_FORMAT_R8_SINT)
            return FFX_ERROR_INVALID_ARGUMENT;  // the network is INT8
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width     = (bytes + 255) & ~uint64_t(255);
        rd.Height    = 1;
        rd.MipLevels = 1;
        rd.Layout    = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    }
    else if (d.type == FFX_RESOURCE_TYPE_TEXTURE2D || imageTensor)
    {
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width     = d.width;
        rd.Height    = d.height;
        rd.Format    = imageTensor ? DXGI_FORMAT_R8G8B8A8_SNORM : Dxgi(d.format);
        if (imageTensor && (d.channel != 4 || d.format != FFX_SURFACE_FORMAT_R8_SINT))
            return FFX_ERROR_INVALID_ARGUMENT;
        if (d.usage & FFX_RESOURCE_USAGE_RENDERTARGET)
            rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    }
    else
        return FFX_ERROR_INVALID_ARGUMENT;
    if (cd->initData.type != FFX_RESOURCE_INIT_DATA_TYPE_UNINITIALIZED)
        return FFX_ERROR_INVALID_ARGUMENT;  // NSS creates everything uninitialised and clears it

    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = cd->heapType == FFX_HEAP_TYPE_UPLOAD ? D3D12_HEAP_TYPE_UPLOAD
            : cd->heapType == FFX_HEAP_TYPE_READBACK ? D3D12_HEAP_TYPE_READBACK : D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_STATES state = hp.Type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ
                                : hp.Type == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST : D3dState(cd->initialState);
    ID3D12Resource* res = nullptr;
    if (FAILED(b->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&res))))
    {
        Print(b, std::string("NSS backend: cannot create ") + (cd->name ? cd->name : "resource"));
        return FFX_ERROR_OUT_OF_MEMORY;
    }
    uint32_t  slot = NewSlot(b);
    Resource& r    = b->resources[slot];
    r              = Resource();
    r.res          = res;
    r.desc         = d;
    r.state = r.restore = state;
    r.context           = id;
    r.used              = true;
    r.name              = cd->name ? cd->name : "";
    *out                = {};
    out->internalIndex  = int32_t(slot);
    return FFX_OK;
}

FfxErrorCode DestroyResource(FfxInterface* i, FfxResourceInternal res, FfxUInt32)
{
    Resource* r = Res(Get(i), res);
    if (r && !r->external)
    {
        Release(r->res);
        *r = Resource();
    }
    return FFX_OK;
}

FfxErrorCode RegisterResource(FfxInterface* i, const FfxResource* in, FfxUInt32 id, FfxResourceInternal* out)
{
    Backend* b = Get(i);
    *out       = {};
    if (!in->resource)
        return FFX_OK;  // null resource
    uint32_t  slot = NewSlot(b);
    Resource& r    = b->resources[slot];
    r              = Resource();
    r.res          = static_cast<ID3D12Resource*>(in->resource);
    r.desc         = in->description;
    r.state = r.restore = D3dState(in->state);
    r.context           = id;
    r.used = r.external = true;
    r.name              = in->name;
    out->internalIndex  = int32_t(slot);
    return FFX_OK;
}

FfxErrorCode UnregisterResources(FfxInterface* i, FfxCommandList commandList, FfxUInt32 id)
{
    Backend* b  = Get(i);
    auto*    cl = static_cast<ID3D12GraphicsCommandList*>(commandList);
    for (Resource& r : b->resources)
        if (r.used && r.external && r.context == id)
        {
            Transition(b, &r, r.restore);
            r.used = false;
        }
    Flush(b, cl);
    for (Resource& r : b->resources)
        if (!r.used && r.external)
            r = Resource();
    return FFX_OK;
}

FfxResource GetResource(FfxInterface* i, FfxResourceInternal res)
{
    FfxResource out{};
    if (Resource* r = Res(Get(i), res))
    {
        out.resource    = r->res;
        out.description = r->desc;
        out.state       = FfxState(r->state);
        std::strncpy(out.name, r->name.c_str(), FFX_RESOURCE_NAME_SIZE - 1);
    }
    return out;
}

FfxResourceDescription GetResourceDescription(FfxInterface* i, FfxResourceInternal res)
{
    Resource* r = Res(Get(i), res);
    return r ? r->desc : FfxResourceDescription{};
}

FfxErrorCode RegisterStaticResource(FfxInterface*, const FfxStaticResourceDescription*, FfxUInt32)
{
    return FFX_ERROR_INVALID_ARGUMENT;  // no bindless resources in NSS
}

FfxErrorCode MapResource(FfxInterface* i, FfxResourceInternal res, void** ptr)
{
    Resource* r = Res(Get(i), res);
    return r && SUCCEEDED(r->res->Map(0, nullptr, ptr)) ? FFX_OK : FFX_ERROR_INVALID_ARGUMENT;
}

FfxErrorCode UnmapResource(FfxInterface* i, FfxResourceInternal res)
{
    if (Resource* r = Res(Get(i), res))
        r->res->Unmap(0, nullptr);
    return FFX_OK;
}

FfxErrorCode StageConstantBufferData(FfxInterface* i, void* data, FfxUInt32 size, FfxConstantBuffer* cb)
{
    Backend* b = Get(i);
    size       = (size + 3) & ~3u;
    if (size > kStagingBytes)
        return FFX_ERROR_INVALID_ARGUMENT;
    if (b->stagingHead + size > kStagingBytes)
        b->stagingHead = 0;
    uint8_t* dst = &b->staging[b->stagingHead];
    std::memcpy(dst, data, size);
    b->stagingHead += size;
    cb->data            = reinterpret_cast<uint32_t*>(dst);
    cb->num32BitEntries = size / 4;
    return FFX_OK;
}

const char* PassName(FfxPass pass)
{
    switch (pass)
    {
    case FFX_NSS_PASS_DEPTH_SCATTER: return "depth_scatter";
    case FFX_NSS_PASS_DISOCCLUSION_MASK: return "disocclusion_mask_lq";
    case FFX_NSS_PASS_PREPROCESS: return "pre_process";
    case FFX_NSS_PASS_GENERATE_OFFSET_LUT: return "generate_offset_lut";
    case FFX_NSS_PASS_POSTPROCESS: return "post_process";
    case FFX_NSS_PASS_DEBUG_VIEW: return "debug_view";
    default: return nullptr;
    }
}

D3D12_STATIC_SAMPLER_DESC StaticSampler(const FfxSamplerDescription& s, UINT slot)
{
    auto mode = [](FfxAddressMode m) {
        switch (m)
        {
        case FFX_ADDRESS_MODE_WRAP: return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        case FFX_ADDRESS_MODE_MIRROR: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
        case FFX_ADDRESS_MODE_BORDER: return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
        case FFX_ADDRESS_MODE_MIRROR_ONCE: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
        default: return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        }
    };
    D3D12_STATIC_SAMPLER_DESC d{};
    d.Filter = s.filter == FFX_FILTER_TYPE_MINMAGMIP_POINT    ? D3D12_FILTER_MIN_MAG_MIP_POINT
             : s.filter == FFX_FILTER_TYPE_MINMAGMIP_LINEAR   ? D3D12_FILTER_MIN_MAG_MIP_LINEAR
                                                              : D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    d.AddressU         = mode(s.addressModeU);
    d.AddressV         = mode(s.addressModeV);
    d.AddressW         = mode(s.addressModeW);
    d.MaxLOD           = D3D12_FLOAT32_MAX;
    d.ComparisonFunc   = D3D12_COMPARISON_FUNC_NEVER;
    d.ShaderRegister   = slot;
    d.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    return d;
}

FfxErrorCode CreateComputePipeline(FfxInterface* i, FfxEffect effect, FfxPass pass, uint32_t options, const FfxPipelineDescription* desc,
                                   FfxUInt32, FfxPipelineState* out)
{
    Backend*    b    = Get(i);
    const char* name = PassName(pass);
    if (effect != FFX_EFFECT_NSS || !name)
        return FFX_ERROR_INVALID_ARGUMENT;
    if (!(options & NSS_SHADER_PERMUTATION_ALLOW_16BIT) || !(options & NSS_SHADER_PERMUTATION_QUANTIZED))
    {
        Print(b, "NSS backend: only the 16-bit INT8 shader permutations are built");
        return FFX_ERROR_INVALID_ARGUMENT;
    }
    uint32_t bits = 0;
    bits |= options & NSS_SHADER_PERMUTATION_REVERSE_Z ? uint32_t(rdnu::kPassReverseZ) : 0u;
    bits |= options & NSS_SHADER_PERMUTATION_SCALE_PRESET_X2 ? uint32_t(rdnu::kPassScaleX2) : 0u;
    bits |= options & NSS_SHADER_PERMUTATION_MANAGE_HISTORY ? uint32_t(rdnu::kPassManageHistory) : 0u;
    const uint32_t quality = (options >> NSS_SHADER_PERMUTATION_QUALITY_MODE_SHIFT) & NSS_SHADER_QUALITY_MODE_MASK;
    std::string    shaderName;
    for (const rdnu::PassShader& p : rdnu::PassShaders())
        if (!std::strcmp(p.name, name))
            shaderName = rdnu::PassShaderName(p, quality, bits);
    const rdnu::ShaderBlob* blob = rdnu::FindShader(shaderName.c_str());
    if (!blob)
    {
        Print(b, "NSS backend: shader " + shaderName + " is not built (only the high quality model ships)");
        return FFX_ERROR_INVALID_ARGUMENT;
    }

    *out = {};
    auto* p = new Pipeline();
    int   cbSlot = -1;
    for (uint32_t k = 0; k < blob->bindingCount; ++k)
    {
        const rdnu::ShaderBinding& s = blob->bindings[k];
        FfxResourceBinding*        dst = nullptr;
        switch (s.kind)
        {
        case rdnu::BindingKind::Cbv: cbSlot = int(s.slot); dst = &out->constantBufferBindings[out->constCount++]; break;
        case rdnu::BindingKind::SrvTexture: dst = &out->srvTextureBindings[out->srvTextureCount++]; break;
        case rdnu::BindingKind::UavTexture: dst = &out->uavTextureBindings[out->uavTextureCount++]; break;
        case rdnu::BindingKind::SrvBuffer: dst = &out->srvBufferBindings[out->srvBufferCount++]; break;
        case rdnu::BindingKind::UavBuffer: dst = &out->uavBufferBindings[out->uavBufferCount++]; break;
        case rdnu::BindingKind::Sampler: continue;
        }
        dst->slotIndex = s.slot;
        dst->bindCount = 1;
        std::strncpy(dst->name, s.name, FFX_RESOURCE_NAME_SIZE - 1);
        if (s.kind == rdnu::BindingKind::SrvTexture || s.kind == rdnu::BindingKind::SrvBuffer)
            p->srvSlots = std::max(p->srvSlots, s.slot + 1);
        if (s.kind == rdnu::BindingKind::UavTexture || s.kind == rdnu::BindingKind::UavBuffer)
            p->uavSlots = std::max(p->uavSlots, s.slot + 1);
    }

    D3D12_ROOT_PARAMETER   params[3] = {};
    D3D12_DESCRIPTOR_RANGE srv{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, p->srvSlots, 0, 0, 0};
    D3D12_DESCRIPTOR_RANGE uav{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, p->uavSlots, 0, 0, 0};
    UINT                   n = 0;
    if (cbSlot >= 0)
    {
        params[n].ParameterType             = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[n].Descriptor.ShaderRegister = UINT(cbSlot);
        ++n;
    }
    if (p->srvSlots)
    {
        p->srvParam                                 = n;
        params[n].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[n].DescriptorTable.NumDescriptorRanges = 1;
        params[n].DescriptorTable.pDescriptorRanges   = &srv;
        ++n;
    }
    if (p->uavSlots)
    {
        p->uavParam                                   = n;
        params[n].ParameterType                       = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[n].DescriptorTable.NumDescriptorRanges = 1;
        params[n].DescriptorTable.pDescriptorRanges   = &uav;
        ++n;
    }
    std::vector<D3D12_STATIC_SAMPLER_DESC> samplers;
    for (size_t k = 0; k < desc->samplerCount; ++k)
        samplers.push_back(StaticSampler(desc->samplers[k], UINT(k)));
    D3D12_ROOT_SIGNATURE_DESC rs{};
    rs.NumParameters     = n;
    rs.pParameters       = params;
    rs.NumStaticSamplers = UINT(samplers.size());
    rs.pStaticSamplers   = samplers.data();
    ID3DBlob* sig = nullptr;
    ID3DBlob* err = nullptr;
    HRESULT   hr  = D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
    if (SUCCEEDED(hr))
        hr = b->device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&p->rootSig));
    Release(sig);
    Release(err);
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = p->rootSig;
    pd.CS             = {blob->data, blob->size};
    if (SUCCEEDED(hr))
        hr = b->device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&p->pso));
    if (FAILED(hr))
    {
        Release(p->rootSig);
        delete p;
        Print(b, "NSS backend: cannot create the pipeline for " + shaderName);
        return FFX_ERROR_BACKEND_API_ERROR;
    }
    out->pipeline      = p;
    out->rootSignature = p->rootSig;
    out->passId        = pass;
    std::strncpy(out->name, desc->name, FFX_RESOURCE_NAME_SIZE - 1);
    return FFX_OK;
}

FfxErrorCode CreateGraphicsPipeline(FfxInterface* i, FfxEffect, FfxPass, uint32_t, const FfxPipelineDescription*, FfxUInt32, FfxPipelineState*)
{
    Print(Get(i), "NSS backend: fragment passes are not supported, use compute");
    return FFX_ERROR_INVALID_ARGUMENT;
}

FfxErrorCode CreateDataGraphPipeline(FfxInterface* i, FfxEffect effect, FfxPass pass, uint32_t options, const FfxPipelineDescription* desc,
                                     FfxUInt32, uint32_t width, uint32_t height, FfxPipelineState* out)
{
    Backend* b = Get(i);
    if (effect != FFX_EFFECT_NSS || pass != FFX_NSS_PASS_DATA_GRAPH || !(options & NSS_SHADER_PERMUTATION_QUANTIZED))
        return FFX_ERROR_INVALID_ARGUMENT;
    // the network reads its input and writes the KPN as buffers, the feedback as a texture
    for (uint32_t k = 0; k < desc->dataGraphTensorInfoCount; ++k)
    {
        const FfxDataGraphTensorInfo& t = desc->dataGraphTensorInfo[k];
        const bool image = !std::strcmp(t.resourceName, "Resource_2_output") || !std::strcmp(t.resourceName, "r_temporal_feedback_tensor");
        if (t.bufferAliased == image)
        {
            Print(b, std::string("NSS backend: unexpected tensor aliasing for ") + t.resourceName);
            return FFX_ERROR_INVALID_ARGUMENT;
        }
    }
    const rdnu::EmbeddedFile* manifest = rdnu::FindModelFile("nss.rdnm");
    const rdnu::EmbeddedFile* weights  = rdnu::FindModelFile("nss_w8.bin");
    if (!manifest || !weights)
        return FFX_ERROR_INVALID_ARGUMENT;

    auto* p   = new Pipeline();
    p->engine = new rdnu::EngineDx12();
    p->width  = width;
    p->height = height;
    rdnu::EngineDx12Desc ed;
    ed.device        = b->device;
    ed.manifest      = manifest->data;
    ed.manifestBytes = manifest->size;
    ed.weights       = weights->data;
    ed.weightBytes   = weights->size;
    ed.maxWidth      = width;
    ed.maxHeight     = height;
    ed.useWmma       = b->wmma;
    std::string err;
    if (!p->engine->Create(ed, err))
    {
        Print(b, "NSS backend: " + err);
        delete p->engine;
        delete p;
        return FFX_ERROR_BACKEND_API_ERROR;
    }
    *out                = {};
    out->pipeline       = p;
    out->passId         = pass;
    out->srvTensorCount = 1;
    out->uavTensorCount = 2;
    std::strncpy(out->srvTensorBindings[0].name, "Resource_0_input", FFX_RESOURCE_NAME_SIZE - 1);
    std::strncpy(out->uavTensorBindings[0].name, "Resource_1_output", FFX_RESOURCE_NAME_SIZE - 1);
    std::strncpy(out->uavTensorBindings[1].name, "Resource_2_output", FFX_RESOURCE_NAME_SIZE - 1);
    std::strncpy(out->name, desc->name, FFX_RESOURCE_NAME_SIZE - 1);
    return FFX_OK;
}

FfxErrorCode CreateOpticalFlowPipeline(FfxInterface*, const char*, const FfxOpticalFlowDescription&, FfxUInt32, FfxPipelineState*)
{
    return FFX_ERROR_INVALID_ARGUMENT;
}

FfxErrorCode DestroyPipeline(FfxInterface*, FfxPipelineState* state, FfxUInt32)
{
    if (!state)
        return FFX_ERROR_INVALID_POINTER;
    if (auto* p = static_cast<Pipeline*>(state->pipeline))
    {
        Release(p->pso);
        Release(p->rootSig);
        delete p->engine;
        delete p;
    }
    state->pipeline      = nullptr;
    state->rootSignature = nullptr;
    return FFX_OK;
}

FfxErrorCode GetPermutationBlob(FfxEffect, FfxPass, uint32_t, FfxShaderBlob*, FfxShaderBlob*, FfxDataGraphBlob*)
{
    return FFX_ERROR_INVALID_ARGUMENT;  // the backend resolves its own blobs
}

FfxErrorCode ScheduleGpuJob(FfxInterface* i, const FfxGpuJobDescription* job)
{
    Get(i)->jobs.push_back(*job);
    return FFX_OK;
}

// ---------------------------------------------------------------------------------- execution

void ExecuteCompute(Backend* b, ID3D12GraphicsCommandList* cl, const FfxComputeJobDescription& j)
{
    auto*                     p = static_cast<Pipeline*>(j.pipeline.pipeline);
    const FfxPipelineState&   s = j.pipeline;
    for (uint32_t k = 0; k < s.srvTextureCount; ++k)
        Transition(b, Res(b, j.srvTextures[k].resource), kSrvState);
    for (uint32_t k = 0; k < s.srvBufferCount; ++k)
        Transition(b, Res(b, j.srvBuffers[k].resource), kSrvState);
    for (uint32_t k = 0; k < s.uavTextureCount; ++k)
        Transition(b, Res(b, j.uavTextures[k].resource), kUavState);
    for (uint32_t k = 0; k < s.uavBufferCount; ++k)
        Transition(b, Res(b, j.uavBuffers[k].resource), kUavState);
    Flush(b, cl);

    cl->SetComputeRootSignature(p->rootSig);
    cl->SetPipelineState(p->pso);
    UINT param = 0;
    if (s.constCount)
    {
        const FfxConstantBuffer& cb    = j.cbs[0];
        const uint64_t           bytes = (uint64_t(cb.num32BitEntries) * 4 + 255) & ~uint64_t(255);
        if (b->uploadHead + bytes > kUploadBytes)
            b->uploadHead = 0;
        std::memcpy(b->uploadPtr + b->uploadHead, cb.data, cb.num32BitEntries * 4);
        cl->SetComputeRootConstantBufferView(param++, b->upload->GetGPUVirtualAddress() + b->uploadHead);
        b->uploadHead += bytes;
    }
    if (p->srvSlots)
    {
        const uint32_t base = AllocGpu(b, p->srvSlots);
        for (uint32_t k = 0; k < p->srvSlots; ++k)
            SrvView(b, nullptr, false, Cpu(b->gpuHeap, b->descSize, base + k));
        for (uint32_t k = 0; k < s.srvTextureCount; ++k)
            SrvView(b, Res(b, j.srvTextures[k].resource), false, Cpu(b->gpuHeap, b->descSize, base + s.srvTextureBindings[k].slotIndex));
        for (uint32_t k = 0; k < s.srvBufferCount; ++k)
            SrvView(b, Res(b, j.srvBuffers[k].resource), true, Cpu(b->gpuHeap, b->descSize, base + s.srvBufferBindings[k].slotIndex));
        cl->SetComputeRootDescriptorTable(p->srvParam, Gpu(b->gpuHeap, b->descSize, base));
    }
    if (p->uavSlots)
    {
        const uint32_t base = AllocGpu(b, p->uavSlots);
        for (uint32_t k = 0; k < p->uavSlots; ++k)
            UavView(b, nullptr, false, 0, Cpu(b->gpuHeap, b->descSize, base + k));
        for (uint32_t k = 0; k < s.uavTextureCount; ++k)
            UavView(b, Res(b, j.uavTextures[k].resource), false, j.uavTextures[k].mip,
                    Cpu(b->gpuHeap, b->descSize, base + s.uavTextureBindings[k].slotIndex));
        for (uint32_t k = 0; k < s.uavBufferCount; ++k)
            UavView(b, Res(b, j.uavBuffers[k].resource), true, 0, Cpu(b->gpuHeap, b->descSize, base + s.uavBufferBindings[k].slotIndex));
        cl->SetComputeRootDescriptorTable(p->uavParam, Gpu(b->gpuHeap, b->descSize, base));
    }
    cl->Dispatch(j.dimensions[0], j.dimensions[1], j.dimensions[2]);
    UavBarrier(cl);
}

bool ExecuteDataGraph(Backend* b, ID3D12GraphicsCommandList* cl, const FfxDataGraphJobDescription& j)
{
    auto*     p        = static_cast<Pipeline*>(j.pipeline.pipeline);
    Resource* input    = Res(b, j.srvTensors[0].resource);
    Resource* kpn      = Res(b, j.uavTensors[0].resource);
    Resource* feedback = Res(b, j.uavTensors[1].resource);
    if (!p || !p->engine || !input || !kpn || !feedback)
        return false;
    Transition(b, input, kUavState);  // the network reads its input through a raw UAV
    Transition(b, kpn, kUavState);
    Transition(b, feedback, kUavState);
    Flush(b, cl);
    const uint32_t slot = AllocGpu(b, 1);
    UavView(b, feedback, false, 0, Cpu(b->gpuHeap, b->descSize, slot));
    std::string err;
    if (!p->engine->Record(cl, input->res->GetGPUVirtualAddress(), kpn->res->GetGPUVirtualAddress(), Gpu(b->gpuHeap, b->descSize, slot),
                           p->width, p->height, err))
    {
        Print(b, "NSS backend: " + err);
        return false;
    }
    return true;
}

void ExecuteClear(Backend* b, ID3D12GraphicsCommandList* cl, FfxResourceInternal target, const float* f, const uint32_t* u)
{
    Resource* r = Res(b, target);
    if (!r)
        return;
    Transition(b, r, kUavState);
    Flush(b, cl);
    const bool     buffer = IsBuffer(*r);
    const uint32_t g      = AllocGpu(b, 1);
    const uint32_t c      = b->cpuHead;
    b->cpuHead            = (b->cpuHead + 1) % kCpuDescriptors;
    UavView(b, r, buffer, 0, Cpu(b->gpuHeap, b->descSize, g));
    UavView(b, r, buffer, 0, Cpu(b->cpuHeap, b->descSize, c));
    if (f)
        cl->ClearUnorderedAccessViewFloat(Gpu(b->gpuHeap, b->descSize, g), Cpu(b->cpuHeap, b->descSize, c), r->res, f, 0, nullptr);
    else
        cl->ClearUnorderedAccessViewUint(Gpu(b->gpuHeap, b->descSize, g), Cpu(b->cpuHeap, b->descSize, c), r->res, u, 0, nullptr);
    UavBarrier(cl);
}

void ExecuteCopy(Backend* b, ID3D12GraphicsCommandList* cl, const FfxCopyJobDescription& j)
{
    Resource* src = Res(b, j.src);
    Resource* dst = Res(b, j.dst);
    if (!src || !dst)
        return;
    Transition(b, src, D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(b, dst, D3D12_RESOURCE_STATE_COPY_DEST);
    Flush(b, cl);
    if (IsBuffer(*src) && IsBuffer(*dst))
        cl->CopyBufferRegion(dst->res, j.dstOffset, src->res, j.srcOffset, j.size ? j.size : src->res->GetDesc().Width - j.srcOffset);
    else
        cl->CopyResource(dst->res, src->res);
}

FfxErrorCode ExecuteGpuJobs(FfxInterface* i, FfxCommandList commandList, FfxUInt32)
{
    Backend* b  = Get(i);
    auto*    cl = static_cast<ID3D12GraphicsCommandList*>(commandList);
    cl->SetDescriptorHeaps(1, &b->gpuHeap);
    FfxErrorCode result = FFX_OK;
    for (const FfxGpuJobDescription& job : b->jobs)
    {
        switch (job.jobType)
        {
        case FFX_GPU_JOB_COMPUTE: ExecuteCompute(b, cl, job.computeJobDescriptor); break;
        case FFX_GPU_JOB_DATA_GRAPH:
            if (!ExecuteDataGraph(b, cl, job.dataGraphJobDescription))
                result = FFX_ERROR_BACKEND_API_ERROR;
            break;
        case FFX_GPU_JOB_CLEAR_FLOAT: ExecuteClear(b, cl, job.clearJobDescriptor.target, job.clearJobDescriptor.color, nullptr); break;
        case FFX_GPU_JOB_CLEAR_UINT: ExecuteClear(b, cl, job.clearUintJobDescriptor.target, nullptr, job.clearUintJobDescriptor.color); break;
        case FFX_GPU_JOB_COPY: ExecuteCopy(b, cl, job.copyJobDescriptor); break;
        default: result = FFX_ERROR_INVALID_ARGUMENT; break;
        }
    }
    b->jobs.clear();
    return result;
}
}  // namespace

size_t ffxGetScratchMemorySizeDX12(size_t)
{
    return sizeof(Backend);
}

FfxErrorCode ffxGetInterfaceDX12(FfxInterface* i, ID3D12Device* device, void* scratch, size_t scratchSize, size_t)
{
    if (!i || !device || !scratch)
        return FFX_ERROR_INVALID_POINTER;
    if (scratchSize < sizeof(Backend))
        return FFX_ERROR_INSUFFICIENT_MEMORY;
    *i                                 = {};
    Backend* b                         = new (scratch) Backend();
    b->device                          = device;
    i->fpSetMessageCallback            = SetMessageCallback;
    i->fpGetSDKVersion                 = GetSdkVersion;
    i->fpGetEffectGpuMemoryUsage       = GetEffectGpuMemoryUsage;
    i->fpCreateBackendContext          = CreateBackendContext;
    i->fpGetDeviceCapabilities         = GetDeviceCapabilities;
    i->fpDestroyBackendContext         = DestroyBackendContext;
    i->fpCreateResource                = CreateResource;
    i->fpRegisterResource              = RegisterResource;
    i->fpGetResource                   = GetResource;
    i->fpUnregisterResources           = UnregisterResources;
    i->fpRegisterStaticResource        = RegisterStaticResource;
    i->fpGetResourceDescription        = GetResourceDescription;
    i->fpDestroyResource               = DestroyResource;
    i->fpMapResource                   = MapResource;
    i->fpUnmapResource                 = UnmapResource;
    i->fpStageConstantBufferDataFunc   = StageConstantBufferData;
    i->fpCreateComputePipeline         = CreateComputePipeline;
    i->fpCreateGraphicsPipeline        = CreateGraphicsPipeline;
    i->fpCreateDataGraphPipeline       = CreateDataGraphPipeline;
    i->fpCreateOpticalFlowPipeline     = CreateOpticalFlowPipeline;
    i->fpDestroyPipeline               = DestroyPipeline;
    i->fpScheduleGpuJob                = ScheduleGpuJob;
    i->fpExecuteGpuJobs                = ExecuteGpuJobs;
    i->fpGetPermutationBlobByIndex     = GetPermutationBlob;
    i->fpSwapChainConfigureFrameGeneration = nullptr;
    i->fpRegisterConstantBufferAllocator   = nullptr;
    i->scratchBuffer                   = scratch;
    i->scratchBufferSize               = scratchSize;
    i->device                          = device;
    return FFX_OK;
}

void ffxReleaseInterfaceDX12(FfxInterface* i)
{
    if (i && i->scratchBuffer)
        static_cast<Backend*>(i->scratchBuffer)->~Backend();
}

FfxDevice ffxGetDeviceDX12(ID3D12Device* device)
{
    return device;
}

FfxCommandList ffxGetCommandListDX12(ID3D12GraphicsCommandList* commandList)
{
    return commandList;
}

FfxResource ffxGetResourceDX12(ID3D12Resource* resource, FfxResourceStates state, const char* name)
{
    FfxResource r{};
    r.resource = resource;
    r.state    = state;
    if (name)
        std::strncpy(r.name, name, FFX_RESOURCE_NAME_SIZE - 1);
    if (!resource)
        return r;
    D3D12_RESOURCE_DESC d = resource->GetDesc();
    if (d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
    {
        r.description.type = FFX_RESOURCE_TYPE_BUFFER;
        r.description.size = uint32_t(d.Width);
    }
    else
    {
        r.description.type   = FFX_RESOURCE_TYPE_TEXTURE2D;
        r.description.width  = uint32_t(d.Width);
        r.description.height = d.Height;
        r.description.depth  = d.DepthOrArraySize;
    }
    r.description.format   = Ffx(d.Format);
    r.description.mipCount = d.MipLevels;
    r.description.usage    = FFX_RESOURCE_USAGE_READ_ONLY;
    if (d.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
        r.description.usage = FfxResourceUsage(r.description.usage | FFX_RESOURCE_USAGE_UAV);
    if (d.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
        r.description.usage = FfxResourceUsage(r.description.usage | FFX_RESOURCE_USAGE_RENDERTARGET);
    return r;
}

void ffxNssDx12ForceDp4a(bool force)
{
    g_forceDp4a = force;
}
