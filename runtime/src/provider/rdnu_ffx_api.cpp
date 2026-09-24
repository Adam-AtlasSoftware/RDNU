// rdnu_ffx_api.cpp - the FidelityFX API (ffx_api.h) with NSS as the upscaler. The same DLL
// works as FSR 3.1's amd_fidelityfx_dx12.dll, as FidelityFX SDK 2's
// amd_fidelityfx_upscaler_dx12.dll behind AMD's loader, and as OptiScaler's FSR 3.1 DX12
// upscaler. Requests for anything but upscaling (frame generation, swap chains) and hosts that
// pick another version go to the AMD DLL it replaced, renamed <name>_original.dll.
//
// An NSS context is fixed to one render and output size: a new size creates a new context
// (history restarts) and the old one is destroyed once the GPU is done with it. Pipelines and
// the network engine are shared, so a new context only allocates its own textures.
//
// RDNU_LOG=<file> appends every message there; games rarely pass a message callback.
// RDNU_FORCE_DP4A=1 runs the DP4a network kernels on RDNA3 too (A/B timing).
// RDNU_CAPTURE=<dir> dumps every dispatch (rdnu_capture.h). With RDNU_COMPARE=<version> as well,
// AMD's upscaler of that version (a substring of its name, or any) runs on the same inputs into
// a private texture captured as "reference", for runtime/tools/eval/compare.py.
#include "rdnu_capture.h"
#include "rdnu_upscale_map.h"

#include "../backend_dx12/ffx_nss_dx12.h"

#include <FidelityFX/host/ffx_nss.h>
#include <dx12/ffx_api_dx12.h>
#include <ffx_api.h>
#include <ffx_upscale.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace
{
constexpr uint64_t kVersionId   = 0x52444E5501000000ull;  // "RDNU" 1.0.0
constexpr const char* kVersion  = "RDNU NSS 1.0.0";
constexpr uint64_t kRetireAfter = 8;  // dispatches before an old context is freed without a GPU marker
constexpr size_t   kMaxNss      = 8;

std::mutex      g_lock;
std::set<void*> g_ours;
ffxApiMessage   g_message = nullptr;

void Say(uint32_t type, const std::string& s)
{
    static FILE* log = [] {
        const char* path = std::getenv("RDNU_LOG");
        return path && *path ? std::fopen(path, "a") : nullptr;
    }();
    if (log)
    {
        std::fprintf(log, "%s: %s\n", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", s.c_str());
        std::fflush(log);
    }
    if (!g_message)
        return;
    std::wstring w(s.begin(), s.end());
    g_message(type, (L"RDNU: " + w).c_str());
}

void BackendMessage(uint32_t type, const char* m)
{
    Say(type == FFX_MESSAGE_TYPE_ERROR ? FFX_API_MESSAGE_TYPE_ERROR : FFX_API_MESSAGE_TYPE_WARNING, m);
}

void NssMessage(FfxMsgType type, const char* m)
{
    BackendMessage(type, m);
}

// ------------------------------------------------------------------------ AMD's DLL, forwarded

struct Original
{
    PfnFfxCreateContext  create    = nullptr;
    PfnFfxDestroyContext destroy   = nullptr;
    PfnFfxConfigure      configure = nullptr;
    PfnFfxQuery          query     = nullptr;
    PfnFfxDispatch       dispatch  = nullptr;
};

#ifdef _WIN32
template <typename T>
T Proc(HMODULE m, const char* name)
{
    return reinterpret_cast<T>(reinterpret_cast<void (*)()>(GetProcAddress(m, name)));
}
#endif

const Original* LoadOriginal()
{
#ifdef _WIN32
    static const Original o = [] {
        Original r;
        HMODULE  self = nullptr;
        wchar_t  path[MAX_PATH];
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&LoadOriginal), &self);
        const DWORD n = GetModuleFileNameW(self, path, MAX_PATH);
        if (n < 4 || n >= MAX_PATH)
            return r;
        const std::wstring original = std::wstring(path, n - 4) + L"_original.dll";
        if (HMODULE m = LoadLibraryW(original.c_str()))
        {
            r.create    = Proc<PfnFfxCreateContext>(m, "ffxCreateContext");
            r.destroy   = Proc<PfnFfxDestroyContext>(m, "ffxDestroyContext");
            r.configure = Proc<PfnFfxConfigure>(m, "ffxConfigure");
            r.query     = Proc<PfnFfxQuery>(m, "ffxQuery");
            r.dispatch  = Proc<PfnFfxDispatch>(m, "ffxDispatch");
        }
        return r;
    }();
    return o.create && o.destroy && o.configure && o.query && o.dispatch ? &o : nullptr;
#else
    return nullptr;
#endif
}

// ------------------------------------------------------------------ SDK 2 loader interface

// AMD's SDK 2 loader creates contexts through the exports below, then calls the ffxProvider
// object each context starts with (api/internal/ffx_provider.h). Its virtual table is spelled
// out as the loader's compiler lays it out, MSVC on Windows (one destructor slot) and the
// Itanium ABI elsewhere (two), so a DLL from any compiler matches it.
struct Allocator
{
    const ffxAllocationCallbacks* cb;
};

#ifdef _WIN32
constexpr int kDestructorSlots = 1;
#else
constexpr int kDestructorSlots = 2;
#endif

struct ProviderVtable
{
    void*           destructor[kDestructorSlots];
    bool            (*canProvide)(const void*, uint64_t);
    bool            (*isSupported)(const void*, void*);
    uint64_t        (*getId)(const void*);
    const char*     (*getVersionName)(const void*);
    ffxReturnCode_t (*createContext)(const void*, ffxContext*, ffxCreateContextDescHeader*, Allocator&);
    ffxReturnCode_t (*destroyContext)(const void*, ffxContext*, Allocator&);
    ffxReturnCode_t (*configure)(const void*, ffxContext*, const ffxConfigureDescHeader*);
    ffxReturnCode_t (*query)(const void*, ffxContext*, ffxQueryDescHeader*);
    ffxReturnCode_t (*dispatch)(const void*, ffxContext*, const ffxDispatchDescHeader*);
};

struct Provider
{
    const ProviderVtable* vtable;
};

extern const Provider g_provider;

// ------------------------------------------------------------------------------- upscaler

struct Upscaler
{
    const Provider*                provider = &g_provider;  // must stay first
    ID3D12Device*                  device = nullptr;
    uint32_t                       flags  = 0;
    FfxApiDimensions2D             maxRender{}, maxUpscale{};
    std::vector<uint8_t>           scratch;
    FfxInterface                   iface{};
    std::unique_ptr<FfxNssContext> nss;
    FfxApiDimensions2D             nssRender{}, nssUpscale{};
    bool                           fresh = true;
    std::vector<std::pair<std::unique_ptr<FfxNssContext>, uint64_t>> retired;
    std::vector<std::pair<ID3D12Resource*, uint64_t>>                 retiredTextures;
    ID3D12Resource*                sharpenInput = nullptr;  // NSS output when sharpening
    ID3D12Resource*                motion       = nullptr;  // prepared motion vectors
    FfxApiFloatCoords2D            lastJitter{};
    ID3D12Resource*                marker       = nullptr;  // readback: dispatches the GPU has finished
    const volatile uint32_t*       done         = nullptr;
    uint64_t                       frame        = 0;
    bool                           warnedColour = false;
    std::unique_ptr<rdnu::Capture> capture;
    ffxContext                     reference       = nullptr;  // AMD's upscaler on the same inputs
    ID3D12Resource*                referenceOutput = nullptr;
};

const ffxApiHeader* Find(const ffxApiHeader* h, uint64_t type)
{
    for (; h; h = h->pNext)
        if (h->type == type)
            return h;
    return nullptr;
}

bool Ours(void* c)
{
    std::lock_guard<std::mutex> l(g_lock);
    return c && g_ours.count(c);
}

D3D12_RESOURCE_STATES D3dState(uint32_t s)
{
    switch (s)
    {
    case FFX_API_RESOURCE_STATE_UNORDERED_ACCESS: return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    case FFX_API_RESOURCE_STATE_COMPUTE_READ: return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    case FFX_API_RESOURCE_STATE_PIXEL_READ: return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    case FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ:
        return D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    case FFX_API_RESOURCE_STATE_COPY_SRC: return D3D12_RESOURCE_STATE_COPY_SOURCE;
    case FFX_API_RESOURCE_STATE_COPY_DEST: return D3D12_RESOURCE_STATE_COPY_DEST;
    case FFX_API_RESOURCE_STATE_GENERIC_READ: return D3D12_RESOURCE_STATE_GENERIC_READ;
    case FFX_API_RESOURCE_STATE_INDIRECT_ARGUMENT: return D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    case FFX_API_RESOURCE_STATE_RENDER_TARGET: return D3D12_RESOURCE_STATE_RENDER_TARGET;
    case FFX_API_RESOURCE_STATE_DEPTH_ATTACHMENT: return D3D12_RESOURCE_STATE_DEPTH_WRITE;
    default: return D3D12_RESOURCE_STATE_COMMON;
    }
}

FfxResource Resource(const FfxApiResource& r, const char* name)
{
    return ffxGetResourceDX12(static_cast<ID3D12Resource*>(r.resource), FfxResourceStates(rdnu::NssStateFromApi(r.state)), name);
}

void Retire(Upscaler* u)
{
    if (u->nss)
        u->retired.push_back({std::move(u->nss), u->frame});
}

// Dispatches the GPU has finished, by the marker, else assumed after kRetireAfter more.
uint64_t Completed(const Upscaler* u)
{
    const uint64_t assumed = u->frame > kRetireAfter ? u->frame - kRetireAfter : 0;
    return u->done ? std::max<uint64_t>(assumed, *u->done) : assumed;
}

// Something retired after dispatch n is free once the GPU has finished n dispatches.
bool Idle(const Upscaler* u, uint64_t retiredAt)
{
    return Completed(u) >= retiredAt;
}

void FreeRetired(Upscaler* u, bool all)
{
    for (size_t i = 0; i < u->retired.size();)
        if (all || Idle(u, u->retired[i].second))
        {
            ffxNssContextDestroy(u->retired[i].first.get());
            u->retired.erase(u->retired.begin() + long(i));
        }
        else
            ++i;
    for (size_t i = 0; i < u->retiredTextures.size();)
        if (all || Idle(u, u->retiredTextures[i].second))
        {
            u->retiredTextures[i].first->Release();
            u->retiredTextures.erase(u->retiredTextures.begin() + long(i));
        }
        else
            ++i;
}

FfxErrorCode CreateNss(Upscaler* u, FfxApiDimensions2D render, FfxApiDimensions2D upscale)
{
    std::vector<std::string> warnings;
    FfxNssContextDescription d{};
    d.qualityMode      = FFX_NSS_SHADER_QUALITY_MODE_QUALITY;
    d.flags            = rdnu::NssFlagsFromUpscale(u->flags, warnings);
    d.renderSize       = {render.width, render.height};
    d.upscaleSize      = {upscale.width, upscale.height};
    d.displaySize      = d.upscaleSize;
    d.backendInterface = u->iface;
    d.fpMessage        = NssMessage;
    auto         nss   = std::make_unique<FfxNssContext>();
    FfxErrorCode e     = ffxNssContextCreate(nss.get(), &d);
    if (e != FFX_OK)
        return e;
    Retire(u);
    u->nss        = std::move(nss);
    u->nssRender  = render;
    u->nssUpscale = upscale;
    u->fresh      = true;
    return FFX_OK;
}

// An internal texture of the given size, replaced (and the old one retired) when it changes.
ID3D12Resource* Internal(Upscaler* u, ID3D12Resource*& slot, uint32_t width, uint32_t height, DXGI_FORMAT format, D3D12_RESOURCE_STATES state)
{
    if (slot)
    {
        const D3D12_RESOURCE_DESC t = slot->GetDesc();
        if (t.Width == width && t.Height == height)
            return slot;
        u->retiredTextures.push_back({slot, u->frame});
        slot = nullptr;
    }
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d{};
    d.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width            = width;
    d.Height           = height;
    d.DepthOrArraySize = 1;
    d.MipLevels        = 1;
    d.Format           = format;
    d.SampleDesc.Count = 1;
    d.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    u->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&slot));
    return slot;
}

// Captures this dispatch; with a reference context, AMD's upscaler runs on the same inputs first.
void Record(Upscaler* u, ID3D12GraphicsCommandList* cl, const ffxDispatchDescUpscale* d, FfxApiDimensions2D upscale)
{
    ID3D12Resource* output = static_cast<ID3D12Resource*>(d->output.resource);
    if (u->reference)
    {
        const D3D12_RESOURCE_DESC o = output->GetDesc();
        Internal(u, u->referenceOutput, uint32_t(o.Width), o.Height, o.Format, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ffxDispatchDescUpscale r = *d;
        r.header.pNext           = nullptr;
        r.output                 = ffxApiGetResourceDX12(u->referenceOutput, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        if (!u->referenceOutput || LoadOriginal()->dispatch(&u->reference, &r.header) != FFX_API_RETURN_OK)
            Say(FFX_API_MESSAGE_TYPE_WARNING, "the reference upscaler failed");
    }
    const rdnu::CaptureImage images[] = {
        {"color", static_cast<ID3D12Resource*>(d->color.resource), D3dState(d->color.state)},
        {"depth", static_cast<ID3D12Resource*>(d->depth.resource), D3dState(d->depth.state)},
        {"motion", static_cast<ID3D12Resource*>(d->motionVectors.resource), D3dState(d->motionVectors.state)},
        {"exposure", static_cast<ID3D12Resource*>(d->exposure.resource), D3dState(d->exposure.state)},
        {"output", output, D3dState(d->output.state)},
        {"reference", u->reference ? u->referenceOutput : nullptr, D3D12_RESOURCE_STATE_UNORDERED_ACCESS},
    };
    rdnu::CaptureParams p;
    p.jitter[0]   = d->jitterOffset.x;
    p.jitter[1]   = d->jitterOffset.y;
    p.mvScale[0]  = d->motionVectorScale.x;
    p.mvScale[1]  = d->motionVectorScale.y;
    p.camera[0]   = d->cameraNear;
    p.camera[1]   = d->cameraFar;
    p.camera[2]   = d->cameraFovAngleVertical;
    p.preExposure = d->preExposure;
    p.sharpness   = d->enableSharpening ? d->sharpness : 0.0f;
    p.render[0]   = d->renderSize.width;
    p.render[1]   = d->renderSize.height;
    p.upscale[0]  = upscale.width;
    p.upscale[1]  = upscale.height;
    p.reset       = d->reset;
    p.flags       = u->flags;
    u->capture->Record(cl, u->frame, images, sizeof(images) / sizeof(images[0]), p);
}

ffxReturnCode_t DispatchUpscale(Upscaler* u, const ffxDispatchDescUpscale* d)
{
    if (!d->commandList || !d->color.resource || !d->depth.resource || !d->motionVectors.resource || !d->output.resource)
        return FFX_API_RETURN_ERROR_PARAMETER;
    if ((d->flags & (FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB | FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_PQ)) && !u->warnedColour)
    {
        Say(FFX_API_MESSAGE_TYPE_WARNING, "non-linear colour input is treated as linear");
        u->warnedColour = true;
    }
    const FfxApiDimensions2D render  = d->renderSize;
    const FfxApiDimensions2D upscale = d->upscaleSize.width ? d->upscaleSize : u->maxUpscale;
    if (!u->nss || render.width != u->nssRender.width || render.height != u->nssRender.height || upscale.width != u->nssUpscale.width ||
        upscale.height != u->nssUpscale.height)
        if (CreateNss(u, render, upscale) != FFX_OK)
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    FreeRetired(u, false);
    if (u->capture)
        u->capture->Write(Completed(u));

    auto* cl = static_cast<ID3D12GraphicsCommandList*>(d->commandList);
    FfxNssDx12Exposure e;
    e.texture      = static_cast<ID3D12Resource*>(d->exposure.resource);
    e.textureState = D3dState(d->exposure.state);
    if (!e.texture && (u->flags & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE))
    {
        e.colour      = static_cast<ID3D12Resource*>(d->color.resource);
        e.colourState = D3dState(d->color.state);
    }
    e.width       = render.width;
    e.height      = render.height;
    e.preExposure = d->preExposure > 0 ? d->preExposure : 1.0f;
    if (ffxNssDx12PrepareExposure(&u->iface, cl, e) != FFX_OK)
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;

    FfxResource      motion     = Resource(d->motionVectors, "motion");
    FfxFloatCoords2D mvScale    = {d->motionVectorScale.x, d->motionVectorScale.y};
    const bool       displayMv  = u->flags & FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS;
    const bool       jitteredMv = u->flags & FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION;
    if (displayMv || jitteredMv)
    {
        FfxNssDx12Motion m;
        m.source       = static_cast<ID3D12Resource*>(d->motionVectors.resource);
        m.sourceState  = D3dState(d->motionVectors.state);
        m.target       = Internal(u, u->motion, render.width, render.height, DXGI_FORMAT_R32G32_FLOAT,
                                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        m.renderWidth  = render.width;
        m.renderHeight = render.height;
        m.sourceWidth  = displayMv ? upscale.width : 0;
        m.sourceHeight = displayMv ? upscale.height : 0;
        m.scale[0]     = d->motionVectorScale.x;
        m.scale[1]     = d->motionVectorScale.y;
        m.jitter[0]    = d->jitterOffset.x;
        m.jitter[1]    = d->jitterOffset.y;
        if (jitteredMv)
        {
            m.cancel[0] = u->lastJitter.x - d->jitterOffset.x;
            m.cancel[1] = u->lastJitter.y - d->jitterOffset.y;
        }
        if (!m.target)
            return FFX_API_RETURN_ERROR_MEMORY;
        if (ffxNssDx12PrepareMotion(&u->iface, cl, m) != FFX_OK)
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        motion  = ffxGetResourceDX12(m.target, FFX_RESOURCE_STATE_COMPUTE_READ, "rdnu_motion");
        mvScale = {1.0f, 1.0f};
    }
    u->lastJitter = d->jitterOffset;

    const bool      sharpen = d->enableSharpening && d->sharpness > 0 && !(d->flags & FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW);
    ID3D12Resource* output  = static_cast<ID3D12Resource*>(d->output.resource);
    ID3D12Resource* target  = sharpen ? Internal(u, u->sharpenInput, upscale.width, upscale.height, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
                                      : output;
    if (!target)
        return FFX_API_RETURN_ERROR_MEMORY;

    FfxNssDispatchDescription n{};
    n.commandList            = ffxGetCommandListDX12(cl);
    n.color                  = Resource(d->color, "color");
    n.depth                  = Resource(d->depth, "depth");
    n.motionVectors          = motion;
    n.output                 = sharpen ? ffxGetResourceDX12(target, FFX_RESOURCE_STATE_GENERIC_UAV, "rdnu_sharpen_input")
                                       : Resource(d->output, "output");
    // NSS's jitter is FSR's negated: on the same frames each loses ~4 dB with the other's sign
    n.jitterOffset           = {-d->jitterOffset.x, -d->jitterOffset.y};
    n.motionVectorScale      = mvScale;
    n.renderSize             = {render.width, render.height};
    n.upscaleSize            = {upscale.width, upscale.height};
    n.cameraNear             = d->cameraNear;
    n.cameraFar              = d->cameraFar;
    n.cameraFovAngleVertical = d->cameraFovAngleVertical;
    n.exposure               = 1.0f / e.preExposure;  // replaced on the GPU by the prepared exposure
    n.frameTimeDelta         = d->frameTimeDelta;
    n.reset                  = d->reset || u->fresh;
    if (d->flags & FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW)
    {
        n.flags      = FFX_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW;
        n.debugViews = n.output;
    }
    if (ffxNssContextDispatch(u->nss.get(), &n) != FFX_OK)
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    if (sharpen && ffxNssDx12Sharpen(&u->iface, cl, target, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, output, D3dState(d->output.state),
                                     upscale.width, upscale.height, d->sharpness) != FFX_OK)
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    if (u->capture)
        Record(u, cl, d, upscale);
    u->fresh = false;
    ++u->frame;
    ID3D12GraphicsCommandList2* cl2 = nullptr;
    if (u->marker && SUCCEEDED(cl->QueryInterface(IID_PPV_ARGS(&cl2))))
    {
        const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER w{u->marker->GetGPUVirtualAddress(), uint32_t(u->frame)};
        const D3D12_WRITEBUFFERIMMEDIATE_MODE      m = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
        cl2->WriteBufferImmediate(1, &w, &m);
        cl2->Release();
    }
    return FFX_API_RETURN_OK;
}

void CreateMarker(Upscaler* u)
{
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC d{};
    d.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width            = 256;
    d.Height           = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels        = 1;
    d.SampleDesc.Count = 1;
    d.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    void* p            = nullptr;
    if (FAILED(u->device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                  IID_PPV_ARGS(&u->marker))))
        return;
    if (FAILED(u->marker->Map(0, nullptr, &p)))
    {
        u->marker->Release();
        u->marker = nullptr;
        return;
    }
    *static_cast<volatile uint32_t*>(p) = 0;
    u->done = static_cast<const volatile uint32_t*>(p);
}

void Teardown(Upscaler* u)
{
    u->capture.reset();
    if (u->reference)
        LoadOriginal()->destroy(&u->reference, nullptr);
    Retire(u);
    FreeRetired(u, true);
    for (ID3D12Resource* r : {u->sharpenInput, u->motion, u->referenceOutput})
        if (r)
            r->Release();
    if (u->marker)
        u->marker->Release();
    if (u->iface.scratchBuffer)
        ffxReleaseInterfaceDX12(&u->iface);
}

// AMD's upscaler named by RDNU_COMPARE, through the original DLL, for captures.
void CreateReference(Upscaler* u, const ffxCreateContextDescUpscale* up, const ffxCreateBackendDX12Desc* be)
{
    const char*     want = std::getenv("RDNU_COMPARE");
    const Original* o    = LoadOriginal();
    if (!want || !o)
        return;
    uint64_t                count = 16;
    uint64_t                ids[16];
    const char*             names[16];
    ffxQueryDescGetVersions q{};
    q.header.type    = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    q.device         = u->device;
    q.outputCount    = &count;
    q.versionIds     = ids;
    q.versionNames   = names;
    if (o->query(nullptr, &q.header) != FFX_API_RETURN_OK)
        return;
    for (uint64_t i = 0; i < count; ++i)
        if (names[i] && std::strstr(names[i], want))
        {
            ffxCreateContextDescUpscale c = *up;
            ffxCreateBackendDX12Desc    b = *be;
            ffxOverrideVersion          v{};
            v.header.type                 = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
            v.versionId                   = ids[i];
            c.header.pNext                = &b.header;
            b.header.pNext                = &v.header;
            if (o->create(&u->reference, &c.header, nullptr) == FFX_API_RETURN_OK)
                Say(FFX_API_MESSAGE_TYPE_WARNING, std::string("capturing AMD ") + names[i] + " as the reference");
            return;
        }
}

ffxReturnCode_t CreateUpscaler(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* mem)
{
    auto* up = reinterpret_cast<const ffxCreateContextDescUpscale*>(Find(desc, FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE));
    auto* be = reinterpret_cast<const ffxCreateBackendDX12Desc*>(Find(desc, FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12));
    if (!up || !be || !be->device || !up->maxRenderSize.width || !up->maxUpscaleSize.width)
        return FFX_API_RETURN_ERROR_PARAMETER;
    if (up->fpMessage)
        g_message = up->fpMessage;

    void* memory = mem && mem->alloc ? mem->alloc(mem->pUserData, sizeof(Upscaler)) : ::operator new(sizeof(Upscaler), std::nothrow);
    if (!memory)
        return FFX_API_RETURN_ERROR_MEMORY;
    auto* u       = new (memory) Upscaler();
    u->device     = be->device;
    u->flags      = up->flags;
    u->maxRender  = up->maxRenderSize;
    u->maxUpscale = up->maxUpscaleSize;
    const char* dp4a = std::getenv("RDNU_FORCE_DP4A");
    ffxNssDx12ForceDp4a(dp4a && *dp4a && *dp4a != '0');
    u->scratch.resize(ffxGetScratchMemorySizeDX12(kMaxNss));
    CreateMarker(u);
    FfxErrorCode e = ffxGetInterfaceDX12(&u->iface, u->device, u->scratch.data(), u->scratch.size(), kMaxNss);
    if (e == FFX_OK)
    {
        u->iface.fpSetMessageCallback(&u->iface, BackendMessage);
        std::vector<std::string> warnings;
        rdnu::NssFlagsFromUpscale(u->flags, warnings);
        for (const std::string& w : warnings)
            Say(FFX_API_MESSAGE_TYPE_WARNING, w);
        // validates the device and builds every pipeline before the first frame
        e = CreateNss(u, u->maxRender, u->maxUpscale);
    }
    if (e == FFX_OK && (u->capture = rdnu::Capture::FromEnvironment(u->device)))
        CreateReference(u, up, be);
    if (e != FFX_OK)
    {
        Say(FFX_API_MESSAGE_TYPE_ERROR, "cannot run NSS on this device (error " + std::to_string(e) + ")");
        Teardown(u);
        u->~Upscaler();
        if (mem && mem->dealloc)
            mem->dealloc(mem->pUserData, memory);
        else
            ::operator delete(memory);
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    FfxDeviceCapabilities caps{};
    u->iface.fpGetDeviceCapabilities(&u->iface, &caps);
    Say(FFX_API_MESSAGE_TYPE_WARNING, "context " + std::to_string(u->maxRender.width) + "x" + std::to_string(u->maxRender.height) + " -> " +
                                          std::to_string(u->maxUpscale.width) + "x" + std::to_string(u->maxUpscale.height) + ", flags " +
                                          std::to_string(u->flags) + ", shader model 6." + std::to_string(int(caps.maximumSupportedShaderModel) - 1) +
                                          (caps.fp16Supported ? "" : ", no fp16") + (ffxNssDx12UsesWmma(&u->iface) ? ", WMMA" : ", DP4a"));
    std::lock_guard<std::mutex> l(g_lock);
    g_ours.insert(u);
    *context = u;
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t QueryUpscaler(Upscaler* u, ffxQueryDescHeader* desc)
{
    switch (desc->type)
    {
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE:
    {
        auto* q = reinterpret_cast<ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode*>(desc);
        float r = rdnu::UpscaleRatio(q->qualityMode);
        if (!r || !q->pOutUpscaleRatio)
            return FFX_API_RETURN_ERROR_PARAMETER;
        *q->pOutUpscaleRatio = r;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE:
    {
        auto*    q = reinterpret_cast<ffxQueryDescUpscaleGetRenderResolutionFromQualityMode*>(desc);
        uint32_t w = 0, h = 0;
        if (!rdnu::RenderResolution(q->displayWidth, q->displayHeight, q->qualityMode, w, h))
            return FFX_API_RETURN_ERROR_PARAMETER;
        if (q->pOutRenderWidth)
            *q->pOutRenderWidth = w;
        if (q->pOutRenderHeight)
            *q->pOutRenderHeight = h;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT:
    {
        auto* q = reinterpret_cast<ffxQueryDescUpscaleGetJitterPhaseCount*>(desc);
        if (!q->pOutPhaseCount || !q->renderWidth)
            return FFX_API_RETURN_ERROR_PARAMETER;
        *q->pOutPhaseCount = ffxNssGetJitterPhaseCount(int32_t(q->renderWidth), int32_t(q->displayWidth));
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET:
    {
        auto* q = reinterpret_cast<ffxQueryDescUpscaleGetJitterOffset*>(desc);
        float x = 0, y = 0;
        if (ffxNssGetJitterOffset(&x, &y, q->index, q->phaseCount) != FFX_OK)
            return FFX_API_RETURN_ERROR_PARAMETER;
        if (q->pOutX)
            *q->pOutX = x;
        if (q->pOutY)
            *q->pOutY = y;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE:
    {
        auto* q = reinterpret_cast<ffxQueryDescUpscaleGetGPUMemoryUsage*>(desc);
        if (!u || !q->gpuMemoryUsageUpscaler)
            return FFX_API_RETURN_ERROR_PARAMETER;
        *q->gpuMemoryUsageUpscaler = {};
        for (FfxUInt32 id = 0; id < kMaxNss; ++id)
        {
            FfxEffectMemoryUsage m{};
            if (u->iface.fpGetEffectGpuMemoryUsage(&u->iface, id, &m) == FFX_OK)
                q->gpuMemoryUsageUpscaler->totalUsageInBytes += m.totalUsageInBytes;
        }
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_RESOURCE_REQUIREMENTS:
    {
        auto* q               = reinterpret_cast<ffxQueryDescUpscaleGetResourceRequirements*>(desc);
        q->required_resources = FFX_API_QUERY_RESOURCE_INPUT_COLOR | FFX_API_QUERY_RESOURCE_INPUT_DEPTH | FFX_API_QUERY_RESOURCE_INPUT_MV;
        q->optional_resources = FFX_API_QUERY_RESOURCE_INPUT_EXPOSURE;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION:
    {
        auto* q        = reinterpret_cast<ffxQueryGetProviderVersion*>(desc);
        q->versionId   = kVersionId;
        q->versionName = kVersion;
        return FFX_API_RETURN_OK;
    }
    default: return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
}

// Our version first, then whatever AMD's DLL offers.
ffxReturnCode_t GetVersions(ffxQueryDescGetVersions* q)
{
    const Original* o = LoadOriginal();
    uint64_t        theirs = 0;
    if (o)
    {
        ffxQueryDescGetVersions c = *q;
        c.outputCount             = &theirs;
        c.versionIds              = nullptr;
        c.versionNames            = nullptr;
        if (o->query(nullptr, &c.header) != FFX_API_RETURN_OK)
            theirs = 0;
    }
    const uint64_t capacity = *q->outputCount;
    const uint64_t total    = 1 + theirs;
    if (!q->versionIds && !q->versionNames)
    {
        *q->outputCount = total;
        return FFX_API_RETURN_OK;
    }
    if (capacity == 0)
    {
        *q->outputCount = 0;
        return FFX_API_RETURN_OK;
    }
    if (q->versionIds)
        q->versionIds[0] = kVersionId;
    if (q->versionNames)
        q->versionNames[0] = kVersion;
    uint64_t written = 1;
    if (o && theirs && capacity > 1)
    {
        ffxQueryDescGetVersions c = *q;
        uint64_t                n = capacity - 1;
        c.outputCount             = &n;
        c.versionIds              = q->versionIds ? q->versionIds + 1 : nullptr;
        c.versionNames            = q->versionNames ? q->versionNames + 1 : nullptr;
        if (o->query(nullptr, &c.header) == FFX_API_RETURN_OK)
            written += n;
    }
    *q->outputCount = written;
    return FFX_API_RETURN_OK;
}
ffxReturnCode_t DestroyUpscaler(ffxContext* context, const ffxAllocationCallbacks* mem)
{
    auto* u = static_cast<Upscaler*>(*context);
    {
        std::lock_guard<std::mutex> l(g_lock);
        g_ours.erase(u);
    }
    Teardown(u);
    u->~Upscaler();
    if (mem && mem->dealloc)
        mem->dealloc(mem->pUserData, u);
    else
        ::operator delete(u);
    *context = nullptr;
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t ConfigureUpscaler(const ffxConfigureDescHeader* desc)
{
    if (desc->type == FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1)
        g_message = reinterpret_cast<const ffxConfigureDescGlobalDebug1*>(desc)->fpMessage;
    return desc->type == FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE || desc->type == FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1
               ? FFX_API_RETURN_OK  // FSR tuning keys have no NSS counterpart
               : FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
}

ffxReturnCode_t DispatchUpscaler(Upscaler* u, const ffxDispatchDescHeader* desc)
{
    switch (desc->type)
    {
    case FFX_API_DISPATCH_DESC_TYPE_UPSCALE: return DispatchUpscale(u, reinterpret_cast<const ffxDispatchDescUpscale*>(desc));
    case FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK: return FFX_API_RETURN_OK;  // NSS reads no reactive mask
    default: return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
}

const ProviderVtable g_vtable = {
    {},
    [](const void*, uint64_t type) { return type == FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE; },
    [](const void*, void*) { return true; },
    [](const void*) { return kVersionId; },
    [](const void*) { return kVersion; },
    [](const void*, ffxContext* c, ffxCreateContextDescHeader* d, Allocator& a) { return CreateUpscaler(c, d, a.cb); },
    [](const void*, ffxContext* c, Allocator& a) { return DestroyUpscaler(c, a.cb); },
    [](const void*, ffxContext*, const ffxConfigureDescHeader* d) { return ConfigureUpscaler(d); },
    [](const void*, ffxContext* c, ffxQueryDescHeader* d) { return QueryUpscaler(static_cast<Upscaler*>(*c), d); },
    [](const void*, ffxContext* c, const ffxDispatchDescHeader* d) { return DispatchUpscaler(static_cast<Upscaler*>(*c), d); },
};

const Provider g_provider{&g_vtable};
}  // namespace

FFX_API_ENTRY ffxReturnCode_t ffxCreateContext(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* memCb)
{
    if (!context || !desc)
        return FFX_API_RETURN_ERROR_PARAMETER;
    const auto* over  = reinterpret_cast<const ffxOverrideVersion*>(Find(desc, FFX_API_DESC_TYPE_OVERRIDE_VERSION));
    const bool  mine  = Find(desc, FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE) && (!over || over->versionId == kVersionId);
    const Original* o = LoadOriginal();
    if (mine)
    {
        ffxReturnCode_t r = CreateUpscaler(context, desc, memCb);
        if (r == FFX_API_RETURN_OK || !o)
            return r;
        Say(FFX_API_MESSAGE_TYPE_WARNING, "falling back to AMD's upscaler");
    }
    return o ? o->create(context, desc, memCb) : ffxReturnCode_t(FFX_API_RETURN_NO_PROVIDER);
}

FFX_API_ENTRY ffxReturnCode_t ffxDestroyContext(ffxContext* context, const ffxAllocationCallbacks* memCb)
{
    if (!context)
        return FFX_API_RETURN_ERROR_PARAMETER;
    if (Ours(*context))
        return DestroyUpscaler(context, memCb);
    const Original* o = LoadOriginal();
    return o ? o->destroy(context, memCb) : ffxReturnCode_t(FFX_API_RETURN_ERROR_PARAMETER);
}

FFX_API_ENTRY ffxReturnCode_t ffxConfigure(ffxContext* context, const ffxConfigureDescHeader* desc)
{
    if (!desc)
        return FFX_API_RETURN_ERROR_PARAMETER;
    if (context && Ours(*context))
        return ConfigureUpscaler(desc);
    if (desc->type == FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1)
        g_message = reinterpret_cast<const ffxConfigureDescGlobalDebug1*>(desc)->fpMessage;
    const Original* o = LoadOriginal();
    if (o)
        return o->configure(context, desc);
    return desc->type == FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1 ? FFX_API_RETURN_OK : FFX_API_RETURN_NO_PROVIDER;
}

FFX_API_ENTRY ffxReturnCode_t ffxQuery(ffxContext* context, ffxQueryDescHeader* desc)
{
    if (!desc)
        return FFX_API_RETURN_ERROR_PARAMETER;
    if (desc->type == FFX_API_QUERY_DESC_TYPE_GET_VERSIONS)
    {
        auto* q = reinterpret_cast<ffxQueryDescGetVersions*>(desc);
        if (q->createDescType == FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE && q->outputCount)
            return GetVersions(q);
    }
    else if ((desc->type & FFX_API_EFFECT_MASK) == FFX_API_EFFECT_ID_UPSCALE || desc->type == FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION)
    {
        if (context && Ours(*context))
            return QueryUpscaler(static_cast<Upscaler*>(*context), desc);
        const auto* over = reinterpret_cast<const ffxOverrideVersion*>(Find(desc, FFX_API_DESC_TYPE_OVERRIDE_VERSION));
        if ((!context || !*context) && (!over || over->versionId == kVersionId))
            return QueryUpscaler(nullptr, desc);  // context-free upscale queries
    }
    const Original* o = LoadOriginal();
    return o ? o->query(context, desc) : ffxReturnCode_t(FFX_API_RETURN_NO_PROVIDER);
}

FFX_API_ENTRY ffxReturnCode_t ffxDispatch(ffxContext* context, const ffxDispatchDescHeader* desc)
{
    if (!context || !desc)
        return FFX_API_RETURN_ERROR_PARAMETER;
    if (Ours(*context))
        return DispatchUpscaler(static_cast<Upscaler*>(*context), desc);
    const Original* o = LoadOriginal();
    return o ? o->dispatch(context, desc) : ffxReturnCode_t(FFX_API_RETURN_ERROR_PARAMETER);
}
