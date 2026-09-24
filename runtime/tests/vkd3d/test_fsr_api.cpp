// test_fsr_api.cpp - drives the RDNU FidelityFX API (runtime/src/provider) the way an FSR 3.1
// game does: context-free queries, create with the DX12 backend desc, per-frame dispatch, and
// destroy. Runs each exposure mode, sharpening, jittered and display resolution motion vectors
// and a render size change on real frames, and checks they agree with each other where FSR's
// definitions say they must. One run goes through the provider object the way FidelityFX SDK
// 2's loader does (api/internal/ffx_provider.h).
//
//   test_fsr_api <golden dir> [--frames N] [--dll file]
//
// On Windows --dll drives a DLL instead of the linked provider: amd_fidelityfx_dx12.dll, or
// AMD's amd_fidelityfx_loader_dx12.dll with RDNU as amd_fidelityfx_upscaler_dx12.dll beside it.
// <golden dir>/nss_frames.rdnut comes from runtime/tools/nss_pass_frames.py. On vkd3d-proton
// set VKD3D_SHADER_MODEL=6_6 (see runtime/tools/setup_vkd3d_proton.sh).
#include "../common/rdnut.h"
#include "d3d12_util.h"

#include <cstdlib>

#include <dx12/ffx_api_dx12.h>
#include <ffx_api.h>
#include <ffx_upscale.h>
#ifndef _WIN32
#include <ffx_provider.h>  // MSVC's layout on Windows: the real loader tests it there
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace
{
int g_failures = 0;

#define CHECK(cond, ...)                                        \
    do                                                          \
    {                                                           \
        if (!(cond))                                            \
        {                                                       \
            ++g_failures;                                       \
            std::printf("  FAIL %s:%d: ", __FILE__, __LINE__);  \
            std::printf(__VA_ARGS__);                           \
            std::printf("\n");                                  \
        }                                                       \
    } while (0)

struct Api
{
    PfnFfxCreateContext  create    = ffxCreateContext;
    PfnFfxDestroyContext destroy   = ffxDestroyContext;
    PfnFfxConfigure      configure = ffxConfigure;
    PfnFfxQuery          query     = ffxQuery;
    PfnFfxDispatch       dispatch  = ffxDispatch;
} g_api;

#ifndef _WIN32
// What AMD's SDK 2 loader does after creation: call the provider object the context starts with.
const ffxProvider* ProviderOf(ffxContext* c)
{
    return reinterpret_cast<const InternalContextHeader*>(*c)->provider;
}

const Api kViaProvider = {
    ffxCreateContext,
    [](ffxContext* c, const ffxAllocationCallbacks* m) {
        Allocator a{m};
        return ProviderOf(c)->DestroyContext(c, a);
    },
    [](ffxContext* c, const ffxConfigureDescHeader* d) { return ProviderOf(c)->Configure(c, d); },
    [](ffxContext* c, ffxQueryDescHeader* d) { return ProviderOf(c)->Query(c, d); },
    [](ffxContext* c, const ffxDispatchDescHeader* d) { return ProviderOf(c)->Dispatch(c, d); },
};
#endif

#ifdef _WIN32
template <typename T>
bool Proc(HMODULE m, const char* name, T& out)
{
    out = reinterpret_cast<T>(reinterpret_cast<void (*)()>(GetProcAddress(m, name)));
    return out != nullptr;
}

bool LoadApi(const char* dll)
{
    HMODULE m = LoadLibraryA(dll);
    return m && Proc(m, "ffxCreateContext", g_api.create) && Proc(m, "ffxDestroyContext", g_api.destroy) &&
           Proc(m, "ffxConfigure", g_api.configure) && Proc(m, "ffxQuery", g_api.query) && Proc(m, "ffxDispatch", g_api.dispatch);
}
#endif

void Message(uint32_t type, const wchar_t* m)
{
    if (type == FFX_API_MESSAGE_TYPE_ERROR)
        std::fprintf(stderr, "ffx error: %ls\n", m);
}

enum class Exposure { Texture, None, Auto };

struct Options
{
    Exposure exposure   = Exposure::Texture;
    float    pre        = 1.0f;   // preExposure; colour is scaled by it
    float    sharpness  = 0.0f;
    bool     jitteredMv = false;  // vectors carry the jitter delta (JITTER_CANCELLATION)
    bool     displayMv  = false;  // vectors at display resolution
    uint32_t smallFrom  = ~0u;    // first frame rendered at the small size
    uint32_t first      = 0;
    bool     viaLoader  = false;  // everything after creation through the context's provider object (not on Windows)
    uint64_t version    = 0;      // another provider through FFX_API_DESC_TYPE_OVERRIDE_VERSION
};

struct Frames
{
    const rdnut::Tensor *colour, *depth, *motion, *motionHr, *jitter, *exposure, *camera, *truth;
    uint32_t W, H, count;
};

struct Result
{
    std::vector<std::vector<float>> out;  // per frame, RGB at the frame's display size
    std::vector<double>             psnr;
    bool                            ok = false;
};

// Tonemapped PSNR of a against b (w x h RGB), both scaled by the exposure.
double Psnr(const float* a, const float* b, size_t n, float ex)
{
    double se = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const double x = a[i] * ex, y = b[i] * ex, d = x / (1 + x) - y / (1 + y);
        se += d * d;
    }
    return 10 * std::log10(1.0 / std::max(1e-20, se / double(n)));
}

Result Run(Gpu& gpu, const Frames& f, const Options& o)
{
    Result         r;
    const uint32_t W = f.W, H = f.H, DW = 2 * W, DH = 2 * H;
    const uint32_t SW = W - 20, SH = H - 20;  // the smaller size, cropped from the top left

    ffxCreateContextDescUpscale cu{};
    cu.header.type    = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    cu.flags          = (f.camera->data[3] != 0 ? FFX_UPSCALE_ENABLE_DEPTH_INFINITE : 0) |
               (o.exposure == Exposure::Auto ? FFX_UPSCALE_ENABLE_AUTO_EXPOSURE : 0) |
               (o.jitteredMv ? FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION : 0) |
               (o.displayMv ? FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS : 0);
    cu.maxRenderSize  = {W, H};
    cu.maxUpscaleSize = {DW, DH};
    cu.fpMessage      = Message;
    ffxCreateBackendDX12Desc be{};
    be.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    be.device      = gpu.device;
    cu.header.pNext = &be.header;
    ffxOverrideVersion ov{};
    ov.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    ov.versionId   = o.version;
    if (o.version)
        be.header.pNext = &ov.header;
    ffxContext ctx  = nullptr;
#ifdef _WIN32
    const Api& api = g_api;
#else
    const Api& api = o.viaLoader ? kViaProvider : g_api;
#endif
    if (api.create(&ctx, &cu.header, nullptr) != FFX_API_RETURN_OK)
        return std::printf("  ffxCreateContext failed\n"), r;

    const D3D12_RESOURCE_STATES read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    ID3D12Resource* tColour  = gpu.Texture(W, H, DXGI_FORMAT_R16G16B16A16_FLOAT, false, read);
    ID3D12Resource* tDepth   = gpu.Texture(W, H, DXGI_FORMAT_R32_TYPELESS, false, read);
    ID3D12Resource* tMotion  = o.displayMv ? gpu.Texture(DW, DH, DXGI_FORMAT_R32G32_FLOAT, false, read)
                                           : gpu.Texture(W, H, DXGI_FORMAT_R32G32_FLOAT, false, read);
    ID3D12Resource* tExp     = gpu.Texture(1, 1, DXGI_FORMAT_R32_FLOAT, false, read);
    ID3D12Resource* tOutput  = gpu.Texture(DW, DH, DXGI_FORMAT_R16G16B16A16_FLOAT, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12Resource* tSmall[] = {gpu.Texture(SW, SH, DXGI_FORMAT_R16G16B16A16_FLOAT, false, read),
                                gpu.Texture(SW, SH, DXGI_FORMAT_R32_TYPELESS, false, read),
                                gpu.Texture(SW, SH, DXGI_FORMAT_R32G32_FLOAT, false, read)};

#ifndef _WIN32
    if (o.viaLoader)
        CHECK(ProviderOf(&ctx)->CanProvide(FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE) &&
                  !std::strncmp(ProviderOf(&ctx)->GetVersionName(), "RDNU", 4),
              "provider object");
#endif

    const size_t np = size_t(W) * H;
    r.ok            = true;
    for (uint32_t t = o.first; t < f.count && r.ok; ++t)
    {
        const bool     small = t >= o.smallFrom;
        const uint32_t rw = small ? SW : W, rh = small ? SH : H, dw = 2 * rw, dh = 2 * rh;
        const float    jx = f.jitter->data[t * 2], jy = f.jitter->data[t * 2 + 1];
        const float    pjx = t ? f.jitter->data[t * 2 - 2] : 0, pjy = t ? f.jitter->data[t * 2 - 1] : 0;
        const float    ex  = f.exposure->data[t];

        std::vector<uint16_t> col(size_t(rw) * rh * 4);
        std::vector<float>    dep(size_t(rw) * rh), mv;
        for (uint32_t y = 0; y < rh; ++y)
            for (uint32_t x = 0; x < rw; ++x)
            {
                const size_t s = t * np + size_t(y) * W + x, d = size_t(y) * rw + x;
                for (int k = 0; k < 4; ++k)
                    col[d * 4 + k] = ToHalf(f.colour->data[s * 4 + k] * (k < 3 ? o.pre : 1.0f));
                dep[d] = f.depth->data[s];
            }
        if (o.displayMv)
            mv.assign(f.motionHr->data.begin() + long(t * size_t(DW) * DH * 2), f.motionHr->data.begin() + long((t + 1) * size_t(DW) * DH * 2));
        else
        {
            mv.resize(size_t(rw) * rh * 2);
            for (uint32_t y = 0; y < rh; ++y)
                for (uint32_t x = 0; x < rw; ++x)
                    for (int k = 0; k < 2; ++k)
                        // FSR 3 cancels (previous - current jitter) from jittered vectors
                        mv[(size_t(y) * rw + x) * 2 + k] =
                            f.motion->data[(t * np + size_t(y) * W + x) * 2 + k] + (o.jitteredMv ? (k ? pjy - jy : pjx - jx) : 0.0f);
        }
        const float expTex = ex;  // FSR divides colour by preExposure, then applies the texture
        ID3D12Resource* c = small ? tSmall[0] : tColour;
        ID3D12Resource* d = small ? tSmall[1] : tDepth;
        ID3D12Resource* m = small ? tSmall[2] : tMotion;
        ID3D12Resource* ups[] = {gpu.Upload(c, col.data(), 8, read), gpu.Upload(d, dep.data(), 4, read), gpu.Upload(m, mv.data(), 8, read),
                                 gpu.Upload(tExp, &expTex, 4, read)};

        ffxDispatchDescUpscale dd{};
        dd.header.type            = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        dd.commandList            = gpu.list;
        dd.color                  = ffxApiGetResourceDX12(c, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dd.depth                  = ffxApiGetResourceDX12(d, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dd.motionVectors          = ffxApiGetResourceDX12(m, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dd.exposure               = o.exposure == Exposure::Texture ? ffxApiGetResourceDX12(tExp, FFX_API_RESOURCE_STATE_COMPUTE_READ) : FfxApiResource{};
        dd.output                 = ffxApiGetResourceDX12(tOutput, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        dd.jitterOffset           = {jx, jy};
        dd.motionVectorScale      = {1, 1};
        dd.renderSize             = {rw, rh};
        dd.upscaleSize            = {dw, dh};
        dd.enableSharpening       = o.sharpness > 0;
        dd.sharpness              = o.sharpness;
        dd.frameTimeDelta         = 16.7f;
        dd.preExposure            = o.exposure == Exposure::None ? 1.0f / ex : o.pre;
        dd.reset                  = t == o.first;
        dd.cameraNear             = f.camera->data[t * 4];
        dd.cameraFar              = f.camera->data[t * 4 + 1];
        // the crop keeps the pixel pitch, so its vertical field of view shrinks
        dd.cameraFovAngleVertical = 2 * std::atan(std::tan(f.camera->data[t * 4 + 2] / 2) * float(rh) / float(H));
        if (api.dispatch(&ctx, &dd.header) != FFX_API_RETURN_OK)
        {
            std::printf("  ffxDispatch failed at frame %u\n", t);
            r.ok = false;
            break;
        }
        gpu.Submit();
        for (ID3D12Resource* u : ups)
            u->Release();

        const std::vector<uint8_t> o16 = gpu.Read(tOutput, 8, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        std::vector<float>         out(size_t(dw) * dh * 3), gt(out.size());
        for (uint32_t y = 0; y < dh; ++y)
            for (uint32_t x = 0; x < dw; ++x)
                for (int k = 0; k < 3; ++k)
                {
                    uint16_t h;
                    std::memcpy(&h, &o16[(size_t(y) * DW + x) * 8 + k * 2], 2);
                    out[(size_t(y) * dw + x) * 3 + k] = FromHalf(h) / o.pre;
                    gt[(size_t(y) * dw + x) * 3 + k]  = f.truth->data[((t * size_t(DH) + y) * DW + x) * 4 + k];
                }
        r.psnr.push_back(Psnr(out.data(), gt.data(), out.size(), ex));
        r.out.push_back(std::move(out));
    }

    if (r.ok)
    {
        FfxApiEffectMemoryUsage             mu{};
        ffxQueryDescUpscaleGetGPUMemoryUsage qm{};
        qm.header.type            = FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE;
        qm.gpuMemoryUsageUpscaler = &mu;
        CHECK(api.query(&ctx, &qm.header) == FFX_API_RETURN_OK && mu.totalUsageInBytes > 0, "GPU memory query");
    }
    CHECK(api.destroy(&ctx, nullptr) == FFX_API_RETURN_OK, "destroy");
    for (ID3D12Resource* x : {tColour, tDepth, tMotion, tExp, tOutput, tSmall[0], tSmall[1], tSmall[2]})
        x->Release();
    return r;
}

void Print(const char* name, const Result& r, uint32_t first)
{
    std::printf("  %-28s", name);
    for (size_t i = 0; i < r.psnr.size(); ++i)
        std::printf(" %5.2f", r.psnr[i]);
    std::printf("%s\n", first ? "  (from frame 4)" : "");
}

// Agreement of two runs over the frames they share: the lowest per-frame PSNR of one against the other.
double Agreement(const Result& a, const Result& b, size_t offsetA, const Frames& f)
{
    double worst = 1e9;
    for (size_t i = 0; i < b.out.size() && i + offsetA < a.out.size(); ++i)
    {
        const std::vector<float>& x = a.out[i + offsetA];
        const std::vector<float>& y = b.out[i];
        if (x.size() != y.size())
            return 0;
        worst = std::min(worst, Psnr(x.data(), y.data(), x.size(), f.exposure->data[i + offsetA]));
    }
    return worst;
}

// Returns the other providers the DLL offers (AMD's, forwarded), by id and name.
std::vector<std::pair<uint64_t, std::string>> Queries(Gpu& gpu, uint32_t W, uint32_t DW)
{
    uint64_t                count = 0;
    ffxQueryDescGetVersions qv{};
    qv.header.type    = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    qv.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    qv.device         = gpu.device;
    qv.outputCount    = &count;
    CHECK(g_api.query(nullptr, &qv.header) == FFX_API_RETURN_OK && count >= 1, "version count %llu", (unsigned long long)count);
    uint64_t    ids[4]   = {};
    const char* names[4] = {};
    count                = 4;
    qv.versionIds        = ids;
    qv.versionNames      = names;
    CHECK(g_api.query(nullptr, &qv.header) == FFX_API_RETURN_OK && count >= 1 && names[0] && !std::strncmp(names[0], "RDNU", 4),
          "versions");
    std::vector<std::pair<uint64_t, std::string>> others;
    for (uint64_t i = 0; i < count; ++i)
    {
        std::printf("  version %llu: %s (0x%016llx)\n", (unsigned long long)i, names[i] ? names[i] : "?", (unsigned long long)ids[i]);
        if (i)
            others.push_back({ids[i], names[i] ? names[i] : "?"});
    }

    float                                              ratio = 0;
    ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode qr{};
    qr.header.type      = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE;
    qr.qualityMode      = FFX_UPSCALE_QUALITY_MODE_PERFORMANCE;
    qr.pOutUpscaleRatio = &ratio;
    CHECK(g_api.query(nullptr, &qr.header) == FFX_API_RETURN_OK && ratio == 2.0f, "performance ratio %g", ratio);

    uint32_t                                              rw = 0, rh = 0;
    ffxQueryDescUpscaleGetRenderResolutionFromQualityMode qs{};
    qs.header.type      = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE;
    qs.displayWidth     = 3840;
    qs.displayHeight    = 2160;
    qs.qualityMode      = FFX_UPSCALE_QUALITY_MODE_QUALITY;
    qs.pOutRenderWidth  = &rw;
    qs.pOutRenderHeight = &rh;
    CHECK(g_api.query(nullptr, &qs.header) == FFX_API_RETURN_OK && rw == 2560 && rh == 1440, "quality at 4K: %ux%u", rw, rh);

    int32_t                                phases = 0;
    ffxQueryDescUpscaleGetJitterPhaseCount qp{};
    qp.header.type    = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT;
    qp.renderWidth    = W;
    qp.displayWidth   = DW;
    qp.pOutPhaseCount = &phases;
    CHECK(g_api.query(nullptr, &qp.header) == FFX_API_RETURN_OK && phases > 0, "jitter phases %d", phases);

    bool inside = true;
    for (int32_t i = 0; i < phases; ++i)
    {
        float                              x = 9, y = 9;
        ffxQueryDescUpscaleGetJitterOffset qj{};
        qj.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET;
        qj.index       = i;
        qj.phaseCount  = phases;
        qj.pOutX       = &x;
        qj.pOutY       = &y;
        inside = inside && g_api.query(nullptr, &qj.header) == FFX_API_RETURN_OK && std::fabs(x) <= 0.5f && std::fabs(y) <= 0.5f;
    }
    CHECK(inside, "jitter offsets within half a pixel");
    std::printf("  ratio %.1f, quality at 4K %ux%u, %d jitter phases\n", ratio, rw, rh, phases);
    return others;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
        return std::printf("usage: test_fsr_api <golden dir> [--frames N] [--dll file]\n"), 1;
    std::string dir = argv[1], err;
    uint32_t    frames = 8;
    for (int i = 2; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--frames") && i + 1 < argc)
            frames = uint32_t(std::atoi(argv[++i]));
#ifdef _WIN32
        else if (!std::strcmp(argv[i], "--dll") && i + 1 < argc && !LoadApi(argv[++i]))
            return std::printf("cannot load %s\n", argv[i]), 1;
#endif
    }
    std::map<std::string, rdnut::Tensor> seq;
    if (!rdnut::Load(dir + "/nss_frames.rdnut", seq, err))
        return std::printf("%s\n", err.c_str()), 1;
    Frames f{&seq.at("colour"), &seq.at("depth"), &seq.at("motion"), &seq.at("motion_hr"), &seq.at("jitter"), &seq.at("exposure"),
             &seq.at("camera"), &seq.at("truth"), 0, 0, 0};
    f.W     = f.colour->dims[2];
    f.H     = f.colour->dims[1];
    f.count = std::min(frames, f.colour->dims[0]);
    if (f.count < 6)
        return std::printf("need at least 6 frames\n"), 1;

    Gpu gpu;
    if (!gpu.Init())
        return std::printf("no D3D12 device\n"), 1;

    std::printf("queries\n");
    const auto others = Queries(gpu, f.W, 2 * f.W);

    std::printf("dispatch, tonemapped PSNR per frame (dB)\n");
    const uint32_t half = f.count / 2;
    Options        o;
    const Result   base = Run(gpu, f, o);
    Print("exposure texture", base, 0);
    o.viaLoader = true;
    const Result loader = Run(gpu, f, o);
#ifdef _WIN32
    Print("again, in a new context", loader, 0);
#else
    Print("through the SDK 2 provider", loader, 0);
#endif
    o = Options();
    o.pre = 4.0f;
    const Result pre = Run(gpu, f, o);
    Print("texture, preExposure 4", pre, 0);
    o = Options();
    o.exposure = Exposure::None;
    const Result none = Run(gpu, f, o);
    Print("1 / preExposure", none, 0);
    o.exposure = Exposure::Auto;
    const Result autoExp = Run(gpu, f, o);
    Print("auto exposure", autoExp, 0);
    o = Options();
    o.sharpness = 0.5f;
    const Result sharp = Run(gpu, f, o);
    Print("sharpening 0.5", sharp, 0);
    o = Options();
    o.jitteredMv = true;
    const Result jittered = Run(gpu, f, o);
    Print("jittered motion vectors", jittered, 0);
    o = Options();
    o.displayMv = true;
    const Result display = Run(gpu, f, o);
    Print("display resolution vectors", display, 0);
    o = Options();
    o.smallFrom = half;
    const Result resize = Run(gpu, f, o);
    Print("render size change", resize, 0);
    o.first = half;
    const Result fresh = Run(gpu, f, o);
    Print("small size only", fresh, half);
    for (const auto& v : others)
    {
        o         = Options();
        o.version = v.first;
        const Result amd = Run(gpu, f, o);
        Print(("AMD " + v.second + ", forwarded").c_str(), amd, 0);
        CHECK(amd.ok, "forwarding to %s", v.second.c_str());
    }

    for (const Result* r : {&base, &loader, &pre, &none, &autoExp, &sharp, &jittered, &display, &resize, &fresh})
        CHECK(r->ok && r->psnr.size() == (r == &fresh ? f.count - half : f.count), "run incomplete");
    if (!g_failures)
    {
        const double aPre = Agreement(base, pre, 0, f), aNone = Agreement(base, none, 0, f), aJit = Agreement(base, jittered, 0, f);
        const double aSharp = Agreement(base, sharp, 0, f), aResize = Agreement(resize, fresh, half, f);
        const double aLoader = Agreement(base, loader, 0, f);
        std::printf("agreement with the texture run (lowest frame PSNR, dB)\n");
        std::printf("  preExposure %.1f, 1/preExposure %.1f, jitter cancelled %.1f, sharpened %.1f\n", aPre, aNone, aJit, aSharp);
        std::printf("  size change against a fresh small context: %.1f, provider object: %.1f\n", aResize, aLoader);
        CHECK(aLoader >= 199, "the provider object path differs from the exports");
        CHECK(aPre > 50, "preExposure changes the result");
        CHECK(aNone > 50, "1 / preExposure differs from the same exposure as a texture");
        CHECK(aJit > 50, "jitter cancellation differs from unjittered vectors");
        CHECK(aSharp < 60 && aSharp > 25, "sharpening has no or too much effect");
        CHECK(aResize >= 199, "a context after a size change differs from a fresh one");
        double dBase = 0, dDisplay = 0, dAuto = 0;
        for (size_t i = 0; i < f.count; ++i)
            dBase += base.psnr[i], dDisplay += display.psnr[i], dAuto += autoExp.psnr[i];
        CHECK(std::fabs(dDisplay - dBase) / f.count < 0.5, "display resolution vectors: %.2f against %.2f dB", dDisplay / f.count,
              dBase / f.count);
        CHECK(dAuto / f.count > 25, "auto exposure: %.2f dB", dAuto / f.count);
    }
    std::printf("%s (%d failures)\n", g_failures ? "FAIL" : "PASS", g_failures);
    return g_failures ? 2 : 0;
}
