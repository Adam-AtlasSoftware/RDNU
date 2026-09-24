// rdnu_bench.cpp - GPU time of a whole upscale through the FidelityFX API, as a game pays it:
// exposure, the NSS passes, the network and optional RCAS, on synthetic inputs.
//
//   rdnu_bench [--render WxH] [--scale S] [--frames N] [--sharpen S]
//
// Defaults: 1920x1080 at 2x, 100 timed frames after 10 warm-up frames. RDNU_FORCE_DP4A=1 runs
// the DP4a kernels on RDNA3. Per-pass times: capture with RGP or PIX, every pass is a named event.
#include "../tests/vkd3d/d3d12_util.h"

#include <dx12/ffx_api_dx12.h>
#include <ffx_api.h>
#include <ffx_upscale.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv)
{
    uint32_t W = 1920, H = 1080, frames = 100;
    double   scale = 2, sharpen = 0;
    for (int i = 1; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--render") && i + 1 < argc)
            std::sscanf(argv[++i], "%ux%u", &W, &H);
        else if (!std::strcmp(argv[i], "--scale") && i + 1 < argc)
            scale = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc)
            frames = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--sharpen") && i + 1 < argc)
            sharpen = std::atof(argv[++i]);
        else
            return std::printf("usage: rdnu_bench [--render WxH] [--scale S] [--frames N] [--sharpen S]\n"), 1;
    }
    const uint32_t DW = uint32_t(std::lround(W * scale)), DH = uint32_t(std::lround(H * scale)), warmup = 10;

    Gpu gpu;
    if (!gpu.Init())
        return std::printf("no D3D12 device\n"), 1;
    ffxCreateContextDescUpscale cu{};
    cu.header.type    = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    cu.flags          = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE;
    cu.maxRenderSize  = {W, H};
    cu.maxUpscaleSize = {DW, DH};
    ffxCreateBackendDX12Desc be{};
    be.header.type  = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    be.device       = gpu.device;
    cu.header.pNext = &be.header;
    ffxContext ctx  = nullptr;
    if (ffxCreateContext(&ctx, &cu.header, nullptr) != FFX_API_RETURN_OK)
        return std::printf("ffxCreateContext failed\n"), 1;

    // smooth colour, a depth ramp and a slow pan: every pass does its full work
    const D3D12_RESOURCE_STATES read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    ID3D12Resource*             tColour = gpu.Texture(W, H, DXGI_FORMAT_R16G16B16A16_FLOAT, false, read);
    ID3D12Resource*             tDepth  = gpu.Texture(W, H, DXGI_FORMAT_R32_FLOAT, false, read);
    ID3D12Resource*             tMotion = gpu.Texture(W, H, DXGI_FORMAT_R16G16_FLOAT, false, read);
    ID3D12Resource*             tOutput = gpu.Texture(DW, DH, DXGI_FORMAT_R16G16B16A16_FLOAT, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    std::vector<uint16_t>       col(size_t(W) * H * 4), mv(size_t(W) * H * 2);
    std::vector<float>          dep(size_t(W) * H);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x)
        {
            const size_t i = size_t(y) * W + x;
            for (int c = 0; c < 3; ++c)
                col[i * 4 + c] = ToHalf(0.5f + 0.4f * std::sin(0.05f * x * (c + 1) + 0.07f * y));
            col[i * 4 + 3] = ToHalf(1);
            dep[i]         = 0.1f + 0.8f * float(y) / float(H);
            mv[i * 2]      = ToHalf(0.25f);
            mv[i * 2 + 1]  = ToHalf(0);
        }
    ID3D12Resource* ups[] = {gpu.Upload(tColour, col.data(), 8, read), gpu.Upload(tDepth, dep.data(), 4, read),
                             gpu.Upload(tMotion, mv.data(), 4, read)};
    gpu.Submit();
    for (ID3D12Resource* u : ups)
        u->Release();

    ID3D12QueryHeap*          heap = nullptr;
    const D3D12_QUERY_HEAP_DESC qd{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2 * (warmup + frames), 0};
    gpu.device->CreateQueryHeap(&qd, IID_PPV_ARGS(&heap));
    ID3D12Resource* result = gpu.Buffer(8 * 2 * (warmup + frames), D3D12_HEAP_TYPE_READBACK);

    int32_t                                phases = 0;
    ffxQueryDescUpscaleGetJitterPhaseCount qp{};
    qp.header.type    = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT;
    qp.renderWidth    = W;
    qp.displayWidth   = DW;
    qp.pOutPhaseCount = &phases;
    ffxQuery(&ctx, &qp.header);
    for (uint32_t f = 0; f < warmup + frames; ++f)
    {
        float                              jx = 0, jy = 0;
        ffxQueryDescUpscaleGetJitterOffset qj{};
        qj.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET;
        qj.index       = int32_t(f);
        qj.phaseCount  = phases;
        qj.pOutX       = &jx;
        qj.pOutY       = &jy;
        ffxQuery(&ctx, &qj.header);

        ffxDispatchDescUpscale dd{};
        dd.header.type            = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        dd.commandList            = gpu.list;
        dd.color                  = ffxApiGetResourceDX12(tColour);
        dd.depth                  = ffxApiGetResourceDX12(tDepth);
        dd.motionVectors          = ffxApiGetResourceDX12(tMotion);
        dd.output                 = ffxApiGetResourceDX12(tOutput, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        dd.jitterOffset           = {jx, jy};
        dd.motionVectorScale      = {1, 1};
        dd.renderSize             = {W, H};
        dd.upscaleSize            = {DW, DH};
        dd.enableSharpening       = sharpen > 0;
        dd.sharpness              = float(sharpen);
        dd.frameTimeDelta         = 16.7f;
        dd.preExposure            = 1;
        dd.reset                  = f == 0;
        dd.cameraNear             = 0.1f;
        dd.cameraFar              = 1000.0f;
        dd.cameraFovAngleVertical = 1.0f;
        gpu.list->EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * f);
        if (ffxDispatch(&ctx, &dd.header) != FFX_API_RETURN_OK)
            return std::printf("ffxDispatch failed\n"), 1;
        gpu.list->EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * f + 1);
        gpu.list->ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 2 * f, 2, result, 16 * f);
        gpu.Submit();  // one frame per submission, as a game runs
    }

    uint64_t freq = 1;
    gpu.queue->GetTimestampFrequency(&freq);
    uint64_t* t = nullptr;
    result->Map(0, nullptr, reinterpret_cast<void**>(&t));
    std::vector<double> ms;
    for (uint32_t f = warmup; f < warmup + frames; ++f)
        ms.push_back(double(t[2 * f + 1] - t[2 * f]) * 1000.0 / double(freq));
    result->Unmap(0, nullptr);
    std::sort(ms.begin(), ms.end());

    FfxApiEffectMemoryUsage             mem{};
    ffxQueryDescUpscaleGetGPUMemoryUsage qm{};
    qm.header.type            = FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE;
    qm.gpuMemoryUsageUpscaler = &mem;
    ffxQuery(&ctx, &qm.header);
    std::printf("%ux%u -> %ux%u%s: median %.3f ms, min %.3f, p95 %.3f over %u frames; %.1f MB\n", W, H, DW, DH,
                sharpen > 0 ? " + RCAS" : "", ms[ms.size() / 2], ms.front(), ms[ms.size() * 95 / 100], frames,
                double(mem.totalUsageInBytes) / 1048576.0);

    ffxDestroyContext(&ctx, nullptr);
    for (ID3D12Resource* r : {tColour, tDepth, tMotion, tOutput, result})
        r->Release();
    heap->Release();
    return 0;
}
