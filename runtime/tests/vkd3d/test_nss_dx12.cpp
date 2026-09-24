// test_nss_dx12.cpp - Arm's NSS component end to end on D3D12: ffxNssContextCreate/Dispatch
// through the RDNU backend (ffx_nss_dx12), embedded DXIL passes and the INT8 network, on real
// frames. Builds on Windows and on Linux against vkd3d-proton (runtime/tests/vkd3d/d3d12.h),
// where it runs on Mesa lavapipe. Prints PSNR per frame and writes a PPM preview.
//
//   test_nss_dx12 <golden dir> [--frames N] [--scale S] [--out file.ppm]
//
// <golden dir>/nss_frames.rdnut comes from runtime/tools/nss_pass_frames.py. On lavapipe set
// RDNU_TEST_ASSUME_CAPS=1 (see LavapipeCaps).
#include "../../src/backend_dx12/ffx_nss_dx12.h"
#include "../common/rdnut.h"

#include <FidelityFX/host/ffx_nss.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

namespace
{
struct Gpu
{
    ID3D12Device*              device = nullptr;
    ID3D12CommandQueue*        queue  = nullptr;
    ID3D12CommandAllocator*    alloc  = nullptr;
    ID3D12GraphicsCommandList* list   = nullptr;
    ID3D12Fence*               fence  = nullptr;
    uint64_t                   value  = 0;

    bool Init()
    {
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        return SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))) &&
               SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) &&
               SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) &&
               SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&list))) &&
               SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    }

    void Submit()
    {
        list->Close();
        ID3D12CommandList* l[] = {list};
        queue->ExecuteCommandLists(1, l);
        queue->Signal(fence, ++value);
        fence->SetEventOnCompletion(value, nullptr);
        alloc->Reset();
        list->Reset(alloc, nullptr);
    }

    ID3D12Resource* Texture(uint32_t w, uint32_t h, DXGI_FORMAT f, bool uav, D3D12_RESOURCE_STATES state)
    {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC d{};
        d.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width            = w;
        d.Height           = h;
        d.DepthOrArraySize = 1;
        d.MipLevels        = 1;
        d.Format           = f;
        d.SampleDesc.Count = 1;
        d.Flags            = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        ID3D12Resource* r  = nullptr;
        device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r));
        return r;
    }

    ID3D12Resource* Buffer(uint64_t bytes, D3D12_HEAP_TYPE heap)
    {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = heap;
        D3D12_RESOURCE_DESC d{};
        d.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width            = bytes;
        d.Height           = 1;
        d.DepthOrArraySize = 1;
        d.MipLevels        = 1;
        d.SampleDesc.Count = 1;
        d.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ID3D12Resource* r  = nullptr;
        device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d,
                                        heap == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST,
                                        nullptr, IID_PPV_ARGS(&r));
        return r;
    }

    void Barrier(ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b)
    {
        D3D12_RESOURCE_BARRIER x{};
        x.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        x.Transition.pResource   = r;
        x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        x.Transition.StateBefore = a;
        x.Transition.StateAfter  = b;
        list->ResourceBarrier(1, &x);
    }

    // Uploads tightly packed texels to a texture in `state`; the upload buffer is returned for release after Submit.
    ID3D12Resource* Upload(ID3D12Resource* t, const void* texels, uint32_t texelBytes, D3D12_RESOURCE_STATES state)
    {
        D3D12_RESOURCE_DESC                d = t->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT64                             total = 0;
        device->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, &total);
        ID3D12Resource* up = Buffer(total, D3D12_HEAP_TYPE_UPLOAD);
        uint8_t*        p  = nullptr;
        up->Map(0, nullptr, reinterpret_cast<void**>(&p));
        for (uint32_t y = 0; y < d.Height; ++y)
            std::memcpy(p + size_t(y) * fp.Footprint.RowPitch, static_cast<const uint8_t*>(texels) + size_t(y) * d.Width * texelBytes,
                        size_t(d.Width) * texelBytes);
        up->Unmap(0, nullptr);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource       = t;
        dst.Type            = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.pResource       = up;
        src.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = fp;
        Barrier(t, state, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        Barrier(t, D3D12_RESOURCE_STATE_COPY_DEST, state);
        return up;
    }

    std::vector<uint8_t> Read(ID3D12Resource* t, uint32_t texelBytes, D3D12_RESOURCE_STATES state)
    {
        D3D12_RESOURCE_DESC                d = t->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT64                             total = 0;
        device->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, &total);
        ID3D12Resource*             rb = Buffer(total, D3D12_HEAP_TYPE_READBACK);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource       = rb;
        dst.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = fp;
        src.pResource       = t;
        src.Type            = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        Barrier(t, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        Barrier(t, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
        Submit();
        std::vector<uint8_t> out(size_t(d.Width) * d.Height * texelBytes);
        uint8_t*             p = nullptr;
        rb->Map(0, nullptr, reinterpret_cast<void**>(&p));
        for (uint32_t y = 0; y < d.Height; ++y)
            std::memcpy(&out[size_t(y) * d.Width * texelBytes], p + size_t(y) * fp.Footprint.RowPitch, size_t(d.Width) * texelBytes);
        rb->Unmap(0, nullptr);
        rb->Release();
        return out;
    }
};

uint16_t ToHalf(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000;
    const int      e = int((x >> 23) & 0xff) - 112;
    uint32_t       m = x & 0x7fffff;
    if (e <= 0)
        return uint16_t(s);  // inputs here are normal or zero
    if (e >= 31)
        return uint16_t(s | 0x7c00);
    uint32_t h = (uint32_t(e) << 10) | (m >> 13), r = m & 0x1fff;
    return uint16_t(s | (h + (r > 0x1000 || (r == 0x1000 && (h & 1)))));
}

float FromHalf(uint16_t h)
{
    const uint32_t s = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t       x = e == 0 ? s : e == 31 ? s | 0x7f800000 | (m << 13) : s | ((e + 112) << 23) | (m << 13);
    if (e == 0 && m)
    {
        float f = std::ldexp(float(m), -24);
        return h & 0x8000 ? -f : f;
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

void Message(FfxMsgType, const char* m)
{
    std::fprintf(stderr, "nss: %s\n", m);
}

void BackendMessage(uint32_t, const char* m)
{
    std::fprintf(stderr, "backend: %s\n", m);
}

// vkd3d-proton on lavapipe runs SM 6.4 and 16-bit shaders but reports SM 6.0 and no native
// 16-bit ops, because lavapipe does not preserve denormals. Report what the device really runs.
FfxGetDeviceCapabilitiesFunc g_realCaps = nullptr;

FfxErrorCode LavapipeCaps(FfxInterface* i, FfxDeviceCapabilities* caps)
{
    FfxErrorCode e = g_realCaps(i, caps);
    if (std::getenv("RDNU_TEST_ASSUME_CAPS"))
    {
        caps->maximumSupportedShaderModel = FFX_SHADER_MODEL_6_6;
        caps->fp16Supported = caps->tensorSupported = caps->dataGraphSupported = true;
        i->deviceCapabilities = *caps;
    }
    return e;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
        return std::printf("usage: test_nss_dx12 <golden dir> [--frames N] [--scale S] [--out file.ppm]\n"), 1;
    std::string dir = argv[1], outPath = "/tmp/rdnu_nss_dx12.ppm", err;
    uint32_t    frames = 4;
    double      scale  = 2;
    for (int i = 2; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--frames") && i + 1 < argc)
            frames = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--scale") && i + 1 < argc)
            scale = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc)
            outPath = argv[++i];
    }
    std::map<std::string, rdnut::Tensor> seq;
    if (!rdnut::Load(dir + "/nss_frames.rdnut", seq, err))
        return std::printf("%s\n", err.c_str()), 1;
    const rdnut::Tensor &colour = seq.at("colour"), &depth = seq.at("depth"), &motion = seq.at("motion");
    const rdnut::Tensor &jitter = seq.at("jitter"), &exposure = seq.at("exposure"), &camera = seq.at("camera");
    const rdnut::Tensor& truth = seq.at("truth");
    frames = std::min(frames, colour.dims[0]);
    const uint32_t W = colour.dims[2], H = colour.dims[1];
    const uint32_t DW = uint32_t(std::lround(W * scale)), DH = uint32_t(std::lround(H * scale));
    const bool     haveTruth = truth.dims[1] == DH && truth.dims[2] == DW;

    Gpu gpu;
    if (!gpu.Init())
        return std::printf("no D3D12 device\n"), 1;

    FfxNssContextDescription cd{};
    cd.qualityMode = FFX_NSS_SHADER_QUALITY_MODE_QUALITY;
    cd.flags       = FFX_NSS_CONTEXT_FLAG_QUANTIZED | FFX_NSS_CONTEXT_FLAG_ALLOW_16BIT | FFX_NSS_CONTEXT_FLAG_MANAGE_HISTORY |
               (camera.data[3] != 0 ? FFX_NSS_CONTEXT_FLAG_DEPTH_INFINITE : 0);
    cd.renderSize  = {W, H};
    cd.upscaleSize = {DW, DH};
    cd.displaySize = {DW, DH};
    cd.fpMessage   = Message;
    std::vector<uint8_t> scratch(ffxGetScratchMemorySizeDX12(1));
    if (ffxGetInterfaceDX12(&cd.backendInterface, gpu.device, scratch.data(), scratch.size(), 1) != FFX_OK)
        return std::printf("backend interface failed\n"), 1;
    cd.backendInterface.fpSetMessageCallback(&cd.backendInterface, BackendMessage);
    g_realCaps                              = cd.backendInterface.fpGetDeviceCapabilities;
    cd.backendInterface.fpGetDeviceCapabilities = LavapipeCaps;
    auto         context = std::make_unique<FfxNssContext>();
    FfxErrorCode e       = ffxNssContextCreate(context.get(), &cd);
    if (e != FFX_OK)
        return std::printf("ffxNssContextCreate failed: %d\n", int(e)), 1;
    std::printf("NSS context %ux%u -> %ux%u created\n", W, H, DW, DH);

    const D3D12_RESOURCE_STATES read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    ID3D12Resource* tColour = gpu.Texture(W, H, DXGI_FORMAT_R16G16B16A16_FLOAT, false, read);
    ID3D12Resource* tDepth  = gpu.Texture(W, H, DXGI_FORMAT_R32_TYPELESS, false, read);
    ID3D12Resource* tMotion = gpu.Texture(W, H, DXGI_FORMAT_R16G16_FLOAT, false, read);
    ID3D12Resource* tOutput = gpu.Texture(DW, DH, DXGI_FORMAT_R16G16B16A16_FLOAT, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    std::vector<float> out(size_t(DW) * DH * 3);
    for (uint32_t t = 0; t < frames; ++t)
    {
        const size_t          np = size_t(W) * H;
        std::vector<uint16_t> col(np * 4), mv(np * 2);
        for (size_t i = 0; i < np * 4; ++i)
            col[i] = ToHalf(colour.data[t * np * 4 + i]);
        for (size_t i = 0; i < np * 2; ++i)
            mv[i] = ToHalf(motion.data[t * np * 2 + i]);
        ID3D12Resource* ups[] = {gpu.Upload(tColour, col.data(), 8, read), gpu.Upload(tDepth, &depth.data[t * np], 4, read),
                                 gpu.Upload(tMotion, mv.data(), 4, read)};

        FfxNssDispatchDescription dd{};
        dd.commandList            = ffxGetCommandListDX12(gpu.list);
        dd.color                  = ffxGetResourceDX12(tColour, FFX_RESOURCE_STATE_COMPUTE_READ, "colour");
        dd.depth                  = ffxGetResourceDX12(tDepth, FFX_RESOURCE_STATE_COMPUTE_READ, "depth");
        dd.motionVectors          = ffxGetResourceDX12(tMotion, FFX_RESOURCE_STATE_COMPUTE_READ, "motion");
        dd.output                 = ffxGetResourceDX12(tOutput, FFX_RESOURCE_STATE_GENERIC_UAV, "output");
        dd.jitterOffset           = {jitter.data[t * 2], jitter.data[t * 2 + 1]};
        dd.renderSize             = {W, H};
        dd.upscaleSize            = {DW, DH};
        dd.cameraNear             = camera.data[t * 4];
        dd.cameraFar              = camera.data[t * 4 + 1];
        dd.cameraFovAngleVertical = camera.data[t * 4 + 2];
        dd.exposure               = exposure.data[t];
        dd.motionVectorScale      = {1, 1};
        dd.frameTimeDelta         = 16.7f;
        dd.reset                  = t == 0;
        if ((e = ffxNssContextDispatch(context.get(), &dd)) != FFX_OK)
            return std::printf("ffxNssContextDispatch failed: %d\n", int(e)), 1;
        gpu.Submit();
        for (ID3D12Resource* u : ups)
            u->Release();

        std::vector<uint8_t> o = gpu.Read(tOutput, 8, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        for (size_t i = 0; i < size_t(DW) * DH; ++i)
            for (int k = 0; k < 3; ++k)
            {
                uint16_t h;
                std::memcpy(&h, &o[i * 8 + k * 2], 2);
                out[i * 3 + k] = FromHalf(h);
            }
        if (haveTruth)
        {
            const float ex = exposure.data[t];
            double      se = 0;
            for (uint32_t y = 0; y < DH; ++y)
                for (uint32_t x = 0; x < DW; ++x)
                    for (int k = 0; k < 3; ++k)
                    {
                        const double a = out[(size_t(y) * DW + x) * 3 + k] * ex, b = truth.data[((t * size_t(DH) + y) * DW + x) * 4 + k] * ex;
                        const double d = a / (1 + a) - b / (1 + b);
                        se += d * d;
                    }
            std::printf("frame %u  psnr (tonemapped) %.2f dB\n", t, 10 * std::log10(1.0 / std::max(1e-12, se / (double(DW) * DH * 3))));
        }
        else
            std::printf("frame %u done\n", t);
    }

    if (FILE* f = std::fopen(outPath.c_str(), "wb"))
    {
        const float ex = exposure.data[frames - 1];
        std::fprintf(f, "P6\n%u %u\n255\n", DW, DH);
        for (float v : out)
        {
            v *= ex;
            std::fputc(int(std::clamp(std::pow(v / (1 + v), 1 / 2.2f), 0.f, 1.f) * 255 + 0.5f), f);
        }
        std::fclose(f);
        std::printf("preview %s\n", outPath.c_str());
    }

    ffxNssContextDestroy(context.get());
    ffxReleaseInterfaceDX12(&cd.backendInterface);
    for (ID3D12Resource* r : {tColour, tDepth, tMotion, tOutput})
        r->Release();
    std::printf("PASS\n");
    return 0;
}
