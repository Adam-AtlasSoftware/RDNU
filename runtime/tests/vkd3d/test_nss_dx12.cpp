// test_nss_dx12.cpp - Arm's NSS component end to end on D3D12: ffxNssContextCreate/Dispatch
// through the RDNU backend (ffx_nss_dx12), embedded DXIL passes and the INT8 network, on real
// frames. Builds on Windows and on Linux against vkd3d-proton (runtime/tests/vkd3d/d3d12.h),
// where it runs on Mesa lavapipe. Prints PSNR per frame and writes a PPM preview.
//
//   test_nss_dx12 <golden dir> [--frames N] [--scale S] [--out file.ppm]
//
// <golden dir>/nss_frames.rdnut comes from runtime/tools/nss_pass_frames.py. On vkd3d-proton
// set VKD3D_SHADER_MODEL=6_6 (see runtime/tools/setup_vkd3d_proton.sh).
#include "../../src/backend_dx12/ffx_nss_dx12.h"
#include "../common/rdnut.h"
#include "d3d12_util.h"

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
void Message(FfxMsgType, const char* m)
{
    std::fprintf(stderr, "nss: %s\n", m);
}

void BackendMessage(uint32_t, const char* m)
{
    std::fprintf(stderr, "backend: %s\n", m);
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
