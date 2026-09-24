// test_net_vk.cpp - runs the production NSS network plan (engine core + INT8 kernels) on a
// Vulkan device and compares every tensor byte for byte with the exporter's integer golden.
//
//   test_net_vk <golden dir> [--wmma] [--drs]
//
// <golden dir> holds nss.rdnm, nss_w8.bin, nss_prod.rdnut and nss_prod_drs.rdnut from
// runtime/tools/nss_export.py.
// --wmma  runs the wave-matrix layers through the plain-HLSL emulation of the AGS shim
//         (RDNU_WMMA_EMULATE), which checks their addressing and epilogue; the matrix
//         intrinsics themselves only exist on AMD hardware.
// --drs   after the full frame, runs a smaller frame in the same arena (stale data present)
//         and checks it against its own golden: dynamic resolution must not leak old pixels.
#include "../common/rdnut.h"
#include "net_runner.h"

#include <cstdio>
#include <cstring>
#include <map>

namespace
{
struct Mismatch
{
    size_t count = 0, total = 0;
    int    firstX = -1, firstY = -1, firstC = -1, got = 0, want = 0;
    void   Add(int x, int y, int c, int g, int w)
    {
        ++total;
        if (g == w)
            return;
        if (!count++)
            firstX = x, firstY = y, firstC = c, got = g, want = w;
    }
};

bool Report(const std::string& name, const Mismatch& m, bool quietOk = false)
{
    if (m.count == 0)
    {
        if (!quietOk)
            std::printf("  %-12s OK   (%zu codes)\n", name.c_str(), m.total);
    }
    else
        std::printf("  %-12s FAIL %zu/%zu codes differ; first at x=%d y=%d c=%d got %d want %d\n", name.c_str(), m.count,
                    m.total, m.firstX, m.firstY, m.firstC, m.got, m.want);
    return m.count == 0;
}

void WriteInput(NetRunner& r, const rdnut::Tensor& in)
{
    const uint32_t H = in.dims[0], W = in.dims[1], C = in.dims[2];
    const rdnu::TensorLayout& t = r.plan.TensorOfKind(rdnu::TensorKind::Input);
    auto* p = static_cast<int8_t*>(r.input.mapped);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x)
            for (uint32_t c = 0; c < C; ++c)
                p[(size_t(y) * t.pitchPixels + x) * t.channels + c] = int8_t(int(in.data[(size_t(y) * W + x) * C + c]));
}

bool Compare(NetRunner& r, const std::map<std::string, rdnut::Tensor>& golden)
{
    const rdnu::Manifest& manifest = r.manifest;
    const rdnu::Plan& plan = r.plan;
    const vkc::Image& temporal = r.temporal;
    bool ok = true;
    const auto* ab = static_cast<const int8_t*>(r.arena.mapped);
    for (size_t ti = 0; ti < manifest.tensors.size(); ++ti)
    {
        const rdnu::TensorRecord& tr = manifest.tensors[ti];
        const rdnu::TensorLayout& t  = plan.Tensors()[ti];
        if (t.kind != rdnu::TensorKind::Arena)
            continue;
        auto at = [&](int x, int y, uint32_t c) {
            return int(ab[t.offset + (size_t(y + 1) * t.pitchPixels + (x + 1)) * t.channels + c]);
        };
        const rdnut::Tensor& g = golden.at(std::string("chk.") + tr.name);
        Mismatch m, ring;
        for (uint32_t y = 0; y < t.h; ++y)
            for (uint32_t x = 0; x < t.w; ++x)
                for (uint32_t c = 0; c < t.channels; ++c)
                    m.Add(int(x), int(y), int(c), at(int(x), int(y), c), int(g.data[(size_t(y) * t.w + x) * t.channels + c]));
        ok &= Report(tr.name, m);
        // every pixel a 3x3 tap can reach outside the valid region must read as -128
        for (int y = -1; y <= int(t.h); ++y)
            for (int x = -1; x <= int(t.w); ++x)
                if (x < 0 || y < 0 || x >= int(t.w) || y >= int(t.h))
                    for (uint32_t c = 0; c < t.channels; ++c)
                        ring.Add(x, y, int(c), at(x, y, c), -128);
        ok &= Report(std::string(tr.name) + ".ring", ring, true);
    }
    const rdnu::TensorLayout& kt = plan.TensorOfKind(rdnu::TensorKind::Kpn);
    const rdnut::Tensor& gk = golden.at("golden.kpn");
    Mismatch mk;
    const auto* kb = static_cast<const int8_t*>(r.kpn.mapped);
    for (uint32_t y = 0; y < kt.h; ++y)
        for (uint32_t x = 0; x < kt.w; ++x)
            for (uint32_t c = 0; c < kt.channels; ++c)
                mk.Add(int(x), int(y), int(c), kb[(size_t(y) * kt.pitchPixels + x) * kt.channels + c],
                       int(gk.data[(size_t(y) * kt.w + x) * kt.channels + c]));
    ok &= Report("kpn", mk);

    const rdnu::TensorLayout& tt = plan.TensorOfKind(rdnu::TensorKind::Temporal);
    std::vector<int8_t> tex(size_t(temporal.width) * temporal.height * 4);
    r.vk.Download(temporal, tex.data());
    const rdnut::Tensor& gt = golden.at("golden.temporal");
    Mismatch mt;
    for (uint32_t y = 0; y < tt.h; ++y)
        for (uint32_t x = 0; x < tt.w; ++x)
            for (uint32_t c = 0; c < 4; ++c)
                mt.Add(int(x), int(y), int(c), tex[(size_t(y) * temporal.width + x) * 4 + c],
                       int(gt.data[(size_t(y) * tt.w + x) * 4 + c]));
    ok &= Report("temporal", mt);
    return ok;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: test_net_vk <golden dir> [--wmma] [--drs]\n");
        return 1;
    }
    std::string dir = argv[1];
    bool wmma = false, drs = false;
    for (int i = 2; i < argc; ++i)
    {
        wmma |= !std::strcmp(argv[i], "--wmma");
        drs |= !std::strcmp(argv[i], "--drs");
    }

    std::string err;
    std::vector<uint8_t> manifestBytes, blob;
    if (!rdnut::ReadBytes(dir + "/nss.rdnm", manifestBytes) || !rdnut::ReadBytes(dir + "/nss_w8.bin", blob))
        return std::printf("cannot read nss.rdnm / nss_w8.bin in %s\n", dir.c_str()), 1;
    rdnu::Manifest manifest;
    if (!rdnu::ParseManifest(manifestBytes.data(), manifestBytes.size(), manifest, err))
        return std::printf("manifest: %s\n", err.c_str()), 1;
    std::map<std::string, rdnut::Tensor> golden, goldenDrs;
    if (!rdnut::Load(dir + "/nss_prod.rdnut", golden, err))
        return std::printf("%s\n", err.c_str()), 1;
    if (drs && !rdnut::Load(dir + "/nss_prod_drs.rdnut", goldenDrs, err))
        return std::printf("%s\n", err.c_str()), 1;
    const rdnut::Tensor& in = golden.at("input_nhwc");
    const uint32_t H = in.dims[0], W = in.dims[1];

    rdnu::PlanConfig cfg;
    cfg.maxWidth  = W;
    cfg.maxHeight = H;
    cfg.useWmma   = wmma;
    rdnu::Plan plan;
    if (!plan.Build(manifest, cfg, err))
        return std::printf("plan: %s\n", err.c_str()), 1;

    vkc::Context vk;
    if (!vk.Init(err))
        return std::printf("vulkan: %s\n", err.c_str()), 1;
    std::printf("device: %s   network %ux%u   %s path   arena %llu bytes\n", vk.DeviceName().c_str(), W, H,
                wmma ? "DP4a + WMMA (emulated)" : "DP4a", (unsigned long long)plan.ArenaBytes());

    NetRunner r{vk, manifest, plan};
    r.shaderDir = std::string(RDNU_SOURCE_DIR) + "/runtime/shaders/net";
    r.Allocate(blob, W, H);

    bool ok = true;
    WriteInput(r, in);
    std::printf("frame %ux%u\n", plan.Width(), plan.Height());
    if (!r.Run(err))
        return std::printf("%s\n", err.c_str()), 1;
    ok &= Compare(r, golden);

    if (drs)
    {
        const rdnut::Tensor& in2 = goldenDrs.at("input_nhwc");
        if (!plan.Update(in2.dims[1], in2.dims[0], err))
            return std::printf("plan: %s\n", err.c_str()), 1;
        WriteInput(r, in2);
        std::printf("frame %ux%u (same allocation, stale data present)\n", plan.Width(), plan.Height());
        if (!r.Run(err))
            return std::printf("%s\n", err.c_str()), 1;
        ok &= Compare(r, goldenDrs);
    }

    r.Release();
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 2;
}
