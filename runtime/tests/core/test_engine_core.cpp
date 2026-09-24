// test_engine_core.cpp - checks the platform-free network planner without a GPU.
//
//   test_engine_core <model dir> [<golden dir>]
//
// <model dir> holds nss.rdnm and nss_w8.bin (runtime/models/NSS_INT8). Checks manifest
// parsing, arena layout, grids, address bounds and barriers for common and odd render sizes.
// With <golden dir> (runtime/tools/nss_export.py output) it also executes the plan on the CPU
// with the kernels' integer semantics and compares every tensor with the exporter's golden,
// including a smaller frame in the same arena to catch stale data.
#include "../../src/engine/rdnu_engine_core.h"
#include "../common/rdnut.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>

using namespace rdnu;

namespace
{
int g_failures = 0;

#define CHECK(cond, ...)                                  \
    do                                                    \
    {                                                     \
        if (!(cond))                                      \
        {                                                 \
            ++g_failures;                                 \
            std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                     \
            std::printf("\n");                            \
        }                                                 \
    } while (0)

// Highest byte a dispatch may touch in its input and output views, from the kernel rules:
// 3x3 taps at stride 1/2 or through the nearest upsample, borders of one pixel, WMMA runs of
// 16 pixels, DP4a tiles of 8.
void CheckBounds(const Plan& plan, const Manifest& m, const Dispatch& d)
{
    const LayerRecord&  l  = m.layers[d.layer];
    const TensorLayout& ti = plan.Tensors()[l.inTensor];
    const TensorLayout& to = plan.Tensors()[l.outTensor];
    const NetConstants& c  = d.constants;

    const bool wmma = d.key.kernel == Kernel::Wmma;
    uint32_t   covW = wmma ? d.groups[0] * 16 * (8 / (l.cout / 16)) : d.groups[0] * 8;
    uint32_t   covH = wmma ? d.groups[1] : d.groups[1] * 8;
    CHECK(covW >= to.w && covH >= to.h, "%s: grid %ux%u does not cover %ux%u", l.name, covW, covH, to.w, to.h);

    auto limit = [&](const TensorLayout& t) -> uint64_t {
        switch (t.kind)
        {
        case TensorKind::Arena: return t.offset + t.bytes;
        case TensorKind::Temporal: return 0;
        default: return t.bytes;
        }
    };

    // input: last tap of the last computed pixel (WMMA computes whole 16-pixel runs)
    int lastX = int(wmma ? RoundUp(to.w, 16) : to.w) - 1, lastY = int(to.h) - 1;
    int sx = lastX + 1, sy = lastY + 1;
    if (l.stride == 2)
        sx = 2 * lastX + 1, sy = 2 * lastY + 1;
    else if (l.upsample)
        sx = (lastX + 1) >> 1, sy = (lastY + 1) >> 1;
    if (ti.kind == TensorKind::Arena)
    {
        CHECK(sx <= int(ti.pitchPixels) - 2 && sy <= int(ti.rows) - 2, "%s: tap (%d,%d) outside %ux%u", l.name, sx, sy,
              ti.pitchPixels, ti.rows);
        uint64_t first = uint64_t(int64_t(c.inBase) - int64_t(c.inPitch) - int64_t(c.inPix));
        CHECK(first >= ti.offset, "%s: first tap before the tensor", l.name);
        uint64_t last = uint64_t(c.inBase) + uint64_t(sy) * c.inPitch + uint64_t(sx) * c.inPix + l.cin;
        CHECK(last <= limit(ti), "%s: last tap %llu beyond %llu", l.name, (unsigned long long)last, (unsigned long long)limit(ti));
    }
    if (to.kind != TensorKind::Temporal)
    {
        uint32_t ring = c.outRing;
        uint64_t last = uint64_t(c.outBase) + uint64_t(to.h - 1 + ring) * c.outPitch + uint64_t(to.w - 1 + ring) * c.outPix + l.cout;
        CHECK(last <= limit(to), "%s: last write beyond the view", l.name);
        CHECK(l.outChannelOffset + l.cout <= to.channels, "%s: channel range", l.name);
    }
}

// Re-derives the barrier list: a barrier is needed before a layer that reads, or writes, a
// channel range written since the previous barrier, or writes a range read since then.
void CheckBarriers(const Plan& plan, const Manifest& m)
{
    struct R { uint32_t t, lo, hi; };
    auto overlap = [](const std::vector<R>& v, R r) {
        for (const R& x : v)
            if (x.t == r.t && x.lo < r.hi && r.lo < x.hi)
                return true;
        return false;
    };
    std::vector<R> w, rd;
    for (size_t i = 0; i < plan.Dispatches().size(); ++i)
    {
        const Dispatch&    d = plan.Dispatches()[i];
        const LayerRecord& l = m.layers[d.layer];
        R in{l.inTensor, l.inChannelOffset, l.inChannelOffset + l.cin};
        R out{l.outTensor, l.outChannelOffset, l.outChannelOffset + l.cout};
        bool need = i == 0 || overlap(w, in) || overlap(w, out) || overlap(rd, out);
        CHECK(need == d.barrierBefore, "%s: barrier %d, expected %d", l.name, int(d.barrierBefore), int(need));
        if (need)
            w.clear(), rd.clear();
        w.push_back(out);
        rd.push_back(in);
    }
}

void CheckPlan(const Manifest& m, uint32_t renderW, uint32_t renderH, bool wmma)
{
    const uint32_t W = RoundUp(renderW, 8), H = RoundUp(renderH, 8);
    PlanConfig cfg;
    cfg.maxWidth  = W;
    cfg.maxHeight = H;
    cfg.useWmma   = wmma;
    Plan        plan;
    std::string err;
    CHECK(plan.Build(m, cfg, err), "%ux%u: %s", W, H, err.c_str());

    uint64_t end = 0;
    for (const TensorLayout& t : plan.Tensors())
        if (t.kind == TensorKind::Arena)
        {
            CHECK(t.offset % 256 == 0, "arena tensor offset %llu not 256-aligned", (unsigned long long)t.offset);
            CHECK(t.offset >= end, "arena tensors overlap");
            CHECK(t.pitchPixels >= t.maxW + 2 && t.rows >= t.maxH + 2, "arena tensor has no border");
            end = t.offset + t.bytes;
        }
    CHECK(end <= plan.ArenaBytes(), "arena size");
    CHECK(plan.InputBytes() == uint64_t(W) * H * 12, "input tensor is dense NHWC");
    CHECK(plan.KpnBytes() == uint64_t(W / 4) * (H / 4) * 36, "KPN tensor is dense NHWC");

    size_t wmmaCount = 0;
    for (const Dispatch& d : plan.Dispatches())
    {
        CheckBounds(plan, m, d);
        wmmaCount += d.key.kernel == Kernel::Wmma;
    }
    CHECK(!wmma || wmmaCount > 0, "no WMMA layers selected");
    CheckBarriers(plan, m);

    // dynamic resolution: a smaller frame reuses the allocation
    const uint64_t arena = plan.ArenaBytes();
    const uint32_t w2 = std::max(8u, RoundUp(W * 2 / 3, 8)), h2 = std::max(8u, RoundUp(H * 2 / 3, 8));
    CHECK(plan.Update(w2, h2, err), "update %ux%u: %s", w2, h2, err.c_str());
    CHECK(plan.ArenaBytes() == arena, "update changed the arena");
    for (const Dispatch& d : plan.Dispatches())
        CheckBounds(plan, m, d);
    CheckBarriers(plan, m);
    CHECK(!plan.Update(W + 8, H, err), "update beyond the maximum accepted");
    CHECK(!plan.Update(W - 4, H, err), "update to a non-multiple of 8 accepted");

    std::printf("  %4ux%-4u -> %4ux%-4u %-5s arena %6.1f MB, %zu dispatches, %zu WMMA\n", renderW, renderH, W, H,
                wmma ? "wmma" : "dp4a", arena / 1048576.0, plan.Dispatches().size(), wmmaCount);
}

// The kernels' arithmetic on the CPU, reading OHWI weights.
struct CpuNet
{
    const Manifest&             m;
    const std::vector<uint8_t>& blob;
    std::vector<int8_t>         arena, input, kpn, temporal;
    uint32_t                    temporalPitch = 0;

    int8_t  W8(uint32_t off) const { return int8_t(blob[off]); }
    int32_t I32(uint32_t off) const { int32_t v; std::memcpy(&v, &blob[off], 4); return v; }
    float   F32(uint32_t off) const { float v; std::memcpy(&v, &blob[off], 4); return v; }

    std::vector<int8_t>& Buf(Resource r) { return r == Resource::Input ? input : r == Resource::Kpn ? kpn : arena; }

    void Run(const Plan& plan)
    {
        for (const Dispatch& d : plan.Dispatches())
        {
            const LayerRecord&  l  = m.layers[d.layer];
            const TensorLayout& ti = plan.Tensors()[l.inTensor];
            const NetConstants& c  = d.constants;
            const bool bordered    = ti.kind == TensorKind::Arena;
            std::vector<int8_t>& in  = Buf(d.in);
            std::vector<int8_t>& out = Buf(d.out);
            for (int y = 0; y < int(c.outH); ++y)
                for (int x = 0; x < int(c.outW); ++x)
                    for (uint32_t oc = 0; oc < l.cout; ++oc)
                    {
                        int32_t acc = I32(l.offBias + 4 * oc);
                        for (int ky = 0; ky < 3; ++ky)
                            for (int kx = 0; kx < 3; ++kx)
                            {
                                int sx = x + kx - 1, sy = y + ky - 1;
                                if (l.stride == 2)
                                    sx = 2 * x + kx - 1, sy = 2 * y + ky - 1;
                                else if (l.upsample)
                                    sx >>= 1, sy >>= 1;
                                const bool inside = sx >= 0 && sy >= 0 && sx < int(c.inW) && sy < int(c.inH);
                                const uint32_t wo = l.offWOhwi + ((oc * 3 + ky) * 3 + kx) * l.cin;
                                for (uint32_t ch = 0; ch < l.cin; ++ch)
                                {
                                    int v = -128;
                                    if (bordered || inside)
                                        v = in[size_t(int64_t(c.inBase) + int64_t(sy) * c.inPitch + int64_t(sx) * c.inPix + ch)];
                                    acc += v * W8(wo + ch);
                                }
                            }
                        const float r = F32(l.offScale + 4 * oc);
                        int q;
                        if (l.act == uint32_t(Act::Relu))
                            q = std::clamp(int(std::nearbyint(float(acc) * r)) - 128, -128, 127);
                        else
                            q = W8(l.offLut + std::clamp(int(std::nearbyint(float(acc) * r)) + l.zpLogit, -128, 127) + 128);
                        if (d.key.outKind == uint32_t(OutKind::Temporal))
                            temporal[(size_t(y) * temporalPitch + x) * 4 + oc] = int8_t(q);
                        else
                            out[size_t(c.outBase) + size_t(y) * c.outPitch + size_t(x) * c.outPix + oc] = int8_t(q);
                    }
            if (c.outRing)
                for (int y = 0; y <= int(c.outH); ++y)
                    for (int x = 0; x <= int(c.outW); ++x)
                        if (x == int(c.outW) || y == int(c.outH))
                            for (uint32_t oc = 0; oc < l.cout; ++oc)
                                out[size_t(c.outBase) + size_t(y) * c.outPitch + size_t(x) * c.outPix + oc] = -128;
        }
    }
};

bool CompareGolden(const Plan& plan, const Manifest& m, CpuNet& net, const std::map<std::string, rdnut::Tensor>& g)
{
    bool ok = true;
    auto cmp = [&](const std::string& name, const rdnut::Tensor& want, auto get) {
        const uint32_t h = want.dims[0], w = want.dims[1], ch = want.dims[2];
        size_t bad = 0;
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                for (uint32_t c = 0; c < ch; ++c)
                    bad += get(x, y, c) != int(want.data[(size_t(y) * w + x) * ch + c]);
        CHECK(bad == 0, "%s: %zu codes differ", name.c_str(), bad);
        ok &= bad == 0;
    };
    for (size_t ti = 0; ti < m.tensors.size(); ++ti)
    {
        const TensorLayout& t = plan.Tensors()[ti];
        if (t.kind != TensorKind::Arena)
            continue;
        cmp(m.tensors[ti].name, g.at(std::string("chk.") + m.tensors[ti].name), [&](uint32_t x, uint32_t y, uint32_t c) {
            return int(net.arena[t.offset + (size_t(y + 1) * t.pitchPixels + x + 1) * t.channels + c]);
        });
    }
    const TensorLayout& k = plan.TensorOfKind(TensorKind::Kpn);
    cmp("kpn", g.at("golden.kpn"), [&](uint32_t x, uint32_t y, uint32_t c) { return int(net.kpn[(size_t(y) * k.pitchPixels + x) * 36 + c]); });
    cmp("temporal", g.at("golden.temporal"),
        [&](uint32_t x, uint32_t y, uint32_t c) { return int(net.temporal[(size_t(y) * net.temporalPitch + x) * 4 + c]); });
    return ok;
}

void WriteInput(const Plan& plan, CpuNet& net, const rdnut::Tensor& in)
{
    const TensorLayout& t = plan.TensorOfKind(TensorKind::Input);
    for (uint32_t y = 0; y < in.dims[0]; ++y)
        for (uint32_t x = 0; x < in.dims[1]; ++x)
            for (uint32_t c = 0; c < 12; ++c)
                net.input[(size_t(y) * t.pitchPixels + x) * 12 + c] = int8_t(int(in.data[(size_t(y) * in.dims[1] + x) * 12 + c]));
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
        return std::printf("usage: test_engine_core <model dir> [<golden dir>]\n"), 1;
    std::string modelDir = argv[1], err;
    std::vector<uint8_t> bytes, blob;
    if (!rdnut::ReadBytes(modelDir + "/nss.rdnm", bytes) || !rdnut::ReadBytes(modelDir + "/nss_w8.bin", blob))
        return std::printf("cannot read %s/nss.rdnm and nss_w8.bin\n", modelDir.c_str()), 1;

    std::printf("manifest\n");
    Manifest m;
    CHECK(ParseManifest(bytes.data(), bytes.size(), m, err), "%s", err.c_str());
    CHECK(m.tensors.size() == 13 && m.layers.size() == 14, "%zu tensors, %zu layers", m.tensors.size(), m.layers.size());
    CHECK(m.blobBytes == blob.size(), "blob %u bytes, file %zu", m.blobBytes, blob.size());
    Manifest bad;
    CHECK(!ParseManifest(bytes.data(), bytes.size() - 1, bad, err), "truncated manifest accepted");
    std::vector<uint8_t> magic = bytes;
    magic[0] = 'X';
    CHECK(!ParseManifest(magic.data(), magic.size(), bad, err), "bad magic accepted");

    std::printf("plans\n");
    const uint32_t sizes[][2] = {{960, 540}, {1280, 720}, {1920, 1080}, {2560, 1440}, {3840, 2160}, {1290, 722}, {8, 8}, {100, 36}};
    for (auto& s : sizes)
        for (bool wmma : {false, true})
            CheckPlan(m, s[0], s[1], wmma);

    if (argc > 2)
    {
        std::printf("cpu execution against the exporter golden\n");
        std::string gdir = argv[2];
        std::map<std::string, rdnut::Tensor> g, g2;
        if (!rdnut::Load(gdir + "/nss_prod.rdnut", g, err) || !rdnut::Load(gdir + "/nss_prod_drs.rdnut", g2, err))
            return std::printf("%s\n", err.c_str()), 1;
        const rdnut::Tensor& in = g.at("input_nhwc");
        PlanConfig cfg;
        cfg.maxWidth  = in.dims[1];
        cfg.maxHeight = in.dims[0];
        Plan plan;
        CHECK(plan.Build(m, cfg, err), "%s", err.c_str());
        CpuNet net{m, blob};
        net.arena.assign(plan.ArenaBytes(), -128);
        net.input.assign(plan.InputBytes(), 0);
        net.kpn.assign(plan.KpnBytes(), 0x55);
        net.temporalPitch = cfg.maxWidth;
        net.temporal.assign(size_t(cfg.maxWidth) * cfg.maxHeight * 4, 0);
        WriteInput(plan, net, in);
        net.Run(plan);
        if (CompareGolden(plan, m, net, g))
            std::printf("  %ux%u ok\n", plan.Width(), plan.Height());
        const rdnut::Tensor& in2 = g2.at("input_nhwc");
        CHECK(plan.Update(in2.dims[1], in2.dims[0], err), "%s", err.c_str());
        WriteInput(plan, net, in2);
        net.Run(plan);
        if (CompareGolden(plan, m, net, g2))
            std::printf("  %ux%u in the same arena ok\n", plan.Width(), plan.Height());
    }

    std::printf("%s (%d failures)\n", g_failures ? "FAIL" : "PASS", g_failures);
    return g_failures ? 2 : 0;
}
