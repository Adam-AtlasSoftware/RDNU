// test_passes_vk.cpp - runs whole NSS frames on a Vulkan device: Arm's passes ported to HLSL
// around the production INT8 network. Every pass is replayed with Arm's original GLSL on the
// same inputs and its outputs are compared, so the port is checked against the shipped
// reference on real content. Prints PSNR against the ground truth and writes PPM previews.
//
//   test_passes_vk <golden dir> [--frames N] [--scale S] [--sharpen stops] [--wmma] [--out dir]
//
// <golden dir> holds nss.rdnm and nss_w8.bin (nss_export.py) and nss_frames.rdnut
// (nss_pass_frames.py). --scale 2 uses the static 2x filter, any other factor the dynamic
// offset LUT. --sharpen also runs the RCAS pass (0 strongest). Needs DXC ($DXC) and
// glslangValidator ($GLSLANG or PATH).
#include "../common/rdnut.h"
#include "net_runner.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

namespace
{
const std::string kSrc  = RDNU_SOURCE_DIR;
const std::string kGpu  = kSrc + "/runtime/external/neural-graphics-sdk-for-game-engines/sdk/include/FidelityFX/gpu";
const std::string kGlsl = kSrc + "/runtime/external/neural-graphics-sdk-for-game-engines/sdk/src/backends/vk/shaders/nss";
const std::string kHlsl = kSrc + "/runtime/shaders/nss";

// NssConstants in ffx_nss_private.h, cbNSS in the shaders
struct NssConstants
{
    float    deviceToViewDepth[4];
    float    jitterOffset[4];
    float    jitterOffsetTm1[4];
    float    scaleFactor[4];
    uint32_t outputDims[2];
    uint32_t inputDims[2];
    float    invOutputDims[2];
    float    invInputDims[2];
    uint32_t depthTm1Size[2];
    float    invDepthTm1Size[2];
    uint32_t inputTensorSize[2];
    float    inputTensorSizeRcp[2];
    uint32_t kpnDimension[2];
    float    motionVectorScale[2];
    float    paddingScale[2];
    float    depthClipRequiredSepScale;
    float    depthClipPower;
    float    kpnScale[2];
    uint32_t debugViewMode;
    float    notHistoryReset;
    float    exposure[2];
    uint32_t indexModulo[2];
    uint32_t reducedInputModulo[2];
    int32_t  lutOffset[2];
};
static_assert(sizeof(NssConstants) == 208, "cbNSS layout");

uint16_t ToHalf(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    const int      e    = int((x >> 23) & 0xff) - 127 + 15;
    uint32_t       m    = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff)
        return uint16_t(sign | 0x7c00 | (m ? 0x200 : 0));
    if (e >= 31)
        return uint16_t(sign | 0x7c00);
    if (e <= 0)
    {
        if (e < -10)
            return uint16_t(sign);
        m |= 0x800000;
        const int      shift = 14 - e;
        uint32_t       h     = m >> shift;
        const uint32_t rem   = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        h += rem > half || (rem == half && (h & 1));
        return uint16_t(sign | h);
    }
    uint32_t       h   = (uint32_t(e) << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fff;
    h += rem > 0x1000 || (rem == 0x1000 && (h & 1));
    return uint16_t(sign | h);
}

float FromHalf(uint16_t h)
{
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    int            e    = (h >> 10) & 0x1f;
    uint32_t       m    = h & 0x3ff, x;
    if (e == 0)
    {
        if (!m)
            x = sign;
        else
        {
            e = 1;
            while (!(m & 0x400))
                m <<= 1, --e;
            x = sign | (uint32_t(e + 112) << 23) | ((m & 0x3ff) << 13);
        }
    }
    else if (e == 31)
        x = sign | 0x7f800000 | (m << 13);
    else
        x = sign | (uint32_t(e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

enum class Fmt { RGBA16F, RG16F, R32F, R32UI, RGBA32UI, RGBA8S, R8U, R11G11B10F };

struct Tex
{
    vkc::Image img;
    Fmt        fmt;
};

VkFormat VkFmt(Fmt f)
{
    switch (f)
    {
    case Fmt::RGBA16F: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case Fmt::RG16F: return VK_FORMAT_R16G16_SFLOAT;
    case Fmt::R32F: return VK_FORMAT_R32_SFLOAT;
    case Fmt::R32UI: return VK_FORMAT_R32_UINT;
    case Fmt::RGBA32UI: return VK_FORMAT_R32G32B32A32_UINT;
    case Fmt::RGBA8S: return VK_FORMAT_R8G8B8A8_SNORM;
    case Fmt::R8U: return VK_FORMAT_R8_UNORM;
    case Fmt::R11G11B10F: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    }
    return VK_FORMAT_UNDEFINED;
}

uint32_t TexelBytes(Fmt f)
{
    switch (f)
    {
    case Fmt::RGBA16F: return 8;
    case Fmt::RGBA32UI: return 16;
    case Fmt::R8U: return 1;
    default: return 4;
    }
}

// Decoded components of one texel, for comparisons and previews.
void Decode(Fmt f, const uint8_t* p, float out[4], int& n)
{
    auto h = [&](int i) { uint16_t v; std::memcpy(&v, p + 2 * i, 2); return FromHalf(v); };
    switch (f)
    {
    case Fmt::RGBA16F: n = 4; for (int i = 0; i < 4; ++i) out[i] = h(i); break;
    case Fmt::RG16F: n = 2; out[0] = h(0), out[1] = h(1); break;
    case Fmt::R32F: n = 1; std::memcpy(out, p, 4); break;
    case Fmt::R32UI: { n = 1; uint32_t v; std::memcpy(&v, p, 4); out[0] = float(v); break; }
    case Fmt::RGBA32UI: { n = 4; uint32_t v[4]; std::memcpy(v, p, 16); for (int i = 0; i < 4; ++i) out[i] = float(v[i]); break; }
    case Fmt::RGBA8S: n = 4; for (int i = 0; i < 4; ++i) out[i] = float(int8_t(p[i])); break;
    case Fmt::R8U: n = 1; out[0] = float(p[0]); break;
    case Fmt::R11G11B10F:
    {
        n = 3;
        uint32_t v;
        std::memcpy(&v, p, 4);
        const uint32_t bits[3] = {v & 0x7ff, (v >> 11) & 0x7ff, v >> 22}, mant[3] = {6, 6, 5};
        for (int i = 0; i < 3; ++i)
            out[i] = FromHalf(uint16_t(bits[i] << (10 - mant[i])));
        break;
    }
    }
}

// Per-output difference statistics. Integer formats count codes, float formats relative error.
struct Diff
{
    size_t values = 0, differ = 0, over = 0;
    double maxAbs = 0;
    int    x = -1, y = -1;
    float  a = 0, b = 0;
};

Diff Compare(Fmt f, const std::vector<uint8_t>& A, const std::vector<uint8_t>& B, uint32_t w, uint32_t h, double tol)
{
    Diff d;
    const uint32_t tb = TexelBytes(f);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
        {
            float a[4], b[4];
            int   n;
            Decode(f, &A[(size_t(y) * w + x) * tb], a, n);
            Decode(f, &B[(size_t(y) * w + x) * tb], b, n);
            for (int c = 0; c < n; ++c)
            {
                ++d.values;
                if (a[c] == b[c] || (std::isnan(a[c]) && std::isnan(b[c])))
                    continue;
                const double e = std::fabs(double(a[c]) - b[c]);
                const double r = e / std::max(1e-3, std::max(std::fabs(double(a[c])), std::fabs(double(b[c]))));
                ++d.differ;
                d.over += (tol >= 1 ? e : r) > tol;
                if (e > d.maxAbs)
                    d.maxAbs = e, d.x = int(x), d.y = int(y), d.a = a[c], d.b = b[c];
            }
        }
    return d;
}

enum Kind { SrvTex, UavTex, SrvBuf, UavBuf, Cb };

struct Slot
{
    const char* res;
    Kind        kind;
    uint32_t    n;
};

struct Pass
{
    const char*       name;  // file stem of ffx_nss_<name>.{hlsl,glsl}
    std::vector<Slot> slots;
};

// The NSS_BIND_* tables of the GLSL wrappers; the HLSL wrappers use the same numbers.
const Pass kDepthScatter = {"depth_scatter", {{"depth", SrvTex, 1}, {"motion", SrvTex, 2}, {"depth_tm1", UavTex, 3}, {"cb", Cb, 4}}};
const Pass kPreprocess   = {"pre_process",
                            {{"colour", SrvTex, 0},
                             {"depth", SrvTex, 1},
                             {"motion", SrvTex, 2},
                             {"history", SrvTex, 3},
                             {"feedback", SrvTex, 4},
                             {"depth_tm1", SrvTex, 5},
                             {"luma_tm1", SrvTex, 6},
                             {"disocclusion", SrvTex, 7},
                             {"tensor", UavBuf, 8},
                             {"luma", UavTex, 9},
                             {"nearest", UavTex, 10},
                             {"depth_tm1", UavTex, 11},
                             {"cb", Cb, 12}}};
const Pass kOffsetLut    = {"generate_offset_lut", {{"offset_lut", UavTex, 0}, {"cb", Cb, 11}}};
const Pass kPostprocess  = {"post_process",
                            {{"colour", SrvTex, 0},
                             {"motion", SrvTex, 1},
                             {"history", SrvTex, 2},
                             {"kpn", SrvBuf, 3},
                             {"feedback", SrvTex, 4},
                             {"nearest", SrvTex, 5},
                             {"offset_lut", SrvTex, 6},
                             {"output", UavTex, 7},
                             {"history_out", UavTex, 8},
                             {"cb", Cb, 9}}};
const Pass kDebugView    = {"debug_view",
                            {{"colour", SrvTex, 0},
                             {"motion", SrvTex, 1},
                             {"history", SrvTex, 2},
                             {"depth", SrvTex, 3},
                             {"depth_tm1", SrvTex, 4},
                             {"luma_tm1", SrvTex, 5},
                             {"nearest", SrvTex, 6},
                             {"feedback", SrvTex, 7},
                             {"disocclusion", SrvTex, 8},
                             {"luma", SrvTex, 9},
                             {"tensor", UavBuf, 10},
                             {"debug", UavTex, 11},
                             {"cb", Cb, 12}}};

struct Bound
{
    Tex*         tex = nullptr;
    vkc::Buffer* buf = nullptr;
};
using Bindings = std::map<std::string, Bound>;

struct Frame
{
    vkc::Context& vk;
    std::vector<std::string> permutation;
    std::map<std::string, vkc::Pipeline> pipelines;
    vkc::Buffer cb;
    bool        ok = true;

    VkDescriptorType Type(Kind k)
    {
        switch (k)
        {
        case SrvTex: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        case UavTex: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        case Cb: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        default: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
    }

    uint32_t Binding(const Slot& s, bool glsl)
    {
        if (glsl)
            return s.n;
        switch (s.kind)
        {
        case SrvTex:
        case SrvBuf: return vkc::kShiftT + s.n;
        case UavTex:
        case UavBuf: return vkc::kShiftU + s.n;
        default: return vkc::kShiftB + s.n;
        }
    }

    vkc::Pipeline* Get(const Pass& p, bool glsl, std::string& err)
    {
        std::string key = std::string("nss_") + p.name + (glsl ? "_glsl" : "_hlsl");
        for (const std::string& d : permutation)
            key += "_" + d.substr(d.find('=') + 1);
        auto it = pipelines.find(key);
        if (it != pipelines.end())
            return &it->second;
        std::vector<std::pair<uint32_t, VkDescriptorType>> layout;
        for (const Slot& s : p.slots)
            if (std::none_of(layout.begin(), layout.end(), [&](auto& l) { return l.first == Binding(s, glsl); }))
                layout.push_back({Binding(s, glsl), Type(s.kind)});
        layout.push_back({glsl ? 1000u : vkc::kShiftS + 0, VK_DESCRIPTOR_TYPE_SAMPLER});
        layout.push_back({glsl ? 1001u : vkc::kShiftS + 1, VK_DESCRIPTOR_TYPE_SAMPLER});
        std::vector<uint32_t> spirv;
        std::vector<std::string> defines = permutation;
        bool compiled;
        if (glsl)
        {
            for (const char* d : {"FFX_GLSL=1", "FFX_GPU=1", "FFX_HALF=1", "QUANTIZED=1", "NSS_SUPPORT_TENSOR=0", "FFX_8_BIT_TYPES=1",
                                  "OUTPUT_IMG_FORMAT=rgba16f"})
                defines.push_back(d);
            compiled = vkc::CompileGlsl(kGlsl + "/ffx_nss_" + p.name + ".glsl", defines, {kGpu}, key, spirv, err);
        }
        else
            compiled = vkc::CompileHlsl(kHlsl + "/ffx_nss_" + p.name + ".hlsl", "main", "cs_6_2", defines, {kHlsl}, key, spirv, err);
        vkc::Pipeline pl;
        if (!compiled || !vk.CreatePipeline(spirv, "main", key, layout, pl, err))
            return nullptr;
        return &(pipelines[key] = pl);
    }

    bool Dispatch(const Pass& p, bool glsl, const Bindings& b, uint32_t gx, uint32_t gy, std::string& err)
    {
        vkc::Pipeline* pl = Get(p, glsl, err);
        if (!pl)
            return false;
        std::map<uint32_t, vkc::Resource> res;
        for (const Slot& s : p.slots)
        {
            const Bound& r = b.at(s.res);
            vkc::Resource x;
            x.type = Type(s.kind);
            if (s.kind == Cb)
                x.buffer = &cb, x.range = sizeof(NssConstants);
            else if (r.buf)
                x.buffer = r.buf;
            else
                x.image = &r.tex->img;
            res[Binding(s, glsl)] = x;
        }
        vkc::Resource smp;
        smp.type                        = VK_DESCRIPTOR_TYPE_SAMPLER;
        smp.sampler                     = vk.PointClampSampler();
        res[glsl ? 1000u : vkc::kShiftS] = smp;
        smp.sampler                     = vk.LinearClampSampler();
        res[glsl ? 1001u : vkc::kShiftS + 1] = smp;
        return vk.Dispatch(*pl, res, gx, gy, 1, err);
    }

    std::vector<uint8_t> Read(const Bound& r)
    {
        if (r.buf)
        {
            const auto* p = static_cast<const uint8_t*>(r.buf->mapped);
            return std::vector<uint8_t>(p, p + r.buf->size);
        }
        std::vector<uint8_t> v(size_t(r.tex->img.width) * r.tex->img.height * r.tex->img.texelBytes);
        vk.Download(r.tex->img, v.data());
        return v;
    }

    void Write(const Bound& r, const std::vector<uint8_t>& v)
    {
        if (r.buf)
            std::memcpy(r.buf->mapped, v.data(), v.size());
        else
            vk.Upload(r.tex->img, v.data());
    }

    // Runs the HLSL pass on `main`, then Arm's GLSL on the same inputs with every output
    // replaced by its entry in `shadow` (preloaded with the outputs' prior contents), and
    // compares each output.
    bool Run(const Pass& p, const Bindings& main, const Bindings& shadow, uint32_t gx, uint32_t gy, std::string& err)
    {
        std::vector<std::string> outs;
        for (const Slot& s : p.slots)
            if ((s.kind == UavTex || s.kind == UavBuf) && shadow.count(s.res))
                outs.push_back(s.res);
        std::map<std::string, std::vector<uint8_t>> before;
        for (const std::string& o : outs)
            before[o] = Read(main.at(o));
        if (!Dispatch(p, false, main, gx, gy, err))
            return false;
        Bindings g = main;
        for (const std::string& o : outs)
        {
            g[o] = shadow.at(o);
            Write(g[o], before[o]);
        }
        if (!Dispatch(p, true, g, gx, gy, err))
            return false;
        for (const std::string& o : outs)
        {
            const Bound& m = main.at(o);
            const Fmt    f = m.buf ? Fmt::RGBA8S : m.tex->fmt;
            const uint32_t w = m.buf ? uint32_t(m.buf->size / 4) : m.tex->img.width, h = m.buf ? 1 : m.tex->img.height;
            // HLSL and GLSL compilers order and contract fp16 math differently: codes may land one
            // step apart at rounding ties and fp16 values up to 2% apart; nothing else is allowed.
            const bool   codes = f == Fmt::R8U || f == Fmt::RGBA8S;
            const double tol   = codes ? 1 : (f == Fmt::R32UI || f == Fmt::RGBA32UI) ? 0 : 2e-2;
            const Diff   d     = Compare(f, Read(m), Read(g.at(o)), w, h, tol);
            if (const char* dump = std::getenv("RDNU_PASS_DUMP"))
                for (int k = 0; k < 2; ++k)
                    if (FILE* fp = std::fopen((std::string(dump) + "/" + p.name + "_" + o + (k ? "_glsl" : "_hlsl") + ".bin").c_str(), "wb"))
                    {
                        auto v = Read(k ? g.at(o) : m);
                        std::fwrite(v.data(), 1, v.size(), fp);
                        std::fclose(fp);
                    }
            const bool pass = d.over == 0 && d.differ * 10 <= d.values;
            std::printf("    %-20s %-12s %s  %zu/%zu differ, max %.3g", p.name, o.c_str(), pass ? "ok  " : "FAIL", d.differ, d.values, d.maxAbs);
            if (d.differ)
                std::printf(" at (%d,%d) hlsl %g glsl %g", d.x, d.y, d.a, d.b);
            std::printf("\n");
            ok &= pass;
        }
        return true;
    }
};

struct Rgb
{
    std::vector<float> v;  // w*h*3, linear
    uint32_t           w = 0, h = 0;
};

double Psnr(const Rgb& a, const Rgb& b, float exposure)
{
    double se = 0;
    for (size_t i = 0; i < a.v.size(); ++i)
    {
        const double x = a.v[i] * exposure / (1 + a.v[i] * exposure), y = b.v[i] * exposure / (1 + b.v[i] * exposure);
        se += (x - y) * (x - y);
    }
    return 10 * std::log10(1.0 / std::max(1e-12, se / a.v.size()));
}

void WritePpm(const std::string& path, const std::vector<const Rgb*>& row, float exposure)
{
    uint32_t w = 0, h = row[0]->h;
    for (const Rgb* r : row)
        w += r->w;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f)
        return;
    std::fprintf(f, "P6\n%u %u\n255\n", w, h);
    for (uint32_t y = 0; y < h; ++y)
        for (const Rgb* r : row)
            for (uint32_t x = 0; x < r->w; ++x)
                for (int c = 0; c < 3; ++c)
                {
                    float v = r->v[(size_t(y) * r->w + x) * 3 + c] * exposure;
                    v       = std::pow(v / (1 + v), 1 / 2.2f);
                    std::fputc(int(std::clamp(v, 0.f, 1.f) * 255 + 0.5f), f);
                }
    std::fclose(f);
}

uint32_t Gcd(uint32_t a, uint32_t b) { return b ? Gcd(b, a % b) : a; }
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
        return std::printf("usage: test_passes_vk <golden dir> [--frames N] [--scale S] [--sharpen stops] [--wmma] [--out dir]\n"), 1;
    std::string dir = argv[1], outDir = "/tmp/rdnu_passes";
    uint32_t    frames = 4;
    double      scale  = 2;
    float       sharpen = -1;
    bool        wmma   = false;
    for (int i = 2; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--frames") && i + 1 < argc)
            frames = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--scale") && i + 1 < argc)
            scale = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--sharpen") && i + 1 < argc)
            sharpen = float(std::atof(argv[++i]));
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc)
            outDir = argv[++i];
        else if (!std::strcmp(argv[i], "--wmma"))
            wmma = true;
    }

    std::string err;
    std::vector<uint8_t> manifestBytes, blob;
    if (!rdnut::ReadBytes(dir + "/nss.rdnm", manifestBytes) || !rdnut::ReadBytes(dir + "/nss_w8.bin", blob))
        return std::printf("cannot read nss.rdnm / nss_w8.bin in %s\n", dir.c_str()), 1;
    rdnu::Manifest manifest;
    if (!rdnu::ParseManifest(manifestBytes.data(), manifestBytes.size(), manifest, err))
        return std::printf("manifest: %s\n", err.c_str()), 1;
    std::map<std::string, rdnut::Tensor> seq;
    if (!rdnut::Load(dir + "/nss_frames.rdnut", seq, err))
        return std::printf("%s (run runtime/tools/nss_pass_frames.py)\n", err.c_str()), 1;

    const rdnut::Tensor &colour = seq.at("colour"), &depth = seq.at("depth"), &motion = seq.at("motion");
    const rdnut::Tensor &jitter = seq.at("jitter"), &exposure = seq.at("exposure"), &camera = seq.at("camera");
    const rdnut::Tensor& truth = seq.at("truth");
    frames = std::min(frames, colour.dims[0]);
    const uint32_t W = colour.dims[2], H = colour.dims[1];
    const bool     x2 = scale == 2;
    const uint32_t DW = uint32_t(std::lround(W * scale)), DH = uint32_t(std::lround(H * scale));
    const uint32_t GW = rdnu::RoundUp(W, 8), GH = rdnu::RoundUp(H, 8);
    const uint32_t SW = W / 2, SH = H / 2, KW = GW / 4, KH = GH / 4;
    const bool     haveTruth = x2 && truth.dims[1] == DH && truth.dims[2] == DW;

    uint32_t modX = 2, modY = 2, redX = 1, redY = 1;
    if (!x2)
    {
        const uint32_t gx = Gcd(DW, W), gy = Gcd(DH, H);
        modX = DW / gx, modY = DH / gy, redX = W / gx, redY = H / gy;
    }

    rdnu::PlanConfig cfg;
    cfg.maxWidth  = GW;
    cfg.maxHeight = GH;
    cfg.useWmma   = wmma;
    rdnu::Plan plan;
    if (!plan.Build(manifest, cfg, err))
        return std::printf("plan: %s\n", err.c_str()), 1;

    vkc::Context vk;
    if (!vk.Init(err))
        return std::printf("vulkan: %s\n", err.c_str()), 1;
    std::printf("device: %s   render %ux%u -> %ux%u (%s)   network %ux%u %s   %u frames\n", vk.DeviceName().c_str(), W, H, DW, DH,
                x2 ? "static 2x filter" : "dynamic offset LUT", GW, GH, wmma ? "DP4a + WMMA (emulated)" : "DP4a", frames);

    NetRunner net{vk, manifest, plan};
    net.shaderDir = kSrc + "/runtime/shaders/net";
    if (!net.Allocate(blob, GW, GH, err))
        return std::printf("%s\n", err.c_str()), 1;

    auto make = [&](uint32_t w, uint32_t h, Fmt f) { return Tex{vk.CreateImage(w, h, VkFmt(f), TexelBytes(f)), f}; };
    Tex tColour = make(W, H, Fmt::RGBA16F), tDepth = make(W, H, Fmt::R32F), tMotion = make(W, H, Fmt::RG16F);
    Tex tDepthTm1 = make(SW, SH, Fmt::R32UI), sDepthTm1 = make(SW, SH, Fmt::R32UI);
    Tex tLuma[2] = {make(GW, GH, Fmt::RGBA8S), make(GW, GH, Fmt::RGBA8S)}, sLuma = make(GW, GH, Fmt::RGBA8S);
    Tex tNearest = make(GW, GH, Fmt::R8U), sNearest = make(GW, GH, Fmt::R8U);
    Tex tDisocc  = make(SW, SH, Fmt::R8U);
    Tex tLut = make((x2 ? 1 : 3) * modX, modY, Fmt::RGBA32UI), sLut = make((x2 ? 1 : 3) * modX, modY, Fmt::RGBA32UI);
    Tex tHistory[2] = {make(DW, DH, Fmt::RGBA16F), make(DW, DH, Fmt::RGBA16F)}, sHistory = make(DW, DH, Fmt::RGBA16F);
    Tex tOutput = make(DW, DH, Fmt::RGBA16F), sOutput = make(DW, DH, Fmt::RGBA16F);
    const Fmt debugFmt = vk.StorageFormat(VkFmt(Fmt::R11G11B10F)) ? Fmt::R11G11B10F : Fmt::RGBA16F;
    Tex tDebug = make(DW, DH, debugFmt), sDebug = make(DW, DH, debugFmt);
    Tex tFeedback{net.temporal, Fmt::RGBA8S};
    Tex tSharp = make(DW, DH, Fmt::RGBA16F), tExposure = make(1, 1, Fmt::R32F);
    vkc::Buffer   rcasCb = vk.CreateBuffer(256);
    vkc::Pipeline rcas;
    if (sharpen >= 0 &&
        !vk.CreatePipeline(kHlsl + "/ffx_rcas_pass.hlsl", "main", "cs_6_2", {}, {}, "rdnu_rcas",
                           {{vkc::kShiftB, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}, {vkc::kShiftT, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE},
                            {vkc::kShiftT + 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE}, {vkc::kShiftU, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}},
                           rcas, err))
        return std::printf("%s\n", err.c_str()), 1;
    vkc::Buffer sTensor = vk.CreateBuffer(plan.InputBytes());

    Frame fr{vk};
    fr.cb          = vk.CreateBuffer(256);
    fr.permutation = {"NSS_SHADER_QUALITY_MODE=0", std::string("SCALE_PRESET_MODE=") + (x2 ? "1" : "0"), "REVERSE_Z=0", "MANAGE_HISTORY=1"};
    for (const rdnu::TensorRecord& t : manifest.tensors)
        if (rdnu::TensorKind(t.kind) == rdnu::TensorKind::Input)
        {
            char inputScale[64];
            std::snprintf(inputScale, sizeof(inputScale), "RDNU_INPUT_SCALE=%.17g", double(t.scale));
            fr.permutation.push_back(inputScale);
        }

    auto zero = [&](Tex& t) { vk.Upload(t.img, std::vector<uint8_t>(size_t(t.img.width) * t.img.height * t.img.texelBytes).data()); };
    for (Tex* t : {&tLuma[0], &tLuma[1], &tNearest, &tDisocc, &tFeedback, &tHistory[0], &tHistory[1], &tLut, &tOutput, &tDebug})
        zero(*t);

    NssConstants c{};
    Rgb out, gt, bilinear;
    for (uint32_t t = 0; t < frames; ++t)
    {
        const size_t np = size_t(W) * H;
        std::vector<uint16_t> col(np * 4), mv(np * 2);
        for (size_t i = 0; i < np * 4; ++i)
            col[i] = ToHalf(colour.data[t * np * 4 + i]);
        for (size_t i = 0; i < np * 2; ++i)
            mv[i] = ToHalf(motion.data[t * np * 2 + i]);
        vk.Upload(tColour.img, col.data());
        vk.Upload(tDepth.img, &depth.data[t * np]);
        vk.Upload(tMotion.img, mv.data());
        vk.Upload(tDepthTm1.img, std::vector<uint32_t>(size_t(SW) * SH, 0x7fffffff).data());

        const float jx = jitter.data[t * 2], jy = jitter.data[t * 2 + 1], e = exposure.data[t];
        const bool  reset = t == 0;
        NssConstants prev = c;
        c = NssConstants{};
        {
            // setupDeviceDepthToViewSpaceDepthParams in ffx_nss.cpp, depth not inverted
            const float zn = camera.data[t * 4], zf = camera.data[t * 4 + 1], fov = camera.data[t * 4 + 2];
            const bool  infinite = camera.data[t * 4 + 3] != 0;
            const float lo = std::min(zn, zf), hi = std::max(zn, zf), q = hi / (lo - hi);
            const float cot = std::cos(0.5f * fov) / std::sin(0.5f * fov);
            c.deviceToViewDepth[0] = -(infinite ? -1.0f - FLT_EPSILON : q);
            c.deviceToViewDepth[1] = infinite ? -lo - FLT_EPSILON : q * lo;
            c.deviceToViewDepth[2] = 1.0f / (cot / (float(W) / float(H)));
            c.deviceToViewDepth[3] = 1.0f / cot;
        }
        const float jo[4] = {jx, jy, jx / W, jy / H};
        std::memcpy(c.jitterOffset, jo, 16);
        std::memcpy(c.jitterOffsetTm1, reset ? jo : prev.jitterOffset, 16);
        c.scaleFactor[0] = float(DW) / W, c.scaleFactor[1] = float(DH) / H;
        c.scaleFactor[2] = float(W) / DW, c.scaleFactor[3] = float(H) / DH;
        c.outputDims[0] = DW, c.outputDims[1] = DH, c.inputDims[0] = W, c.inputDims[1] = H;
        c.invOutputDims[0] = 1.f / DW, c.invOutputDims[1] = 1.f / DH, c.invInputDims[0] = 1.f / W, c.invInputDims[1] = 1.f / H;
        c.depthTm1Size[0] = SW, c.depthTm1Size[1] = SH, c.invDepthTm1Size[0] = 1.f / SW, c.invDepthTm1Size[1] = 1.f / SH;
        c.inputTensorSize[0] = GW, c.inputTensorSize[1] = GH, c.inputTensorSizeRcp[0] = 1.f / GW, c.inputTensorSizeRcp[1] = 1.f / GH;
        c.kpnDimension[0] = KW, c.kpnDimension[1] = KH;
        c.motionVectorScale[0] = c.motionVectorScale[1] = 1;
        c.paddingScale[0] = float(W) / GW, c.paddingScale[1] = float(H) / GH;
        const float diag = std::sqrt(float(W * W + H * H));
        c.depthClipRequiredSepScale = 1.37e-05f * std::sqrt(c.deviceToViewDepth[2] * c.deviceToViewDepth[2] +
                                                            c.deviceToViewDepth[3] * c.deviceToViewDepth[3] + 1) * diag;
        c.depthClipPower = 1 + 2 * std::min(std::max(diag / 2202.9071700822983f, 0.f), 1.f);
        c.kpnScale[0] = float(KW) / GW, c.kpnScale[1] = float(KH) / GH;
        c.notHistoryReset = reset ? 0.f : 1.f;
        c.exposure[0] = e, c.exposure[1] = 1 / e;
        c.indexModulo[0] = modX, c.indexModulo[1] = modY, c.reducedInputModulo[0] = redX, c.reducedInputModulo[1] = redY;
        if (x2)
        {
            auto tile = [](float j, float s, int m) {
                int o = (int(std::floor((j + 0.5f) * s)) - int(std::floor(0.5f * s))) % m;
                return o < 0 ? o + m : o;
            };
            c.lutOffset[0] = tile(jx, c.scaleFactor[0], int(modX)), c.lutOffset[1] = tile(jy, c.scaleFactor[1], int(modY));
        }
        std::memcpy(fr.cb.mapped, &c, sizeof(c));

        Tex& lumaSrv = tLuma[t & 1];
        Tex& lumaUav = tLuma[(t + 1) & 1];
        Tex& histSrv = tHistory[t & 1];
        Tex& histUav = tHistory[(t + 1) & 1];
        Bindings main = {{"colour", {&tColour}},     {"depth", {&tDepth}},       {"motion", {&tMotion}},       {"depth_tm1", {&tDepthTm1}},
                         {"history", {&histSrv}},    {"feedback", {&tFeedback}}, {"luma_tm1", {&lumaSrv}},     {"disocclusion", {&tDisocc}},
                         {"tensor", {nullptr, &net.input}}, {"luma", {&lumaUav}}, {"nearest", {&tNearest}},    {"kpn", {nullptr, &net.kpn}},
                         {"offset_lut", {&tLut}},    {"output", {&tOutput}},     {"history_out", {&histUav}},  {"debug", {&tDebug}},
                         {"cb", {}}};
        Bindings shadow = {{"depth_tm1", {&sDepthTm1}}, {"tensor", {nullptr, &sTensor}}, {"luma", {&sLuma}},
                           {"nearest", {&sNearest}},    {"offset_lut", {&sLut}},         {"output", {&sOutput}},
                           {"history_out", {&sHistory}}, {"debug", {&sDebug}}};

        std::printf("frame %u%s\n", t, reset ? " (reset)" : "");
        auto groups = [](uint32_t n, uint32_t g) { return (n + g - 1) / g; };
        if (!fr.Run(kDepthScatter, main, shadow, groups(SW, 16), groups(SH, 16), err) ||
            !fr.Run(kPreprocess, main, shadow, groups(GW, 16), groups(GH, 16), err) || !net.Run(err) ||
            (!x2 && !fr.Run(kOffsetLut, main, shadow, groups(modX * modY, 64), 1, err)) ||
            !fr.Run(kPostprocess, main, shadow, groups(DW, 16), groups(DH, 16), err) ||
            !fr.Run(kDebugView, main, shadow, groups(DW, 16), groups(DH, 16), err))
            return std::printf("%s\n", err.c_str()), 1;

        if (sharpen >= 0)
        {
            struct { uint32_t w, h; float sharpness, pad; } rc = {DW, DH, std::exp2(-sharpen), 0};
            std::memcpy(rcasCb.mapped, &rc, sizeof(rc));
            vk.Upload(tExposure.img, &e);
            vkc::Resource cbr{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &rcasCb, 0, 256}, in, ex, outr;
            in.type = ex.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, in.image = &tOutput.img, ex.image = &tExposure.img;
            outr.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, outr.image = &tSharp.img;
            if (!vk.Dispatch(rcas, {{vkc::kShiftB, cbr}, {vkc::kShiftT, in}, {vkc::kShiftT + 1, ex}, {vkc::kShiftU, outr}}, groups(DW, 8),
                             groups(DH, 8), 1, err))
                return std::printf("%s\n", err.c_str()), 1;
        }
        std::vector<uint8_t> o = fr.Read({sharpen >= 0 ? &tSharp : &tOutput});
        out = {std::vector<float>(size_t(DW) * DH * 3), DW, DH};
        for (size_t i = 0; i < size_t(DW) * DH; ++i)
            for (int k = 0; k < 3; ++k)
            {
                uint16_t h;
                std::memcpy(&h, &o[i * 8 + k * 2], 2);
                out.v[i * 3 + k] = FromHalf(h);
            }
        if (haveTruth)
        {
            gt = bilinear = {std::vector<float>(size_t(DW) * DH * 3), DW, DH};
            for (uint32_t y = 0; y < DH; ++y)
                for (uint32_t x = 0; x < DW; ++x)
                    for (int k = 0; k < 3; ++k)
                    {
                        gt.v[(size_t(y) * DW + x) * 3 + k] = truth.data[((t * size_t(DH) + y) * DW + x) * 4 + k];
                        float sx = std::clamp((x + 0.5f) / 2 - 0.5f, 0.f, W - 1.f), sy = std::clamp((y + 0.5f) / 2 - 0.5f, 0.f, H - 1.f);
                        uint32_t x0 = uint32_t(sx), y0 = uint32_t(sy), x1 = std::min(x0 + 1, W - 1), y1 = std::min(y0 + 1, H - 1);
                        float fx = sx - x0, fy = sy - y0;
                        auto at = [&](uint32_t xx, uint32_t yy) { return colour.data[((t * size_t(H) + yy) * W + xx) * 4 + k]; };
                        bilinear.v[(size_t(y) * DW + x) * 3 + k] =
                            (at(x0, y0) * (1 - fx) + at(x1, y0) * fx) * (1 - fy) + (at(x0, y1) * (1 - fx) + at(x1, y1) * fx) * fy;
                    }
            std::printf("    psnr (tonemapped)   nss%s %.2f dB   bilinear %.2f dB\n", sharpen >= 0 ? "+rcas" : "", Psnr(out, gt, e),
                        Psnr(bilinear, gt, e));
        }
    }

    std::string cmd = "mkdir -p '" + outDir + "'";
    if (std::system(cmd.c_str()) == 0)
    {
        const float e = exposure.data[frames - 1];
        if (haveTruth)
            WritePpm(outDir + "/compare.ppm", {&bilinear, &out, &gt}, e);
        else
            WritePpm(outDir + "/output.ppm", {&out}, e);
        std::printf("preview in %s%s\n", outDir.c_str(), haveTruth ? " (bilinear | nss | ground truth)" : "");
    }

    for (auto& kv : fr.pipelines)
        vk.Destroy(kv.second);
    for (Tex* t : {&tColour, &tDepth, &tMotion, &tDepthTm1, &sDepthTm1, &tLuma[0], &tLuma[1], &sLuma, &tNearest, &sNearest, &tDisocc,
                   &tLut, &sLut, &tHistory[0], &tHistory[1], &sHistory, &tOutput, &sOutput, &tDebug, &sDebug})
        vk.Destroy(t->img);
    if (sharpen >= 0)
        vk.Destroy(rcas);
    vk.Destroy(tSharp.img), vk.Destroy(tExposure.img);
    vk.Destroy(sTensor), vk.Destroy(fr.cb), vk.Destroy(rcasCb);
    net.Release();
    std::printf("%s\n", fr.ok ? "PASS" : "FAIL");
    return fr.ok ? 0 : 2;
}
