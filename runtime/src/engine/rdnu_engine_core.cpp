// rdnu_engine_core.cpp - manifest parsing, arena layout and dispatch planning. No graphics API.
#include "rdnu_engine_core.h"

#include <algorithm>
#include <cstring>

namespace rdnu
{
namespace
{
constexpr uint32_t kMagic       = 0x4D4E4452u;  // "RDNM"
constexpr uint32_t kVersion     = 1;
constexpr uint32_t kTile        = 8;            // DP4a output tile
constexpr uint64_t kArenaAlign  = 256;

struct Header
{
    uint32_t magic, version, tensorCount, layerCount, blobBytes, reserved[3];
};
static_assert(sizeof(Header) == 32, "manifest header");

// A channel range of one tensor, for hazard tracking.
struct Range
{
    uint32_t tensor, first, count;
    bool Overlaps(const Range& o) const
    {
        return tensor == o.tensor && first < o.first + o.count && o.first < first + count;
    }
};

bool AnyOverlap(const std::vector<Range>& set, const Range& r)
{
    for (const Range& s : set)
        if (s.Overlaps(r))
            return true;
    return false;
}

Resource ResourceOf(TensorKind k)
{
    switch (k)
    {
    case TensorKind::Input: return Resource::Input;
    case TensorKind::Kpn: return Resource::Kpn;
    case TensorKind::Temporal: return Resource::Temporal;
    default: return Resource::Arena;
    }
}
}  // namespace

bool ParseManifest(const void* data, size_t size, Manifest& out, std::string& error)
{
    if (size < sizeof(Header))
        return error = "manifest too small", false;
    Header h;
    std::memcpy(&h, data, sizeof(h));
    if (h.magic != kMagic || h.version != kVersion)
        return error = "not an RDNM v1 manifest", false;
    size_t need = sizeof(Header) + size_t(h.tensorCount) * sizeof(TensorRecord) + size_t(h.layerCount) * sizeof(LayerRecord);
    if (size < need)
        return error = "manifest truncated", false;

    const auto* p = static_cast<const uint8_t*>(data) + sizeof(Header);
    out.tensors.resize(h.tensorCount);
    std::memcpy(out.tensors.data(), p, h.tensorCount * sizeof(TensorRecord));
    p += h.tensorCount * sizeof(TensorRecord);
    out.layers.resize(h.layerCount);
    std::memcpy(out.layers.data(), p, h.layerCount * sizeof(LayerRecord));
    out.blobBytes = h.blobBytes;

    for (const LayerRecord& l : out.layers)
    {
        if (l.inTensor >= h.tensorCount || l.outTensor >= h.tensorCount)
            return error = std::string("layer ") + l.name + ": tensor index out of range", false;
        const TensorRecord& ti = out.tensors[l.inTensor];
        const TensorRecord& to = out.tensors[l.outTensor];
        if (l.inChannelOffset + l.cin > ti.channels || l.outChannelOffset + l.cout > to.channels)
            return error = std::string("layer ") + l.name + ": channel range exceeds tensor", false;
        if (l.cin % 4 || l.cout % 4 || l.ocb == 0 || l.cout % l.ocb || l.inChannelOffset % 4 || l.outChannelOffset % 4)
            return error = std::string("layer ") + l.name + ": channels must be multiples of 4", false;
        if (l.stride != 1 && l.stride != 2)
            return error = std::string("layer ") + l.name + ": stride must be 1 or 2", false;
        uint32_t expect = ti.level + (l.stride == 2 ? 1 : 0) - (l.upsample ? 1 : 0);
        if (to.level != expect)
            return error = std::string("layer ") + l.name + ": resolution levels inconsistent", false;
        uint64_t end = std::max({uint64_t(l.offWDp4a), uint64_t(l.offWOhwi), uint64_t(l.offBias), uint64_t(l.offScale)});
        if (end >= h.blobBytes)
            return error = std::string("layer ") + l.name + ": blob offset out of range", false;
    }
    return true;
}

bool KernelKey::operator==(const KernelKey& o) const
{
    return kernel == o.kernel && cin == o.cin && cout == o.cout && ocb == o.ocb && stride == o.stride &&
           upsample == o.upsample && inBordered == o.inBordered && outKind == o.outKind && act == o.act;
}

std::string KernelKey::Name() const
{
    if (kernel == Kernel::Wmma)
        return "wmma_c" + std::to_string(cin) + "_o" + std::to_string(cout);
    return "dp4a_c" + std::to_string(cin) + "_o" + std::to_string(ocb) + "_s" + std::to_string(stride) + "_u" +
           std::to_string(upsample) + "_b" + std::to_string(inBordered) + "_k" + std::to_string(outKind) + "_a" +
           std::to_string(act);
}

std::vector<std::string> KernelKey::Defines() const
{
    if (kernel == Kernel::Wmma)
        return {"CIN=" + std::to_string(cin), "COUT=" + std::to_string(cout)};
    return {"CIN=" + std::to_string(cin),       "OCB=" + std::to_string(ocb),         "STRIDE=" + std::to_string(stride),
            "UPSAMPLE=" + std::to_string(upsample), "IN_BORDERED=" + std::to_string(inBordered),
            "OUT_KIND=" + std::to_string(outKind), "ACT=" + std::to_string(act)};
}

const char* KernelKey::Source() const
{
    return kernel == Kernel::Wmma ? "nss_conv3x3_wmma.hlsl" : "nss_conv3x3_dp4a.hlsl";
}

KernelKey Plan::KeyFor(const LayerRecord& l, bool wmma) const
{
    const TensorRecord& ti = manifest_.tensors[l.inTensor];
    const TensorRecord& to = manifest_.tensors[l.outTensor];
    KernelKey k;
    k.cin        = l.cin;
    k.stride     = l.stride;
    k.upsample   = l.upsample;
    k.inBordered = TensorKind(ti.kind) == TensorKind::Arena ? 1 : 0;
    k.outKind    = TensorKind(to.kind) == TensorKind::Temporal ? uint32_t(OutKind::Temporal) : uint32_t(OutKind::Int8);
    k.act        = l.act;
    if (wmma)
    {
        k.kernel = Kernel::Wmma;
        k.cout   = l.cout;
    }
    else
    {
        k.kernel = Kernel::Dp4a;
        k.ocb    = l.ocb;
    }
    return k;
}

bool Plan::Build(const Manifest& manifest, const PlanConfig& config, std::string& error)
{
    if (config.maxWidth == 0 || config.maxHeight == 0 || config.maxWidth % 8 || config.maxHeight % 8)
        return error = "network size must be a non-zero multiple of 8", false;
    manifest_ = manifest;
    config_   = config;
    if (!config_.inputPitch)
        config_.inputPitch = config.maxWidth;
    if (!config_.kpnPitch)
        config_.kpnPitch = config.maxWidth / 4;

    tensors_.clear();
    arenaBytes_ = 0;
    for (const TensorRecord& t : manifest_.tensors)
    {
        TensorLayout L{};
        L.kind     = TensorKind(t.kind);
        L.channels = t.channels;
        L.level    = t.level;
        L.maxW     = config.maxWidth >> t.level;
        L.maxH     = config.maxHeight >> t.level;
        switch (L.kind)
        {
        case TensorKind::Arena:
            // One pixel of border on every side; rows padded so 16-pixel WMMA runs and 8x8
            // DP4a tiles never leave the allocation.
            L.border      = 1;
            L.pitchPixels = RoundUp(L.maxW, 16) + 2;
            L.rows        = RoundUp(L.maxH, 16) + 2;
            L.bytes       = uint64_t(L.pitchPixels) * L.rows * L.channels;
            L.offset      = arenaBytes_;
            arenaBytes_   = (arenaBytes_ + L.bytes + kArenaAlign - 1) / kArenaAlign * kArenaAlign;
            break;
        case TensorKind::Input:
            L.pitchPixels = config_.inputPitch;
            L.rows        = L.maxH;
            L.bytes       = uint64_t(L.pitchPixels) * L.rows * L.channels;
            break;
        case TensorKind::Kpn:
            L.pitchPixels = config_.kpnPitch;
            L.rows        = L.maxH;
            L.bytes       = uint64_t(L.pitchPixels) * L.rows * L.channels;
            break;
        case TensorKind::Temporal:
            L.pitchPixels = L.maxW;
            L.rows        = L.maxH;
            L.bytes       = 0;  // texture
            break;
        }
        tensors_.push_back(L);
    }
    return Update(config.maxWidth, config.maxHeight, error);
}

uint64_t Plan::InputBytes() const { return TensorOfKind(TensorKind::Input).bytes; }
uint64_t Plan::KpnBytes() const { return TensorOfKind(TensorKind::Kpn).bytes; }

const TensorLayout& Plan::TensorOfKind(TensorKind kind) const
{
    for (const TensorLayout& t : tensors_)
        if (t.kind == kind)
            return t;
    return tensors_.front();
}

bool Plan::Update(uint32_t width, uint32_t height, std::string& error)
{
    if (width == 0 || height == 0 || width % 8 || height % 8 || width > config_.maxWidth || height > config_.maxHeight)
        return error = "frame size must be a multiple of 8 within the planned maximum", false;
    width_  = width;
    height_ = height;
    for (TensorLayout& t : tensors_)
    {
        t.w = width >> t.level;
        t.h = height >> t.level;
    }

    // Byte address of valid pixel (0,0), channel `c`, of a tensor.
    auto base = [&](const TensorLayout& t, uint32_t c) -> uint64_t {
        uint64_t b = t.offset + c;
        if (t.border)
            b += (uint64_t(t.pitchPixels) + 1) * t.channels;
        return b;
    };

    dispatches_.clear();
    std::vector<Range> writes, reads;  // since the last barrier
    for (uint32_t i = 0; i < manifest_.layers.size(); ++i)
    {
        const LayerRecord&  l  = manifest_.layers[i];
        const TensorLayout& ti = tensors_[l.inTensor];
        const TensorLayout& to = tensors_[l.outTensor];
        const bool wmma = config_.useWmma && l.wmma && ti.kind == TensorKind::Arena && to.kind == TensorKind::Arena;

        Dispatch d{};
        d.layer = i;
        d.key   = KeyFor(l, wmma);
        d.in    = ResourceOf(ti.kind);
        d.out   = ResourceOf(to.kind);

        NetConstants& c = d.constants;
        uint64_t inBase  = base(ti, l.inChannelOffset);
        uint64_t outBase = base(to, l.outChannelOffset);
        if (inBase > 0xFFFFFFFFull || outBase + to.bytes > 0xFFFFFFFFull + 1ull)
            return error = "arena exceeds 4 GiB of byte addressing", false;
        c.inBase   = uint32_t(inBase);
        c.inPitch  = ti.pitchPixels * ti.channels;
        c.inPix    = ti.channels;
        c.inW      = ti.w;
        c.inH      = ti.h;
        c.outBase  = uint32_t(outBase);
        c.outPitch = to.pitchPixels * to.channels;
        c.outPix   = to.channels;
        c.outW     = to.w;
        c.outH     = to.h;
        c.outRing  = to.kind == TensorKind::Arena ? 1u : 0u;
        c.cout     = l.cout;
        c.wOff     = wmma ? l.offWOhwi : l.offWDp4a;
        c.biasOff  = l.offBias;
        c.scaleOff = l.offScale;
        c.lutOff   = l.offLut;
        c.zpLogit  = l.zpLogit;

        if (wmma)
        {
            uint32_t runs = 16 * WmmaPixelTiles(l.cout);
            d.groups[0]   = DivUp(RoundUp(to.w, 16), runs);
            d.groups[1]   = to.h;
        }
        else
        {
            d.groups[0] = DivUp(to.w, kTile);
            d.groups[1] = DivUp(to.h, kTile);
        }
        d.groups[2] = 1;
        c.groupsX   = d.groups[0];

        // RAW, WAR and WAW hazards on channel ranges since the last barrier.
        Range r{l.inTensor, l.inChannelOffset, l.cin};
        Range w{l.outTensor, l.outChannelOffset, l.cout};
        if (AnyOverlap(writes, r) || AnyOverlap(reads, w) || AnyOverlap(writes, w))
        {
            d.barrierBefore = true;
            writes.clear();
            reads.clear();
        }
        if (i == 0)
            d.barrierBefore = true;  // the input tensor was written by the pre-process pass
        reads.push_back(r);
        writes.push_back(w);
        dispatches_.push_back(d);
    }
    return true;
}

std::vector<KernelKey> Plan::RequiredKernels(bool includeWmma) const
{
    std::vector<KernelKey> keys;
    auto add = [&](const KernelKey& k) {
        if (std::find(keys.begin(), keys.end(), k) == keys.end())
            keys.push_back(k);
    };
    for (const LayerRecord& l : manifest_.layers)
    {
        add(KeyFor(l, false));
        const TensorLayout& ti = tensors_[l.inTensor];
        const TensorLayout& to = tensors_[l.outTensor];
        if (includeWmma && l.wmma && ti.kind == TensorKind::Arena && to.kind == TensorKind::Arena)
            add(KeyFor(l, true));
    }
    return keys;
}

}  // namespace rdnu
