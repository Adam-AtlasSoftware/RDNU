// rdnu_engine_core.h - platform-free planning for the NSS INT8 network.
//
// Parses the model manifest (nss.rdnm, written by runtime/tools/nss_export.py), lays out the
// activation arena, and turns every layer into a dispatch: kernel permutation, grid, root
// constants, resource views and the barriers between dependent layers. The DX12 recorder and
// the Vulkan test runner execute the same plan, so everything testable lives here.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rdnu
{

enum class TensorKind : uint32_t { Input = 0, Arena = 1, Kpn = 2, Temporal = 3 };
enum class Act : uint32_t { Relu = 0, Sigmoid = 1 };
enum class Kernel : uint32_t { Dp4a = 0, Wmma = 1 };
enum class OutKind : uint32_t { Int8 = 0, Temporal = 1 };

#pragma pack(push, 1)
struct TensorRecord
{
    char     name[16];
    uint32_t channels;
    uint32_t level;       // resolution = network input >> level
    uint32_t kind;        // TensorKind
    uint32_t reserved0;
    int32_t  zp;
    float    scale;
    uint32_t reserved1[2];
};

struct LayerRecord
{
    char     name[32];
    uint32_t inTensor;
    uint32_t inChannelOffset;
    uint32_t cin;
    uint32_t stride;
    uint32_t upsample;
    uint32_t outTensor;
    uint32_t outChannelOffset;
    uint32_t cout;
    uint32_t act;         // Act
    uint32_t ocb;         // DP4a output-channel block
    uint32_t wmma;        // eligible for the wave-matrix kernel
    int32_t  zpLogit;
    uint32_t offWDp4a;
    uint32_t offWOhwi;
    uint32_t offBias;
    uint32_t offScale;
    uint32_t offLut;
    uint32_t reserved[7];
};
#pragma pack(pop)
static_assert(sizeof(TensorRecord) == 48, "manifest tensor record");
static_assert(sizeof(LayerRecord) == 128, "manifest layer record");

struct Manifest
{
    std::vector<TensorRecord> tensors;
    std::vector<LayerRecord>  layers;
    uint32_t                  blobBytes = 0;
};

bool ParseManifest(const void* data, size_t size, Manifest& out, std::string& error);

// Layout matches cbuffer NetConstants in runtime/shaders/net/nss_net_common.hlsli.
struct NetConstants
{
    uint32_t inBase, inPitch, inPix, inW, inH;
    uint32_t outBase, outPitch, outPix, outW, outH, outRing;
    uint32_t cout, wOff, biasOff, scaleOff, lutOff;
    int32_t  zpLogit;
    uint32_t groupsX, pad0, pad1;
};
static_assert(sizeof(NetConstants) == 80, "NetConstants must match the HLSL cbuffer");
constexpr uint32_t kNetConstantDwords = sizeof(NetConstants) / 4;

struct KernelKey
{
    Kernel   kernel     = Kernel::Dp4a;
    uint32_t cin        = 0;
    uint32_t cout       = 0;   // WMMA only (tile shape); 0 for DP4a
    uint32_t ocb        = 0;   // DP4a only
    uint32_t stride     = 1;
    uint32_t upsample   = 0;
    uint32_t inBordered = 1;
    uint32_t outKind    = 0;
    uint32_t act        = 0;

    bool        operator==(const KernelKey& o) const;
    std::string Name() const;                        // e.g. dp4a_c32_o16_s1_u0_b1_k0_a0
    std::vector<std::string> Defines() const;        // e.g. {"CIN=32", "OCB=16", ...}
    const char* Source() const;                      // shader file name under shaders/net
};

// Which resource a tensor view lives in.
enum class Resource : uint32_t { Arena = 0, Input = 1, Kpn = 2, Temporal = 3 };

struct TensorLayout
{
    TensorKind kind;
    uint32_t   channels;
    uint32_t   level;
    uint32_t   w, h;            // valid size this frame
    uint32_t   maxW, maxH;      // allocation bound
    uint32_t   pitchPixels;     // pixels per row
    uint32_t   rows;            // allocated rows
    uint32_t   border;          // 1 for arena tensors
    uint64_t   offset;          // byte offset in the arena (arena tensors)
    uint64_t   bytes;
};

struct Dispatch
{
    uint32_t     layer;
    KernelKey    key;
    uint32_t     groups[3];
    NetConstants constants;
    Resource     in;
    Resource     out;
    bool         barrierBefore;  // UAV barrier on the arena (and input) before this dispatch
};

struct PlanConfig
{
    uint32_t maxWidth    = 0;    // network input bound (padded render size), multiple of 8
    uint32_t maxHeight   = 0;
    uint32_t inputPitch  = 0;    // pixels per row of the external input tensor (0: maxWidth)
    uint32_t kpnPitch    = 0;    // pixels per row of the KPN buffer (0: maxWidth / 4)
    bool     useWmma     = false;
};

class Plan
{
public:
    bool Build(const Manifest& manifest, const PlanConfig& config, std::string& error);
    // Per-frame network input size (multiple of 8, <= max). Rebuilds constants and grids.
    bool Update(uint32_t width, uint32_t height, std::string& error);

    uint64_t                         ArenaBytes() const { return arenaBytes_; }
    uint64_t                         InputBytes() const;
    uint64_t                         KpnBytes() const;
    const std::vector<TensorLayout>& Tensors() const { return tensors_; }
    const std::vector<Dispatch>&     Dispatches() const { return dispatches_; }
    std::vector<KernelKey>           RequiredKernels(bool includeWmma) const;
    uint32_t                         Width() const { return width_; }
    uint32_t                         Height() const { return height_; }
    const TensorLayout&              TensorOfKind(TensorKind kind) const;

private:
    KernelKey KeyFor(const LayerRecord& layer, bool wmma) const;

    Manifest                  manifest_;
    PlanConfig                config_;
    std::vector<TensorLayout> tensors_;
    std::vector<Dispatch>     dispatches_;
    uint64_t                  arenaBytes_ = 0;
    uint32_t                  width_      = 0;
    uint32_t                  height_     = 0;
};

constexpr uint32_t RoundUp(uint32_t v, uint32_t m) { return (v + m - 1) / m * m; }
constexpr uint32_t DivUp(uint32_t v, uint32_t m) { return (v + m - 1) / m; }

// Tile shape of the WMMA kernel: 16 x (16 * NT_PX) outputs per wave, all output channels.
constexpr uint32_t WmmaPixelTiles(uint32_t cout) { return 8 / (cout / 16); }

}  // namespace rdnu
