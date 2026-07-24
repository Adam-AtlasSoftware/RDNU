// rdnu_engine.cpp - minimal on-GPU execution graph for a whole RDNU block.
//
// Where rdnu_harness validates ONE kernel per run, the engine chains several kernels on
// the GPU through intermediate ("ping-pong") feature buffers, with no CPU round-trips
// between layers, and validates the block's end-to-end result against the PyTorch golden.
//
// Blocks it can run (program selected by which weights the bundle carries):
//   * CCM/FFN : conv3x3 -> GELU -> conv1x1
//   * DFM     : proj_in(dw3x3, 1x1) -> chunk -> [mlp_h/mlp_w shift+1x1] + gate + merge -> proj_out
//
// Ops: conv2d, gelu, mul (elementwise), concat (channels), chunk (a free offset-view: two
// StructuredBuffer root SRVs into one parent buffer at a channel byte-offset). This is the
// seed of the metadata-driven engine that will later run whole models. Correctness first:
// the same plain FP32 conv2d/gelu/mul/concat kernels the primitives were validated with.
//
// Usage: rdnu_engine <block.rdnut> [shaderDir]
//   shaderDir defaults to the executable's directory (where the *.hlsl kernels live).

#include "rdnu_dx12.h"

#include <set>

namespace
{
uint32_t U(float f) { return uint32_t(std::lround(f)); }

struct Shape { uint32_t c, h, w; size_t count() const { return size_t(c) * h * w; } };

// A feature-map reference: a byte range inside an owning resource. `chunk` yields views
// that share a parent resource (only the offset differs) — no copy, no dispatch.
struct Value { ID3D12Resource* res; uint64_t off; Shape shape; };

// One node of the block graph. Feature tensors are referenced by name; weight/bias name
// bundle entries. Conv fields are unused for non-conv ops.
struct Op
{
    std::string kind;                 // conv2d | gelu | mul | concat | chunk
    std::vector<std::string> ins;
    std::vector<std::string> outs;
    std::string weight, bias;
    uint32_t KH, KW, PadH, PadW, StrideH, StrideW, Groups;
};

Op conv(std::string in, std::string out, std::string w, std::string b,
        uint32_t kh, uint32_t kw, uint32_t ph, uint32_t pw, uint32_t sh, uint32_t sw, uint32_t g)
{
    return Op{ "conv2d", { in }, { out }, w, b, kh, kw, ph, pw, sh, sw, g };
}

// INT8 (W8A8) conv. `wbase` names the bundle group: <wbase>.w / .scale / .bias / .ascale.
Op convI8(std::string in, std::string out, std::string wbase,
          uint32_t kh, uint32_t kw, uint32_t ph, uint32_t pw, uint32_t sh, uint32_t sw, uint32_t g)
{
    return Op{ "conv2d_int8", { in }, { out }, wbase, "", kh, kw, ph, pw, sh, sw, g };
}

// Sub-graphs are appended with a `p` prefix on their weight-bundle keys AND intermediate
// value names, and explicit `in`/`out` feature names, so a block can be reused standalone
// or composed inside a larger block (EncodeLayer, and later DecodeLayer) without collisions.

// DFM (dim=36, hidden=72). Mirrors DFM.forward in rdg_block.py.
void appendDFM(std::vector<Op>& ops, const std::string& p, const std::string& in, const std::string& out)
{
    auto n = [&](const std::string& s) { return p + s; };  // prefixed name (weight key or intermediate)
    ops.push_back(conv(in,       n("pi0"), n("proj_in.0.weight"), n("proj_in.0.bias"), 3, 3, 1, 1, 1, 1, 0)); // dw3x3 (g=dim) dim->2dim
    ops.push_back(conv(n("pi0"), n("pi"),  n("proj_in.1.weight"), n("proj_in.1.bias"), 1, 1, 0, 0, 1, 1, 1));  // 1x1 2dim->2dim
    ops.push_back(Op{ "chunk", { n("pi") }, { n("g"), n("c") } });                                             // -> g(dim), c(dim)
    ops.push_back(conv(n("c"),   n("hsh"), n("mlp_h.shift_weight"), "", 1, 7, 0, 3, 1, 1, 0));                 // 1x7 shift (g=dim, no bias)
    ops.push_back(conv(n("hsh"), n("hc"),  n("mlp_h.weight"), n("mlp_h.bias"), 1, 1, 0, 0, 1, 1, 1));          // 1x1 dim->dim
    ops.push_back(conv(n("hc"),  n("wsh"), n("mlp_w.shift_weight"), "", 7, 1, 3, 0, 1, 1, 0));                 // 7x1 shift (g=dim, no bias)
    ops.push_back(conv(n("wsh"), n("c2"),  n("mlp_w.weight"), n("mlp_w.bias"), 1, 1, 0, 0, 1, 1, 1));          // 1x1 36->36
    ops.push_back(Op{ "concat", { in, n("c2") }, { n("cat") } });                                              // cat[x, c2] -> 72
    ops.push_back(conv(n("cat"), n("merged"), n("merge.weight"), n("merge.bias"), 1, 1, 0, 0, 1, 1, 1));       // 1x1 72->36
    ops.push_back(Op{ "gelu", { n("g") }, { n("gact") } });                                                    // gelu(g)
    ops.push_back(Op{ "mul", { n("gact"), n("merged") }, { n("gated") } });                                    // gelu(g) * merged
    ops.push_back(conv(n("gated"), out, n("proj_out.weight"), n("proj_out.bias"), 1, 1, 0, 0, 1, 1, 1));       // 1x1 36->36
}

// CCM/FFN: conv3x3 -> GELU -> conv1x1.
void appendCCM(std::vector<Op>& ops, const std::string& p, const std::string& in, const std::string& out)
{
    auto n = [&](const std::string& s) { return p + s; };
    ops.push_back(conv(in,       n("h0"), n("fn.0.weight"), n("fn.0.bias"), 3, 3, 1, 1, 1, 1, 1));
    ops.push_back(Op{ "gelu", { n("h0") }, { n("h1") } });
    ops.push_back(conv(n("h1"),  out,     n("fn.2.weight"), n("fn.2.bias"), 1, 1, 0, 0, 1, 1, 1));
}

// HFB(dim, g_dim=6): cross-GELU-gated fusion of feature x and g-buffer g. Mirrors HFB.forward.
// (The g -> x-size bilinear resize is an identity where the sizes already match, so it is omitted
// here; a resize op will be added when validating HFB instances at coarser scales.)
void appendHFB(std::vector<Op>& ops, const std::string& p, const std::string& xin, const std::string& gin, const std::string& out)
{
    auto n = [&](const std::string& s) { return p + s; };
    ops.push_back(Op{ "resize", { gin, xin }, { n("g_rs") } });                                  // g -> x size (bilinear)
    ops.push_back(conv(n("g_rs"), n("x2"), n("conv_g.weight"), n("conv_g.bias"), 3, 3, 1, 1, 1, 1, 1)); // 6->dim
    ops.push_back(conv(xin, n("x1"), n("conv_x.weight"), n("conv_x.bias"), 3, 3, 1, 1, 1, 1, 1)); // dim->dim
    ops.push_back(Op{ "gelu", { n("x2") }, { n("gx2") } });
    ops.push_back(Op{ "gelu", { n("x1") }, { n("gx1") } });
    ops.push_back(Op{ "mul", { n("x1"), n("gx2") }, { n("a") } });                                // x1 * gelu(x2)
    ops.push_back(Op{ "mul", { n("x2"), n("gx1") }, { n("b") } });                                // x2 * gelu(x1)
    ops.push_back(Op{ "concat", { n("a"), n("b") }, { n("cat") } });                              // -> 2*dim
    ops.push_back(conv(n("cat"), out, n("proj_out.weight"), n("proj_out.bias"), 1, 1, 0, 0, 1, 1, 1)); // 2*dim->dim
}

// CTR (num_head=4): channel-wise cosine attention with motion/depth guidance. Mirrors CTR.forward
// for the frame-0 case (pre_h=None -> self-attention with an identity motion-warp, so `to_kv` reads
// `xin` directly and no grid-sample is needed). Op kinds: l2norm / scores / softmax / apply.
// `Groups` carries num_head for the attention ops; dw3x3 convs use Groups=0 (depthwise, groups=Cin).
void appendCTR(std::vector<Op>& ops, const std::string& p, const std::string& xin, const std::string& depthin, const std::string& out)
{
    auto n = [&](const std::string& s) { return p + s; };
    const uint32_t H = 4;  // num_head
    ops.push_back(Op{ "resize", { depthin, xin }, { n("d_rs") } });   // depth -> x size (bilinear)
    ops.push_back(conv(xin, n("tq0"), n("to_q.0.weight"), n("to_q.0.bias"), 1, 1, 0, 0, 1, 1, 1));    // 1x1 dim->2dim
    ops.push_back(conv(n("tq0"), n("tq"), n("to_q.1.weight"), n("to_q.1.bias"), 3, 3, 1, 1, 1, 1, 0)); // dw3x3
    ops.push_back(Op{ "chunk", { n("tq") }, { n("q1"), n("q2") } });
    ops.push_back(conv(xin, n("tkv0"), n("to_kv.0.weight"), n("to_kv.0.bias"), 1, 1, 0, 0, 1, 1, 1));  // 1x1 dim->4dim
    ops.push_back(conv(n("tkv0"), n("tkv"), n("to_kv.1.weight"), n("to_kv.1.bias"), 3, 3, 1, 1, 1, 1, 0)); // dw3x3
    ops.push_back(Op{ "chunk", { n("tkv") }, { n("k"), n("v") } });
    ops.push_back(Op{ "l2norm", { n("q1") }, { n("q1n") } });
    ops.push_back(Op{ "l2norm", { n("k") }, { n("kn") } });
    ops.push_back(Op{ "scores", { n("q1n"), n("kn") }, { n("attn") }, "", "", 0, 0, 0, 0, 0, 0, H });
    ops.push_back(Op{ "softmax", { n("attn") }, { n("attnp") }, "", "", 0, 0, 0, 0, 0, 0, H });
    ops.push_back(Op{ "apply", { n("attnp"), n("v") }, { n("att") }, "", "", 0, 0, 0, 0, 0, 0, H });
    ops.push_back(Op{ "concat", { n("att"), n("q2") }, { n("cat1") } });                 // out + q2
    ops.push_back(Op{ "concat", { n("cat1"), n("d_rs") }, { n("cat2") } });              // + depth
    ops.push_back(conv(n("cat2"), out, n("proj_out.weight"), n("proj_out.bias"), 1, 1, 0, 0, 1, 1, 1)); // 1x1 (2dim+1)->dim
}

// Upsampling: F.interpolate(x, scale_factor=2, bilinear) then a 3x3 conv. (ups.* in rdg_arch.py)
void appendUpsample(std::vector<Op>& ops, const std::string& p, const std::string& in, const std::string& out)
{
    auto n = [&](const std::string& s) { return p + s; };
    ops.push_back(Op{ "resize", { in }, { n("up") }, "", "", 0, 0, 0, 0, /*StrideH=scale*/2, 2, 0 });
    ops.push_back(conv(n("up"), out, n("conv.weight"), n("conv.bias"), 3, 3, 1, 1, 1, 1, 1));
}

// Final UpSample head: Conv2d(dim -> 3*scale*scale) then PixelShuffle(scale). (UpSample in rdg_arch.py)
void appendFinalUpSample(std::vector<Op>& ops, const std::string& p, const std::string& in, const std::string& out)
{
    auto n = [&](const std::string& s) { return p + s; };
    ops.push_back(conv(in, n("conv"), n("0.weight"), n("0.bias"), 3, 3, 1, 1, 1, 1, 1));       // UpSample.0
    ops.push_back(Op{ "pixelshuffle", { n("conv") }, { out }, "", "", 0, 0, 0, 0, /*r*/2, 2, 0 }); // PixelShuffle(2)
}

std::vector<Op> programDFM() { std::vector<Op> ops; appendDFM(ops, "", "input", "output"); return ops; }
std::vector<Op> programCCM() { std::vector<Op> ops; appendCCM(ops, "", "input", "output"); return ops; }
std::vector<Op> programHFB() { std::vector<Op> ops; appendHFB(ops, "", "input", "g", "output"); return ops; }
std::vector<Op> programCTR() { std::vector<Op> ops; appendCTR(ops, "", "input", "depth", "output"); return ops; }
std::vector<Op> programUpsample() { std::vector<Op> ops; appendUpsample(ops, "", "input", "output"); return ops; }
std::vector<Op> programFinalUpSample() { std::vector<Op> ops; appendFinalUpSample(ops, "UpSample.", "input", "output"); return ops; }
// Single W8A8 conv (bundle carries params in "i8p" = [KH,KW,PadH,PadW,StrideH,StrideW,Groups]).
std::vector<Op> programConvInt8(const std::vector<float>& p)
{
    std::vector<Op> ops;
    auto u = [&](int i) { return uint32_t(std::lround(p[i])); };
    ops.push_back(convI8("input", "output", "i8", u(0), u(1), u(2), u(3), u(4), u(5), u(6)));
    return ops;
}

// EncodeLayer: x = dfm(x) + x; x = ffn(x) + x. (encoders.*.* in rdg_arch.py)
void appendEncodeLayer(std::vector<Op>& ops, const std::string& p, const std::string& in, const std::string& out)
{
    auto n = [&](const std::string& s) { return p + s; };
    appendDFM(ops, p + "dfm.", in, n("dfm_out"));
    ops.push_back(Op{ "add", { in, n("dfm_out") }, { n("e1") } });           // residual
    appendCCM(ops, p + "ffn.", n("e1"), n("ffn_out"));
    ops.push_back(Op{ "add", { n("e1"), n("ffn_out") }, { out } });          // residual
}

// DecodeLayer: x = hfb(x, g) + x; x = ctr(x, d) + x; x = ffn(x) + x. (decoders.*/middle_decoders)
// Frame-0 CTR (identity warp). g/depth are block-level feature inputs.
void appendDecodeLayer(std::vector<Op>& ops, const std::string& p, const std::string& xin,
                       const std::string& gin, const std::string& din, const std::string& out)
{
    auto n = [&](const std::string& s) { return p + s; };
    appendHFB(ops, p + "hfb.", xin, gin, n("hfb_out"));
    ops.push_back(Op{ "add", { xin, n("hfb_out") }, { n("d1") } });          // residual
    appendCTR(ops, p + "ctr.", n("d1"), din, n("ctr_out"));
    ops.push_back(Op{ "add", { n("d1"), n("ctr_out") }, { n("d2") } });      // residual
    appendCCM(ops, p + "ffn.", n("d2"), n("ffn_out"));
    ops.push_back(Op{ "add", { n("d2"), n("ffn_out") }, { out } });          // residual
}

std::vector<Op> programEncodeLayer() { std::vector<Op> ops; appendEncodeLayer(ops, "", "input", "output"); return ops; }
std::vector<Op> programDecodeLayer() { std::vector<Op> ops; appendDecodeLayer(ops, "", "input", "g", "depth", "output"); return ops; }

// Whole single-frame network — RDG.process_forward, frame-0 (pre_h=None => identity warps, m=None).
// Config pinned by the weights: num_feat=36, enc/dec [1,1], middle 1, scale 2. External inputs:
// image (3ch), g (6ch = normal+brdf), depth (1ch); weights keyed by their full state_dict names.
std::vector<Op> programFull()
{
    std::vector<Op> ops;
    ops.push_back(conv("image", "x0", "first_conv.weight", "first_conv.bias", 3, 3, 1, 1, 1, 1, 1)); // 3->36, res=x0
    // encoder 0 (36) -> down -> 72@32
    appendEncodeLayer(ops, "encoders.0.0.", "x0", "enc0");
    ops.push_back(conv("enc0", "down0", "downs.0.weight", "downs.0.bias", 3, 3, 1, 1, 2, 2, 1));      // strided
    // encoder 1 (72) -> down -> 108@16
    appendEncodeLayer(ops, "encoders.1.0.", "down0", "enc1");
    ops.push_back(conv("enc1", "down1", "downs.1.weight", "downs.1.bias", 3, 3, 1, 1, 2, 2, 1));
    // middle enc + dec (108@16)
    appendEncodeLayer(ops, "middle_encoders.0.", "down1", "mid_e");
    appendDecodeLayer(ops, "middle_decoders.", "mid_e", "g", "depth", "mid_d");
    // decoder 0: up 108->72@32, skip += alpha_72*enc1, decode
    appendUpsample(ops, "ups.0.", "mid_d", "up0");
    ops.push_back(Op{ "axpy", { "up0", "enc1" }, { "dm0" }, "skip_alphas.1" });
    appendDecodeLayer(ops, "decoders.0.", "dm0", "g", "depth", "dec0");
    // decoder 1: up 72->36@64, skip += alpha_36*enc0, decode
    appendUpsample(ops, "ups.1.", "dec0", "up1");
    ops.push_back(Op{ "axpy", { "up1", "enc0" }, { "dm1" }, "skip_alphas.0" });
    appendDecodeLayer(ops, "decoders.1.", "dm1", "g", "depth", "dec1");
    // x += res, then the UpSample head (conv + PixelShuffle) -> 3@128
    ops.push_back(Op{ "add", { "dec1", "x0" }, { "xres" } });
    appendFinalUpSample(ops, "UpSample.", "xres", "output");
    return ops;
}
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: rdnu_engine <block.rdnut> [shaderDir]\n"); return 1; }

    std::wstring shaderDir;
    if (argc >= 3)
    {
        std::string s(argv[2]); shaderDir.assign(s.begin(), s.end());
        if (shaderDir.back() != L'\\' && shaderDir.back() != L'/') shaderDir += L'\\';
    }
    else
    {
        wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring dir(exe); shaderDir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
    }

    auto bundle = LoadRdnut(argv[1]);
    const Tensor& golden = bundle.at("golden_out");

    // Pick the block program from which weights the bundle carries.
    std::vector<Op> ops; const char* blockName;
    if (bundle.count("i8p")) { ops = programConvInt8(bundle.at("i8p").data); blockName = "ConvInt8"; }
    else if (bundle.count("first_conv.weight")) { ops = programFull(); blockName = "WholeModel"; }
    else if (bundle.count("hfb.conv_g.weight")) { ops = programDecodeLayer(); blockName = "DecodeLayer"; }
    else if (bundle.count("dfm.proj_in.0.weight")) { ops = programEncodeLayer(); blockName = "EncodeLayer"; }
    else if (bundle.count("conv_g.weight")) { ops = programHFB(); blockName = "HFB"; }
    else if (bundle.count("to_q.0.weight")) { ops = programCTR(); blockName = "CTR"; }
    else if (bundle.count("proj_in.0.weight")) { ops = programDFM(); blockName = "DFM"; }
    else if (bundle.count("UpSample.0.weight")) { ops = programFinalUpSample(); blockName = "UpSample"; }
    else if (bundle.count("conv.weight")) { ops = programUpsample(); blockName = "Upsample"; }
    else { ops = programCCM(); blockName = "CCM/FFN"; }
    std::printf("block: %s   (%zu ops)\n", blockName, ops.size());

    // ---- device / queue / command list ----
    ComPtr<ID3D12Device> dev = CreateHighPerfDevice();
    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    Check(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
    ComPtr<ID3D12CommandAllocator> alloc;
    Check(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc)), "CreateCommandAllocator");
    ComPtr<ID3D12GraphicsCommandList> cl;
    Check(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&cl)), "CreateCommandList");

    // ---- root signature: root constants(b0,13) + SRV t0..t4 + UAV u0 (all root descriptors).
    // 5 SRVs so the INT8 conv can bind input/weights/w_scale/bias/a_scale; FP ops use a subset. ----
    D3D12_ROOT_PARAMETER params[7] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 13;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    for (int i = 1; i <= 5; ++i)   // t0..t4
    {
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[i].Descriptor.ShaderRegister = UINT(i - 1);
        params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    params[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;   // u0
    params[6].Descriptor.ShaderRegister = 0;
    params[6].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = _countof(params);
    rsDesc.pParameters = params;
    ComPtr<ID3DBlob> rsBlob, rsErr;
    Check(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsErr), "SerializeRootSignature");
    ComPtr<ID3D12RootSignature> rootSig;
    Check(dev->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig)), "CreateRootSignature");

    // ---- one PSO per op kind, shared across ops ----
    auto makePSO = [&](const wchar_t* file, const wchar_t* entry) {
        auto dxil = CompileCS(shaderDir + file, entry, L"cs_6_2");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = rootSig.Get();
        pd.CS = { dxil.data(), dxil.size() };
        ComPtr<ID3D12PipelineState> pso;
        Check(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)), "CreateComputePipelineState");
        return pso;
    };
    ComPtr<ID3D12PipelineState> psoConv    = makePSO(L"conv2d.hlsl", L"conv2d_CS");
    ComPtr<ID3D12PipelineState> psoGelu    = makePSO(L"gelu.hlsl",   L"gelu_CS");
    ComPtr<ID3D12PipelineState> psoMul     = makePSO(L"mul.hlsl",    L"mul_CS");
    ComPtr<ID3D12PipelineState> psoAdd     = makePSO(L"add.hlsl",    L"add_CS");
    ComPtr<ID3D12PipelineState> psoConcat  = makePSO(L"concat.hlsl", L"concat_CS");
    // CTR attention kernels.
    ComPtr<ID3D12PipelineState> psoL2      = makePSO(L"l2norm_spatial.hlsl", L"l2norm_spatial_CS");
    ComPtr<ID3D12PipelineState> psoScores  = makePSO(L"ctr_scores.hlsl",     L"ctr_scores_CS");
    ComPtr<ID3D12PipelineState> psoSoftmax = makePSO(L"softmax_rows.hlsl",   L"softmax_rows_CS");
    ComPtr<ID3D12PipelineState> psoApply   = makePSO(L"ctr_apply.hlsl",      L"ctr_apply_CS");
    ComPtr<ID3D12PipelineState> psoResize  = makePSO(L"resize.hlsl",         L"resize_CS");
    ComPtr<ID3D12PipelineState> psoPShuf   = makePSO(L"pixelshuffle.hlsl",   L"pixelshuffle_CS");
    ComPtr<ID3D12PipelineState> psoAxpy    = makePSO(L"axpy.hlsl",           L"axpy_CS");
    ComPtr<ID3D12PipelineState> psoConvI8  = makePSO(L"conv2d_int8.hlsl",    L"conv2d_int8_CS");

    // ---- resource bookkeeping ----
    std::vector<ComPtr<ID3D12Resource>> alive;                    // keep every buffer alive
    std::map<std::string, Value> vals;                            // feature values by name
    std::map<ID3D12Resource*, D3D12_RESOURCE_STATES> rstate;      // DEFAULT-buffer current state
    std::map<std::string, ComPtr<ID3D12Resource>> wbuf;          // weight/bias uploads
    std::map<uint32_t, ComPtr<ID3D12Resource>> zeroBias;         // synthesized zero-bias by length

    auto VA = [](const Value& v) { return v.res->GetGPUVirtualAddress() + v.off; };

    // External feature inputs = op-input names that no op (or chunk) produces. Load each from the
    // bundle onto an upload-heap buffer (always readable, so not tracked in rstate). This covers
    // "input", plus extra feature inputs like HFB's "g" — weights ride op.weight/op.bias, not ins.
    {
        std::set<std::string> produced;
        for (const Op& op : ops) for (const std::string& o : op.outs) produced.insert(o);
        for (const Op& op : ops) for (const std::string& in : op.ins)
        {
            if (produced.count(in) || vals.count(in)) continue;
            const Tensor& t = bundle.at(in);
            auto up = UploadBuffer(dev.Get(), t.data.data(), t.data.size() * 4);
            alive.push_back(up);
            vals[in] = { up.Get(), 0, { t.dims[0], t.dims[1], t.dims[2] } };
        }
    }
    auto uploadWB = [&](const std::string& name) {
        if (name.empty() || wbuf.count(name)) return;
        const Tensor& t = bundle.at(name);
        wbuf[name] = UploadBuffer(dev.Get(), t.data.data(), t.data.size() * 4);
    };
    auto zeroBiasVA = [&](uint32_t n) {
        if (!zeroBias.count(n))
        {
            std::vector<float> z(n, 0.0f);
            zeroBias[n] = UploadBuffer(dev.Get(), z.data(), size_t(n) * 4);
        }
        return zeroBias[n]->GetGPUVirtualAddress();
    };
    // Transition a DEFAULT buffer to be readable as an SRV (shared views transition once).
    auto ensureSRV = [&](ID3D12Resource* r) {
        auto it = rstate.find(r);
        if (it != rstate.end() && it->second != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        {
            Transition(cl.Get(), r, it->second, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            it->second = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
    };

    // ---- record the graph ----
    cl->SetComputeRootSignature(rootSig.Get());
    for (const Op& op : ops)
    {
        if (op.kind == "chunk")
        {
            // Free offset views: split the parent's channels into equal slices.
            Value src = vals.at(op.ins[0]);
            uint32_t n = uint32_t(op.outs.size());
            uint32_t half = src.shape.c / n;
            size_t plane = size_t(src.shape.h) * src.shape.w * 4;
            for (uint32_t k = 0; k < n; ++k)
                vals[op.outs[k]] = { src.res, src.off + size_t(k) * half * plane, { half, src.shape.h, src.shape.w } };
            std::printf("  %-6s %-6s -> %ux (%u,%u,%u)\n", op.kind.c_str(), op.ins[0].c_str(),
                        n, half, src.shape.h, src.shape.w);
            continue;
        }

        for (const std::string& in : op.ins) ensureSRV(vals.at(in).res);

        Value a = vals.at(op.ins[0]);
        Shape out{};
        ID3D12PipelineState* pso = nullptr;
        uint32_t cst[13] = { 0 };            // root constants (13 uints; slot meaning is per op kind)
        uint32_t gx = 1, gy = 1, gz = 1;     // dispatch grid
        bool twoInput = false;               // bind t1 = ins[1] (mul/add/concat/scores/apply)

        if (op.kind == "conv2d")
        {
            uploadWB(op.weight); uploadWB(op.bias);
            uint32_t groups = (op.Groups == 0) ? a.shape.c : op.Groups;   // 0 => depthwise (groups = Cin)
            out.c = bundle.at(op.weight).dims[0];
            out.h = (a.shape.h + 2 * op.PadH - op.KH) / op.StrideH + 1;
            out.w = (a.shape.w + 2 * op.PadW - op.KW) / op.StrideW + 1;
            pso = psoConv.Get();
            cst[0] = a.shape.c; cst[1] = a.shape.h; cst[2] = a.shape.w; cst[3] = out.c;
            cst[4] = op.KH; cst[5] = op.KW; cst[6] = op.PadH; cst[7] = op.PadW;
            cst[8] = op.StrideH; cst[9] = op.StrideW; cst[10] = groups; cst[11] = out.h; cst[12] = out.w;
            gx = (out.w + 7) / 8; gy = (out.h + 7) / 8; gz = out.c;
        }
        else if (op.kind == "conv2d_int8")   // W8A8 conv; weights keyed off op.weight (+.w/.scale/.bias/.ascale)
        {
            uploadWB(op.weight + ".w"); uploadWB(op.weight + ".scale");
            uploadWB(op.weight + ".bias"); uploadWB(op.weight + ".ascale");
            uint32_t groups = (op.Groups == 0) ? a.shape.c : op.Groups;
            out.c = bundle.at(op.weight + ".w").dims[0];
            out.h = (a.shape.h + 2 * op.PadH - op.KH) / op.StrideH + 1;
            out.w = (a.shape.w + 2 * op.PadW - op.KW) / op.StrideW + 1;
            pso = psoConvI8.Get();
            cst[0] = a.shape.c; cst[1] = a.shape.h; cst[2] = a.shape.w; cst[3] = out.c;
            cst[4] = op.KH; cst[5] = op.KW; cst[6] = op.PadH; cst[7] = op.PadW;
            cst[8] = op.StrideH; cst[9] = op.StrideW; cst[10] = groups; cst[11] = out.h; cst[12] = out.w;
            gx = (out.w + 7) / 8; gy = (out.h + 7) / 8; gz = out.c;
        }
        else if (op.kind == "gelu" || op.kind == "mul" || op.kind == "add" || op.kind == "concat" || op.kind == "axpy")
        {
            twoInput = (op.kind != "gelu");
            if (op.kind == "concat") { Value b = vals.at(op.ins[1]); out = { a.shape.c + b.shape.c, a.shape.h, a.shape.w }; }
            else out = a.shape;
            pso = op.kind == "gelu" ? psoGelu.Get() : op.kind == "mul" ? psoMul.Get()
                : op.kind == "add" ? psoAdd.Get() : op.kind == "axpy" ? psoAxpy.Get() : psoConcat.Get();
            if (op.kind == "axpy") uploadWB(op.weight);   // per-channel alpha on t2
            cst[0] = a.shape.c; cst[1] = a.shape.h; cst[2] = a.shape.w; cst[3] = out.c;
            cst[11] = out.h; cst[12] = out.w;
            gx = (out.w + 7) / 8; gy = (out.h + 7) / 8; gz = out.c;
        }
        else if (op.kind == "resize")   // bilinear; slots: Cout=C, H, W, OH, OW
        {
            // Target size: from a reference input's spatial dims (ins[1]) if given, else ×StrideH.
            if (op.ins.size() >= 2) { Value r = vals.at(op.ins[1]); out = { a.shape.c, r.shape.h, r.shape.w }; }
            else { uint32_t s = op.StrideH; out = { a.shape.c, a.shape.h * s, a.shape.w * s }; }
            pso = psoResize.Get();
            cst[3] = a.shape.c; cst[1] = a.shape.h; cst[2] = a.shape.w; cst[11] = out.h; cst[12] = out.w;
            gx = (out.w + 7) / 8; gy = (out.h + 7) / 8; gz = out.c;
        }
        else if (op.kind == "pixelshuffle")   // (C*r*r,H,W)->(C,H*r,W*r); slots: Cin,H,W,Cout,StrideH=r,OH,OW
        {
            uint32_t r = op.StrideH;
            out = { a.shape.c / (r * r), a.shape.h * r, a.shape.w * r }; pso = psoPShuf.Get();
            cst[0] = a.shape.c; cst[1] = a.shape.h; cst[2] = a.shape.w; cst[3] = out.c;
            cst[8] = r; cst[11] = out.h; cst[12] = out.w;
            gx = (out.w + 7) / 8; gy = (out.h + 7) / 8; gz = out.c;
        }
        else if (op.kind == "l2norm")   // per-channel spatial L2 normalize; slots: Cout=C, H, W
        {
            out = a.shape; pso = psoL2.Get();
            cst[1] = a.shape.h; cst[2] = a.shape.w; cst[3] = a.shape.c;
            gx = 1; gy = 1; gz = (a.shape.c + 63) / 64;
        }
        else if (op.kind == "scores")   // attn(head,Cq,Ck); slots: Cin=head, H=Cq, W=Ck, Cout=hw
        {
            twoInput = true; Value b = vals.at(op.ins[1]);
            uint32_t head = op.Groups, Cq = a.shape.c / head, Ck = b.shape.c / head, hw = a.shape.h * a.shape.w;
            out = { head, Cq, Ck }; pso = psoScores.Get();
            cst[0] = head; cst[1] = Cq; cst[2] = Ck; cst[3] = hw;
            gx = (Ck + 7) / 8; gy = (Cq + 7) / 8; gz = head;
        }
        else if (op.kind == "softmax")  // over Ck; slots: Cin=head, H=Cq, W=Ck
        {
            out = a.shape; pso = psoSoftmax.Get();
            uint32_t head = a.shape.c, Cq = a.shape.h, Ck = a.shape.w;
            cst[0] = head; cst[1] = Cq; cst[2] = Ck;
            gx = (Cq + 7) / 8; gy = (head + 7) / 8; gz = 1;
        }
        else if (op.kind == "apply")    // out(head*Cq,H,W); slots: Cout, H, W, Groups=head, KH=Cq, KW=Ck
        {
            twoInput = true; Value b = vals.at(op.ins[1]);   // v: (head*Ck, H, W)
            uint32_t head = op.Groups, Cq = a.shape.h, Ck = a.shape.w;
            out = { head * Cq, b.shape.h, b.shape.w }; pso = psoApply.Get();
            cst[1] = out.h; cst[2] = out.w; cst[3] = out.c; cst[4] = Cq; cst[5] = Ck; cst[10] = head;
            gx = (out.w + 7) / 8; gy = (out.h + 7) / 8; gz = out.c;
        }

        auto outBuf = DefaultBuffer(dev.Get(), out.count() * 4, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        alive.push_back(outBuf);
        rstate[outBuf.Get()] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        vals[op.outs[0]] = { outBuf.Get(), 0, out };

        cl->SetPipelineState(pso);
        cl->SetComputeRoot32BitConstants(0, 13, cst, 0);
        cl->SetComputeRootShaderResourceView(1, VA(a));                              // t0 = first input
        if (op.kind == "conv2d")
        {
            cl->SetComputeRootShaderResourceView(2, wbuf.at(op.weight)->GetGPUVirtualAddress());
            cl->SetComputeRootShaderResourceView(3, op.bias.empty() ? zeroBiasVA(out.c)
                                                                    : wbuf.at(op.bias)->GetGPUVirtualAddress());
        }
        else if (op.kind == "conv2d_int8")   // t1=wi8, t2=w_scale, t3=bias, t4=a_scale (keyed off op.weight)
        {
            cl->SetComputeRootShaderResourceView(2, wbuf.at(op.weight + ".w")->GetGPUVirtualAddress());
            cl->SetComputeRootShaderResourceView(3, wbuf.at(op.weight + ".scale")->GetGPUVirtualAddress());
            cl->SetComputeRootShaderResourceView(4, wbuf.at(op.weight + ".bias")->GetGPUVirtualAddress());
            cl->SetComputeRootShaderResourceView(5, wbuf.at(op.weight + ".ascale")->GetGPUVirtualAddress());
        }
        else if (twoInput)
        {
            cl->SetComputeRootShaderResourceView(2, VA(vals.at(op.ins[1])));         // t1 = second input
            if (op.kind == "axpy")                                                   // t2 = per-channel alpha
                cl->SetComputeRootShaderResourceView(3, wbuf.at(op.weight)->GetGPUVirtualAddress());
        }
        cl->SetComputeRootUnorderedAccessView(6, outBuf->GetGPUVirtualAddress());
        cl->Dispatch(gx, gy, gz);

        std::printf("  %-8s %-9s -> %-9s (%u,%u,%u)\n", op.kind.c_str(), op.ins[0].c_str(),
                    op.outs[0].c_str(), out.c, out.h, out.w);
    }

    // ---- optional debug checkpoints: bundle tensors "chk.<value>" are compared against the engine's
    // intermediate <value>, to localize a divergence in a large graph. Read back after execution. ----
    struct Chk { std::string name; ComPtr<ID3D12Resource> rb; const Tensor* gold; size_t bytes; };
    std::vector<Chk> chks;
    for (const auto& kv : bundle)
    {
        if (kv.first.rfind("chk.", 0) != 0) continue;
        auto it = vals.find(kv.first.substr(4));
        if (it == vals.end()) continue;
        Value v = it->second;
        Transition(cl.Get(), v.res, rstate.at(v.res), D3D12_RESOURCE_STATE_COPY_SOURCE);
        rstate[v.res] = D3D12_RESOURCE_STATE_COPY_SOURCE;
        size_t bytes = v.shape.count() * 4;
        auto rb = ReadbackBuffer(dev.Get(), bytes);
        cl->CopyBufferRegion(rb.Get(), 0, v.res, v.off, bytes);
        chks.push_back({ kv.first.substr(4), rb, &kv.second, bytes });
    }

    // ---- final output: UAV -> COPY_SOURCE, read back ----
    Value outv = vals.at("output");
    Transition(cl.Get(), outv.res, rstate.at(outv.res), D3D12_RESOURCE_STATE_COPY_SOURCE);
    const size_t outBytes = outv.shape.count() * 4;
    ComPtr<ID3D12Resource> readback = ReadbackBuffer(dev.Get(), outBytes);
    cl->CopyResource(readback.Get(), outv.res);
    Check(cl->Close(), "Close");

    ID3D12CommandList* lists[] = { cl.Get() };
    queue->ExecuteCommandLists(1, lists);

    ComPtr<ID3D12Fence> fence;
    Check(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    Check(queue->Signal(fence.Get(), 1), "Signal");
    fence->SetEventOnCompletion(1, ev);
    WaitForSingleObject(ev, INFINITE);
    CloseHandle(ev);

    // ---- report checkpoints (first divergence localizes the bug) ----
    for (Chk& c : chks)
    {
        float* p = nullptr; D3D12_RANGE r{ 0, c.bytes };
        Check(c.rb->Map(0, &r, reinterpret_cast<void**>(&p)), "Map(chk)");
        double mx = 0.0; size_t n = c.gold->count();
        for (size_t i = 0; i < n; ++i) { double d = std::fabs(double(p[i]) - double(c.gold->data[i])); if (d > mx) mx = d; }
        c.rb->Unmap(0, nullptr);
        std::printf("  chk %-10s max|err| = %-12.5g %s\n", c.name.c_str(), mx, mx < 1e-2 ? "ok" : "<< DIVERGE");
    }

    // ---- compare block output vs golden ----
    float* gpu = nullptr; D3D12_RANGE rng{ 0, outBytes };
    Check(readback->Map(0, &rng, reinterpret_cast<void**>(&gpu)), "Map(readback)");
    double maxAbs = 0.0, sumAbs = 0.0, gmax = 0.0;
    size_t n = golden.count();
    for (size_t i = 0; i < n; ++i)
    {
        double d = std::fabs(double(gpu[i]) - double(golden.data[i]));
        if (d > maxAbs) maxAbs = d;
        sumAbs += d;
        double av = std::fabs(golden.data[i]); if (av > gmax) gmax = av;
    }
    std::printf("golden[0..2] = %.5f %.5f %.5f   gpu[0..2] = %.5f %.5f %.5f\n",
        golden.data[0], golden.data[1], golden.data[2], gpu[0], gpu[1], gpu[2]);
    readback->Unmap(0, nullptr);

    std::printf("block max|err| = %.6g   mean|err| = %.6g   (golden max|v| = %.4g)\n",
        maxAbs, sumAbs / double(n), gmax);
    const double tol = 1e-3;
    std::printf("%s (tol %.0e)\n", maxAbs < tol ? "PASS" : "FAIL", tol);
    return maxAbs < tol ? 0 : 2;
}
