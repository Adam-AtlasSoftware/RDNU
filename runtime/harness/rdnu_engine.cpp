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

// Sub-graphs are appended with a `p` prefix on their weight-bundle keys AND intermediate
// value names, and explicit `in`/`out` feature names, so a block can be reused standalone
// or composed inside a larger block (EncodeLayer, and later DecodeLayer) without collisions.

// DFM (dim=36, hidden=72). Mirrors DFM.forward in rdg_block.py.
void appendDFM(std::vector<Op>& ops, const std::string& p, const std::string& in, const std::string& out)
{
    auto n = [&](const std::string& s) { return p + s; };  // prefixed name (weight key or intermediate)
    ops.push_back(conv(in,       n("pi0"), n("proj_in.0.weight"), n("proj_in.0.bias"), 3, 3, 1, 1, 1, 1, 36)); // dw3x3 g36 36->72
    ops.push_back(conv(n("pi0"), n("pi"),  n("proj_in.1.weight"), n("proj_in.1.bias"), 1, 1, 0, 0, 1, 1, 1));  // 1x1 72->72
    ops.push_back(Op{ "chunk", { n("pi") }, { n("g"), n("c") } });                                             // -> g(36), c(36)
    ops.push_back(conv(n("c"),   n("hsh"), n("mlp_h.shift_weight"), "", 1, 7, 0, 3, 1, 1, 36));                // 1x7 shift g36 (no bias)
    ops.push_back(conv(n("hsh"), n("hc"),  n("mlp_h.weight"), n("mlp_h.bias"), 1, 1, 0, 0, 1, 1, 1));          // 1x1 36->36
    ops.push_back(conv(n("hc"),  n("wsh"), n("mlp_w.shift_weight"), "", 7, 1, 3, 0, 1, 1, 36));                // 7x1 shift g36 (no bias)
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
    ops.push_back(conv(gin, n("x2"), n("conv_g.weight"), n("conv_g.bias"), 3, 3, 1, 1, 1, 1, 1)); // 6->dim
    ops.push_back(conv(xin, n("x1"), n("conv_x.weight"), n("conv_x.bias"), 3, 3, 1, 1, 1, 1, 1)); // dim->dim
    ops.push_back(Op{ "gelu", { n("x2") }, { n("gx2") } });
    ops.push_back(Op{ "gelu", { n("x1") }, { n("gx1") } });
    ops.push_back(Op{ "mul", { n("x1"), n("gx2") }, { n("a") } });                                // x1 * gelu(x2)
    ops.push_back(Op{ "mul", { n("x2"), n("gx1") }, { n("b") } });                                // x2 * gelu(x1)
    ops.push_back(Op{ "concat", { n("a"), n("b") }, { n("cat") } });                              // -> 2*dim
    ops.push_back(conv(n("cat"), out, n("proj_out.weight"), n("proj_out.bias"), 1, 1, 0, 0, 1, 1, 1)); // 2*dim->dim
}

std::vector<Op> programDFM() { std::vector<Op> ops; appendDFM(ops, "", "input", "output"); return ops; }
std::vector<Op> programCCM() { std::vector<Op> ops; appendCCM(ops, "", "input", "output"); return ops; }
std::vector<Op> programHFB() { std::vector<Op> ops; appendHFB(ops, "", "input", "g", "output"); return ops; }

// EncodeLayer: x = dfm(x) + x; x = ffn(x) + x. (encoders.*.* in rdg_arch.py)
std::vector<Op> programEncodeLayer()
{
    std::vector<Op> ops;
    appendDFM(ops, "dfm.", "input", "dfm_out");
    ops.push_back(Op{ "add", { "input", "dfm_out" }, { "x1" } });   // residual
    appendCCM(ops, "ffn.", "x1", "ffn_out");
    ops.push_back(Op{ "add", { "x1", "ffn_out" }, { "output" } });  // residual
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
    if (bundle.count("dfm.proj_in.0.weight")) { ops = programEncodeLayer(); blockName = "EncodeLayer"; }
    else if (bundle.count("conv_g.weight")) { ops = programHFB(); blockName = "HFB"; }
    else if (bundle.count("proj_in.0.weight")) { ops = programDFM(); blockName = "DFM"; }
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

    // ---- root signature: root constants(b0,13) + SRV t0/t1/t2 + UAV u0 (all root descriptors) ----
    D3D12_ROOT_PARAMETER params[5] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 13;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    for (int i = 1; i <= 3; ++i)
    {
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[i].Descriptor.ShaderRegister = UINT(i - 1);
        params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[4].Descriptor.ShaderRegister = 0;
    params[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

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
    ComPtr<ID3D12PipelineState> psoConv   = makePSO(L"conv2d.hlsl", L"conv2d_CS");
    ComPtr<ID3D12PipelineState> psoGelu   = makePSO(L"gelu.hlsl",   L"gelu_CS");
    ComPtr<ID3D12PipelineState> psoMul     = makePSO(L"mul.hlsl",    L"mul_CS");
    ComPtr<ID3D12PipelineState> psoAdd     = makePSO(L"add.hlsl",    L"add_CS");
    ComPtr<ID3D12PipelineState> psoConcat = makePSO(L"concat.hlsl", L"concat_CS");

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
        if (op.kind == "conv2d")
        {
            uploadWB(op.weight); uploadWB(op.bias);
            out.c = bundle.at(op.weight).dims[0];
            out.h = (a.shape.h + 2 * op.PadH - op.KH) / op.StrideH + 1;
            out.w = (a.shape.w + 2 * op.PadW - op.KW) / op.StrideW + 1;
            pso = psoConv.Get();
        }
        else if (op.kind == "gelu") { out = a.shape; pso = psoGelu.Get(); }
        else if (op.kind == "mul")  { out = a.shape; pso = psoMul.Get(); }
        else if (op.kind == "add")  { out = a.shape; pso = psoAdd.Get(); }
        else if (op.kind == "concat")
        {
            Value b = vals.at(op.ins[1]);
            out = { a.shape.c + b.shape.c, a.shape.h, a.shape.w };
            pso = psoConcat.Get();
        }

        auto outBuf = DefaultBuffer(dev.Get(), out.count() * 4, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        alive.push_back(outBuf);
        rstate[outBuf.Get()] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        vals[op.outs[0]] = { outBuf.Get(), 0, out };

        cl->SetPipelineState(pso);
        uint32_t consts[13] = { a.shape.c, a.shape.h, a.shape.w, out.c,
                                op.KH, op.KW, op.PadH, op.PadW, op.StrideH, op.StrideW, op.Groups,
                                out.h, out.w };
        cl->SetComputeRoot32BitConstants(0, 13, consts, 0);
        cl->SetComputeRootShaderResourceView(1, VA(a));                               // t0 = first input
        if (op.kind == "conv2d")
        {
            cl->SetComputeRootShaderResourceView(2, wbuf.at(op.weight)->GetGPUVirtualAddress());
            cl->SetComputeRootShaderResourceView(3, op.bias.empty() ? zeroBiasVA(out.c)
                                                                    : wbuf.at(op.bias)->GetGPUVirtualAddress());
        }
        else if (op.kind == "mul" || op.kind == "add" || op.kind == "concat")
        {
            cl->SetComputeRootShaderResourceView(2, VA(vals.at(op.ins[1])));          // t1 = second input
        }
        cl->SetComputeRootUnorderedAccessView(4, outBuf->GetGPUVirtualAddress());
        cl->Dispatch((out.w + 7) / 8, (out.h + 7) / 8, out.c);

        std::printf("  %-6s %-8s -> %-8s (%u,%u,%u)%s\n", op.kind.c_str(), op.ins[0].c_str(),
                    op.outs[0].c_str(), out.c, out.h, out.w,
                    (op.kind == "conv2d" && op.Groups > 1) ? "  (grouped)" : "");
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
