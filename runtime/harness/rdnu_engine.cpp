// rdnu_engine.cpp - minimal on-GPU execution graph for a whole RDNU block.
//
// Where rdnu_harness validates ONE kernel per run, the engine chains several kernels on
// the GPU through intermediate ("ping-pong") feature buffers, with no CPU round-trips
// between layers, and validates the block's end-to-end result against the PyTorch golden.
//
// This first version runs the CCM/FFN block: conv3x3 -> GELU -> conv1x1. The op list is
// built from the bundle (weights + per-conv params) and executed as a small graph; it is
// the seed of the metadata-driven engine that will later run whole models. Correctness
// first: same plain FP32 conv2d.hlsl / gelu.hlsl kernels the primitives were validated with.
//
// Usage: rdnu_engine <block.rdnut> [shaderDir]
//   shaderDir defaults to the executable's directory (where conv2d.hlsl / gelu.hlsl live).

#include "rdnu_dx12.h"

#include <array>

namespace
{
uint32_t U(float f) { return uint32_t(std::lround(f)); }

struct Shape { uint32_t c, h, w; size_t count() const { return size_t(c) * h * w; } };

// One node of the block graph. Feature tensors are referenced by name; weights/biases
// name entries in the bundle. Conv fields are unused for pointwise ops (kind == "gelu").
struct Op
{
    std::string kind;               // "conv2d" | "gelu"
    std::string in, out;            // feature-buffer names
    std::string weight, bias;       // conv only
    uint32_t KH, KW, PadH, PadW, StrideH, StrideW, Groups;
};
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: rdnu_engine <block.rdnut> [shaderDir]\n"); return 1; }

    std::wstring shaderDir;
    if (argc >= 3) { std::string s(argv[2]); shaderDir.assign(s.begin(), s.end()); if (shaderDir.back() != L'\\' && shaderDir.back() != L'/') shaderDir += L'\\'; }
    else
    {
        wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring dir(exe); shaderDir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
    }

    auto bundle = LoadRdnut(argv[1]);
    const Tensor& input  = bundle.at("input");
    const Tensor& golden = bundle.at("golden_out");

    // ---- build the CCM/FFN block graph (conv3x3 -> GELU -> conv1x1) ----
    const auto& p0 = bundle.at("fn.0.params").data;
    const auto& p2 = bundle.at("fn.2.params").data;
    std::vector<Op> ops = {
        { "conv2d", "input", "h0",     "fn.0.weight", "fn.0.bias",
          U(p0[0]), U(p0[1]), U(p0[2]), U(p0[3]), U(p0[4]), U(p0[5]), U(p0[6]) },
        { "gelu",   "h0",    "h1",     "",            "",
          0, 0, 0, 0, 0, 0, 0 },
        { "conv2d", "h1",    "output", "fn.2.weight", "fn.2.bias",
          U(p2[0]), U(p2[1]), U(p2[2]), U(p2[3]), U(p2[4]), U(p2[5]), U(p2[6]) },
    };

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
    ComPtr<ID3D12PipelineState> psoConv = makePSO(L"conv2d.hlsl", L"conv2d_CS");
    ComPtr<ID3D12PipelineState> psoGelu = makePSO(L"gelu.hlsl",  L"gelu_CS");

    // ---- allocate buffers: input + weights (upload SRVs), intermediates + output (default UAVs) ----
    std::map<std::string, ComPtr<ID3D12Resource>> feat;    // feature buffers by name
    std::map<std::string, Shape> shape;
    std::map<std::string, D3D12_RESOURCE_STATES> state;    // current state of default buffers

    feat["input"] = UploadBuffer(dev.Get(), input.data.data(), input.data.size() * 4);
    shape["input"] = { input.dims[0], input.dims[1], input.dims[2] };

    std::map<std::string, ComPtr<ID3D12Resource>> wbuf;    // weight/bias upload buffers
    auto upload = [&](const std::string& name) {
        if (!name.empty() && !wbuf.count(name))
        {
            const Tensor& t = bundle.at(name);
            wbuf[name] = UploadBuffer(dev.Get(), t.data.data(), t.data.size() * 4);
        }
    };

    // Walk the graph: derive each op's output shape, allocate its (default, UAV) buffer.
    for (const Op& op : ops)
    {
        Shape in = shape.at(op.in);
        Shape out;
        if (op.kind == "conv2d")
        {
            out.c = bundle.at(op.weight).dims[0];
            out.h = (in.h + 2 * op.PadH - op.KH) / op.StrideH + 1;
            out.w = (in.w + 2 * op.PadW - op.KW) / op.StrideW + 1;
            upload(op.weight); upload(op.bias);
        }
        else // gelu (pointwise)
        {
            out = in;
        }
        shape[op.out] = out;
        feat[op.out] = DefaultBuffer(dev.Get(), out.count() * 4, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        state[op.out] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }

    Shape outShape = shape.at("output");
    const size_t outBytes = outShape.count() * 4;
    ComPtr<ID3D12Resource> readback = ReadbackBuffer(dev.Get(), outBytes);

    // ---- record the graph ----
    cl->SetComputeRootSignature(rootSig.Get());
    for (size_t i = 0; i < ops.size(); ++i)
    {
        const Op& op = ops[i];
        Shape in = shape.at(op.in), out = shape.at(op.out);

        cl->SetPipelineState(op.kind == "conv2d" ? psoConv.Get() : psoGelu.Get());
        uint32_t consts[13] = { in.c, in.h, in.w, out.c,
                                op.KH, op.KW, op.PadH, op.PadW, op.StrideH, op.StrideW, op.Groups,
                                out.h, out.w };
        cl->SetComputeRoot32BitConstants(0, 13, consts, 0);
        cl->SetComputeRootShaderResourceView(1, feat.at(op.in)->GetGPUVirtualAddress());
        if (!op.weight.empty()) cl->SetComputeRootShaderResourceView(2, wbuf.at(op.weight)->GetGPUVirtualAddress());
        if (!op.bias.empty())   cl->SetComputeRootShaderResourceView(3, wbuf.at(op.bias)->GetGPUVirtualAddress());
        cl->SetComputeRootUnorderedAccessView(4, feat.at(op.out)->GetGPUVirtualAddress());
        cl->Dispatch((out.w + 7) / 8, (out.h + 7) / 8, out.c);

        // Hand this op's output to its consumer: SRV for the next op, or COPY_SOURCE for the
        // final readback. (Linear chain: each intermediate is produced then read by the next op.)
        bool last = (i + 1 == ops.size());
        D3D12_RESOURCE_STATES nextState = last ? D3D12_RESOURCE_STATE_COPY_SOURCE
                                               : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        Transition(cl.Get(), feat.at(op.out).Get(), state.at(op.out), nextState);
        state[op.out] = nextState;

        std::printf("  op[%zu] %-6s  in(%u,%u,%u) -> out(%u,%u,%u)%s\n", i, op.kind.c_str(),
                    in.c, in.h, in.w, out.c, out.h, out.w,
                    op.kind == "conv2d" ? (op.Groups > 1 ? "  (grouped)" : "") : "  (pointwise)");
    }

    cl->CopyResource(readback.Get(), feat.at("output").Get());
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
        double a = std::fabs(golden.data[i]); if (a > gmax) gmax = a;
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
