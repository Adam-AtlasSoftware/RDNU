// rdnu_harness.cpp - standalone DX12 compute validation harness for RDNU layers.
//
// Phase 0: runs the `first_conv` compute shader on the golden input and compares the
// GPU result to the PyTorch golden output (both carried in a .rdnut bundle produced by
// runtime/tools/dump_golden.py). Decoupled from Cauldron so we can validate numerically,
// layer by layer, with a ~1s console loop.
//
// Usage: rdnu_harness <bundle.rdnut> [shader.hlsl]
//   shader.hlsl defaults to first_conv.hlsl next to the executable.
//
// Uses plain Windows-SDK D3D12 (no d3dx12.h helper, to stay independent of the newer
// DirectX-Headers submodule / Agility SDK).

#include "rdnu_dx12.h"

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: rdnu_harness <bundle.rdnut> [shader.hlsl]\n"); return 1; }

    // Optional args: [shader.hlsl] [entryPoint]. Default to the general conv2d kernel.
    std::wstring shaderPath, entry = L"conv2d_CS";
    if (argc >= 3)
    {
        std::string s(argv[2]); shaderPath.assign(s.begin(), s.end());
    }
    else
    {
        wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring dir(exe); dir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
        shaderPath = dir + L"conv2d.hlsl";
    }
    if (argc >= 4) { std::string s(argv[3]); entry.assign(s.begin(), s.end()); }

    auto bundle = LoadRdnut(argv[1]);
    const Tensor& input  = bundle.at("input");       // (Cin, H, W)
    const Tensor& golden = bundle.at("golden_out");  // (Cout, OH, OW)
    // weight/bias are optional: conv ops carry them, pointwise ops (e.g. GELU) do not.
    const Tensor* weight = bundle.count("weight") ? &bundle.at("weight") : nullptr;
    const Tensor* bias   = bundle.count("bias")   ? &bundle.at("bias")   : nullptr;

    const uint32_t Cin = input.dims[0], H = input.dims[1], W = input.dims[2];
    // Output channels + spatial dims all come from the golden tensor (Cout, OH, OW), so
    // strided/grouped convs and paramless pointwise ops all size correctly.
    const size_t gnd = golden.dims.size();
    const uint32_t Cout = golden.dims[gnd - 3];
    const uint32_t OH   = golden.dims[gnd - 2];
    const uint32_t OW   = golden.dims[gnd - 1];

    // Conv params record: [KH, KW, PadH, PadW, StrideH, StrideW, Groups].
    // Absent (older bundles) -> default 3x3 / pad 1 / stride 1 / groups 1 (first_conv).
    uint32_t KH = 3, KW = 3, PadH = 1, PadW = 1, StrideH = 1, StrideW = 1, Groups = 1;
    if (auto it = bundle.find("params"); it != bundle.end() && it->second.data.size() >= 7)
    {
        const auto& p = it->second.data;
        auto U = [](float f) { return uint32_t(std::lround(f)); };
        KH = U(p[0]); KW = U(p[1]); PadH = U(p[2]); PadW = U(p[3]);
        StrideH = U(p[4]); StrideW = U(p[5]); Groups = U(p[6]);
    }
    std::printf("conv: Cin=%u H=%u W=%u -> Cout=%u OH=%u OW=%u  k%ux%u s%ux%u p%ux%u g%u\n",
        Cin, H, W, Cout, OH, OW, KH, KW, StrideH, StrideW, PadH, PadW, Groups);

    // ---- device (prefer the high-performance / discrete adapter) ----
    UINT flags = 0;
#if defined(_DEBUG)
    { ComPtr<ID3D12Debug> dbg; if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) dbg->EnableDebugLayer(); flags |= DXGI_CREATE_FACTORY_DEBUG; }
#endif
    ComPtr<IDXGIFactory6> factory;
    Check(CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
    ComPtr<IDXGIAdapter1> adapter;
    Check(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)),
          "EnumAdapterByGpuPreference");
    DXGI_ADAPTER_DESC1 ad{}; adapter->GetDesc1(&ad);
    std::wprintf(L"adapter: %s\n", ad.Description);

    ComPtr<ID3D12Device> dev;
    Check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)), "D3D12CreateDevice");

    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    Check(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
    ComPtr<ID3D12CommandAllocator> alloc;
    Check(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc)), "CreateCommandAllocator");
    ComPtr<ID3D12GraphicsCommandList> cl;
    Check(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&cl)), "CreateCommandList");

    // ---- resources (inputs via upload heap bound as root SRVs; output default UAV) ----
    auto inBuf = UploadBuffer(dev.Get(), input.data.data(), input.data.size() * 4);
    ComPtr<ID3D12Resource> wBuf, bBuf;
    if (weight) wBuf = UploadBuffer(dev.Get(), weight->data.data(), weight->data.size() * 4);
    if (bias)   bBuf = UploadBuffer(dev.Get(), bias->data.data(),   bias->data.size()   * 4);

    const size_t outBytes = size_t(Cout) * OH * OW * 4;
    ComPtr<ID3D12Resource> outBuf;
    {
        auto hp = HeapProps(D3D12_HEAP_TYPE_DEFAULT);
        auto rd = BufferDesc(outBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        Check(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&outBuf)), "CreateCommittedResource(out)");
    }
    ComPtr<ID3D12Resource> readback;
    {
        auto hp = HeapProps(D3D12_HEAP_TYPE_READBACK);
        auto rd = BufferDesc(outBytes);
        Check(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)), "CreateCommittedResource(readback)");
    }

    // ---- root signature: root constants(b0) + SRV t0/t1/t2 + UAV u0 (all root descriptors) ----
    D3D12_ROOT_PARAMETER params[5] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.ShaderRegister = 0;
    params[0].Constants.RegisterSpace = 0;
    params[0].Constants.Num32BitValues = 13;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    for (int i = 1; i <= 3; ++i)
    {
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[i].Descriptor.ShaderRegister = UINT(i - 1);
        params[i].Descriptor.RegisterSpace = 0;
        params[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[4].Descriptor.ShaderRegister = 0;
    params[4].Descriptor.RegisterSpace = 0;
    params[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = _countof(params);
    rsDesc.pParameters = params;
    rsDesc.NumStaticSamplers = 0;
    rsDesc.pStaticSamplers = nullptr;
    rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> rsBlob, rsErr;
    Check(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsErr), "SerializeRootSignature");
    ComPtr<ID3D12RootSignature> rootSig;
    Check(dev->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig)), "CreateRootSignature");

    auto dxil = CompileCS(shaderPath, entry.c_str(), L"cs_6_2");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = rootSig.Get();
    pd.CS = { dxil.data(), dxil.size() };
    ComPtr<ID3D12PipelineState> pso;
    Check(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)), "CreateComputePipelineState");

    // ---- record ----
    cl->SetPipelineState(pso.Get());
    cl->SetComputeRootSignature(rootSig.Get());
    uint32_t consts[13] = { Cin, H, W, Cout, KH, KW, PadH, PadW, StrideH, StrideW, Groups, OH, OW };
    cl->SetComputeRoot32BitConstants(0, 13, consts, 0);
    cl->SetComputeRootShaderResourceView(1, inBuf->GetGPUVirtualAddress());
    if (weight) cl->SetComputeRootShaderResourceView(2, wBuf->GetGPUVirtualAddress());
    if (bias)   cl->SetComputeRootShaderResourceView(3, bBuf->GetGPUVirtualAddress());
    cl->SetComputeRootUnorderedAccessView(4, outBuf->GetGPUVirtualAddress());
    cl->Dispatch((OW + 7) / 8, (OH + 7) / 8, Cout);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = outBuf.Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cl->ResourceBarrier(1, &barrier);
    cl->CopyResource(readback.Get(), outBuf.Get());
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

    // ---- compare ----
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

    std::printf("max|err| = %.6g   mean|err| = %.6g   (golden max|v| = %.4g)\n",
        maxAbs, sumAbs / double(n), gmax);
    const double tol = 1e-3;
    std::printf("%s (tol %.0e)\n", maxAbs < tol ? "PASS" : "FAIL", tol);
    return maxAbs < tol ? 0 : 2;
}
