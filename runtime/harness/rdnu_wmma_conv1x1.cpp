// rdnu_wmma_conv1x1.cpp - WMMA-accelerated INT8 1x1 conv, validated vs the conv2d_int8 golden.
//
// Runs the 3-pass wmma_conv1x1.hlsl (quantize -> WMMA GEMM -> dequantize) on an int8 1x1-conv
// bundle (e.g. int8_ffn_1x1.rdnut: Cin=72 -> Cout=36, N=64*64) and compares to the numpy int8
// golden already validated for conv2d_int8. Proves WMMA on a real conv layer.
//
// Usage: rdnu_wmma_conv1x1 <int8_1x1_bundle.rdnut> [shaderDir]

#include "rdnu_dx12.h"

#ifndef RDNU_AGS_INC
#define RDNU_AGS_INC ""
#endif
static const uint32_t AGS_SPACE = 2147420894u;

static void UavBarrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* r)
{
    D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; b.UAV.pResource = r;
    cl->ResourceBarrier(1, &b);
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: rdnu_wmma_conv1x1 <bundle.rdnut> [shaderDir]\n"); return 1; }
    std::wstring shaderDir;
    if (argc >= 3) { std::string s(argv[2]); shaderDir.assign(s.begin(), s.end()); if (shaderDir.back() != L'\\' && shaderDir.back() != L'/') shaderDir += L'\\'; }
    else { wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH); std::wstring d(exe); shaderDir = d.substr(0, d.find_last_of(L"\\/") + 1); }
    std::string agsA = RDNU_AGS_INC; std::wstring agsInc(agsA.begin(), agsA.end());

    auto bundle = LoadRdnut(argv[1]);
    const Tensor& input  = bundle.at("input");       // (Cin, H, W) fp32
    const Tensor& wi8    = bundle.at("i8.w");        // (Cout,1,1,Cin) fp32 (int8 values)
    const Tensor& wscale = bundle.at("i8.scale");    // (Cout)
    const Tensor& ascale = bundle.at("i8.ascale");   // (1)
    const Tensor& bias   = bundle.at("i8.bias");     // (Cout)
    const Tensor& golden = bundle.at("golden_out");  // (Cout, OH, OW)

    const uint32_t Cin = input.dims[0], H = input.dims[1], W = input.dims[2];
    const uint32_t Cout = wi8.dims[0], N = H * W;
    const uint32_t Cin_pad = (Cin + 15) / 16 * 16, Cout_pad = (Cout + 15) / 16 * 16;
    std::printf("wmma conv1x1: Cin=%u Cout=%u N=%u  (padded %u x %u)\n", Cin, Cout, N, Cout_pad, Cin_pad);

    // pack weights -> int8 [Cout_pad * Cin_pad], oc-major, zero-padded
    std::vector<int8_t> wpack(size_t(Cout_pad) * Cin_pad, 0);
    for (uint32_t oc = 0; oc < Cout; ++oc)
        for (uint32_t ic = 0; ic < Cin; ++ic)
            wpack[oc * Cin_pad + ic] = int8_t(std::lround(wi8.data[oc * Cin + ic]));

    ComPtr<ID3D12Device> dev = CreateHighPerfDevice();
    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    Check(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
    ComPtr<ID3D12CommandAllocator> alloc;
    Check(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc)), "CreateCommandAllocator");
    ComPtr<ID3D12GraphicsCommandList> cl;
    Check(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&cl)), "CreateCommandList");

    // root sig: b0(8) + SRV t0..t3 + UAV u1,u2,u3 + UAV u0@AGS
    D3D12_ROOT_PARAMETER p[9] = {};
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; p[0].Constants.ShaderRegister = 0; p[0].Constants.Num32BitValues = 8;
    for (int i = 0; i < 4; ++i) { p[1 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[1 + i].Descriptor.ShaderRegister = i; }
    for (int i = 0; i < 3; ++i) { p[5 + i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; p[5 + i].Descriptor.ShaderRegister = 1 + i; }
    p[8].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; p[8].Descriptor.ShaderRegister = 0; p[8].Descriptor.RegisterSpace = AGS_SPACE;
    for (auto& q : p) q.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rs{}; rs.NumParameters = 9; rs.pParameters = p;
    ComPtr<ID3DBlob> rsBlob, rsErr;
    Check(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsErr), "SerializeRootSignature");
    ComPtr<ID3D12RootSignature> rootSig;
    Check(dev->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig)), "CreateRootSignature");

    auto makePSO = [&](const wchar_t* entry) {
        auto dxil = CompileCS(shaderDir + L"wmma_conv1x1.hlsl", entry, L"cs_6_6", agsInc);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{}; pd.pRootSignature = rootSig.Get(); pd.CS = { dxil.data(), dxil.size() };
        ComPtr<ID3D12PipelineState> pso; Check(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)), "PSO"); return pso;
    };
    std::printf("compiling wmma_conv1x1.hlsl (quant/gemm/dequant)...\n");
    auto psoQuant = makePSO(L"quant_CS");
    auto psoGemm  = makePSO(L"gemm_CS");
    auto psoDeq   = makePSO(L"dequant_CS");
    std::printf("compile OK\n");

    auto actBuf = UploadBuffer(dev.Get(), input.data.data(), input.data.size() * 4);
    auto wBuf   = UploadBuffer(dev.Get(), wpack.data(), wpack.size());
    auto wsBuf  = UploadBuffer(dev.Get(), wscale.data.data(), wscale.data.size() * 4);
    auto bBuf   = UploadBuffer(dev.Get(), bias.data.data(), bias.data.size() * 4);
    auto qactBuf = DefaultBuffer(dev.Get(), size_t(Cin_pad) * N, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto caccBuf = DefaultBuffer(dev.Get(), size_t(Cout_pad) * N * 4, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const size_t outBytes = size_t(Cout) * N * 4;
    auto outBuf = DefaultBuffer(dev.Get(), outBytes, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto agsBuf = DefaultBuffer(dev.Get(), 65536, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto rb = ReadbackBuffer(dev.Get(), outBytes);

    uint32_t as; std::memcpy(&as, &ascale.data[0], 4);
    uint32_t consts[8] = { Cin, N, Cin_pad, Cout_pad, Cout, H, W, as };

    cl->SetComputeRootSignature(rootSig.Get());
    cl->SetComputeRoot32BitConstants(0, 8, consts, 0);
    cl->SetComputeRootShaderResourceView(1, actBuf->GetGPUVirtualAddress());
    cl->SetComputeRootShaderResourceView(2, wBuf->GetGPUVirtualAddress());
    cl->SetComputeRootShaderResourceView(3, wsBuf->GetGPUVirtualAddress());
    cl->SetComputeRootShaderResourceView(4, bBuf->GetGPUVirtualAddress());
    cl->SetComputeRootUnorderedAccessView(5, qactBuf->GetGPUVirtualAddress());
    cl->SetComputeRootUnorderedAccessView(6, caccBuf->GetGPUVirtualAddress());
    cl->SetComputeRootUnorderedAccessView(7, outBuf->GetGPUVirtualAddress());
    cl->SetComputeRootUnorderedAccessView(8, agsBuf->GetGPUVirtualAddress());

    cl->SetPipelineState(psoQuant.Get());
    cl->Dispatch((N / 4 + 63) / 64, Cin_pad, 1);
    UavBarrier(cl.Get(), qactBuf.Get());
    cl->SetPipelineState(psoGemm.Get());
    cl->Dispatch(N / 16, Cout_pad / 16, 1);
    UavBarrier(cl.Get(), caccBuf.Get());
    cl->SetPipelineState(psoDeq.Get());
    cl->Dispatch((N + 63) / 64, Cout, 1);

    Transition(cl.Get(), outBuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cl->CopyResource(rb.Get(), outBuf.Get());
    Check(cl->Close(), "Close");
    ID3D12CommandList* lists[] = { cl.Get() };
    queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence;
    Check(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    Check(queue->Signal(fence.Get(), 1), "Signal");
    fence->SetEventOnCompletion(1, ev); WaitForSingleObject(ev, INFINITE); CloseHandle(ev);

    float* gpu = nullptr; D3D12_RANGE rng{ 0, outBytes };
    Check(rb->Map(0, &rng, reinterpret_cast<void**>(&gpu)), "Map(readback)");
    double maxAbs = 0, sumAbs = 0, gmax = 0; size_t n = golden.count();
    for (size_t i = 0; i < n; ++i) { double d = std::fabs(double(gpu[i]) - double(golden.data[i])); if (d > maxAbs) maxAbs = d; sumAbs += d; double a = std::fabs(golden.data[i]); if (a > gmax) gmax = a; }
    std::printf("golden[0..2] = %.5f %.5f %.5f   gpu[0..2] = %.5f %.5f %.5f\n", golden.data[0], golden.data[1], golden.data[2], gpu[0], gpu[1], gpu[2]);
    rb->Unmap(0, nullptr);
    std::printf("WMMA int8 1x1 conv: max|err| = %.6g   mean|err| = %.6g   (golden max|v| = %.4g)\n", maxAbs, sumAbs / double(n), gmax);
    const double tol = 1e-3;
    std::printf("%s (tol %.0e)\n", maxAbs < tol ? "PASS" : "FAIL", tol);
    return maxAbs < tol ? 0 : 2;
}
