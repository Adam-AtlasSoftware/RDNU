// rdnu_wmma_probe.cpp - de-risk the RDNA3 WMMA path with a single 16x16x16 INT8 GEMM.
//
// Compiles wmma_gemm.hlsl (AMD AGS wave-matrix intrinsics, Wave32) with the vendored
// wave-matrix-capable dxc, runs one tile on the 7900 XTX, and checks C == A*B vs a plain
// C++ int8 matmul. If this passes, the whole WMMA toolchain (compiler + magic UAV + hardware)
// works and we can build the tiled INT8 conv on top. Uses the shared DX12 helpers.
//
// The AGS intrinsics need a magic mailbox UAV bound at (u0, space2147420894); the driver
// intercepts operations on it. RDNU_AGS_INC (from CMake) is the include dir for the AGS header.

#include "rdnu_dx12.h"

#ifndef RDNU_AGS_INC
#define RDNU_AGS_INC ""
#endif

static const uint32_t AGS_SPACE = 2147420894u;   // space2147420894

int main(int argc, char** argv)
{
    std::wstring shaderDir;
    {
        wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring dir(exe); shaderDir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
    }
    std::string agsIncA = RDNU_AGS_INC;
    std::wstring agsInc(agsIncA.begin(), agsIncA.end());

    // ---- reference matrices + golden (row-major 16x16) ----
    int8_t A[256], B[256];
    for (int i = 0; i < 16; ++i)
        for (int k = 0; k < 16; ++k)
        {
            A[i * 16 + k] = int8_t(((i * 3 + k * 5) % 13) - 6);
            B[i * 16 + k] = int8_t(((i * 7 + k * 2) % 11) - 5);
        }
    int32_t golden[256];
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
        {
            int32_t acc = 0;
            for (int k = 0; k < 16; ++k) acc += int32_t(A[i * 16 + k]) * int32_t(B[k * 16 + j]);
            golden[i * 16 + j] = acc;
        }

    ComPtr<ID3D12Device> dev = CreateHighPerfDevice();
    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    Check(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
    ComPtr<ID3D12CommandAllocator> alloc;
    Check(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc)), "CreateCommandAllocator");
    ComPtr<ID3D12GraphicsCommandList> cl;
    Check(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&cl)), "CreateCommandList");

    // ---- root signature: SRV t0(A), t1(B); UAV u1(C); UAV u0@AGS_SPACE (mailbox) ----
    D3D12_ROOT_PARAMETER p[4] = {};
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[0].Descriptor.ShaderRegister = 0; p[0].Descriptor.RegisterSpace = 0;
    p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; p[1].Descriptor.ShaderRegister = 1; p[1].Descriptor.RegisterSpace = 0;
    p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; p[2].Descriptor.ShaderRegister = 1; p[2].Descriptor.RegisterSpace = 0;
    p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; p[3].Descriptor.ShaderRegister = 0; p[3].Descriptor.RegisterSpace = AGS_SPACE;
    for (auto& q : p) q.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rs{}; rs.NumParameters = 4; rs.pParameters = p;
    ComPtr<ID3DBlob> rsBlob, rsErr;
    HRESULT hr = D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsErr);
    if (FAILED(hr)) { std::printf("FATAL: SerializeRootSignature (%s)\n", rsErr ? (char*)rsErr->GetBufferPointer() : "?"); return 1; }
    ComPtr<ID3D12RootSignature> rootSig;
    Check(dev->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig)), "CreateRootSignature");

    std::printf("compiling wmma_gemm.hlsl (AGS wave-matrix, Wave32, cs_6_6)...\n");
    auto dxil = CompileCS(shaderDir + L"wmma_gemm.hlsl", L"wmma_gemm_CS", L"cs_6_6", agsInc);
    std::printf("compile OK (%zu bytes DXIL)\n", dxil.size());

    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{}; pd.pRootSignature = rootSig.Get(); pd.CS = { dxil.data(), dxil.size() };
    ComPtr<ID3D12PipelineState> pso;
    Check(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)), "CreateComputePipelineState");

    auto aBuf = UploadBuffer(dev.Get(), A, sizeof(A));
    auto bBuf = UploadBuffer(dev.Get(), B, sizeof(B));
    auto cBuf = DefaultBuffer(dev.Get(), sizeof(golden), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto ags  = DefaultBuffer(dev.Get(), 65536, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);   // AGS mailbox
    auto rb   = ReadbackBuffer(dev.Get(), sizeof(golden));

    cl->SetPipelineState(pso.Get());
    cl->SetComputeRootSignature(rootSig.Get());
    cl->SetComputeRootShaderResourceView(0, aBuf->GetGPUVirtualAddress());
    cl->SetComputeRootShaderResourceView(1, bBuf->GetGPUVirtualAddress());
    cl->SetComputeRootUnorderedAccessView(2, cBuf->GetGPUVirtualAddress());
    cl->SetComputeRootUnorderedAccessView(3, ags->GetGPUVirtualAddress());
    cl->Dispatch(1, 1, 1);

    Transition(cl.Get(), cBuf.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cl->CopyResource(rb.Get(), cBuf.Get());
    Check(cl->Close(), "Close");
    ID3D12CommandList* lists[] = { cl.Get() };
    queue->ExecuteCommandLists(1, lists);

    ComPtr<ID3D12Fence> fence;
    Check(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    Check(queue->Signal(fence.Get(), 1), "Signal");
    fence->SetEventOnCompletion(1, ev); WaitForSingleObject(ev, INFINITE); CloseHandle(ev);

    int32_t* gpu = nullptr; D3D12_RANGE rng{ 0, sizeof(golden) };
    Check(rb->Map(0, &rng, reinterpret_cast<void**>(&gpu)), "Map(readback)");
    int mism = 0; int64_t maxAbs = 0;
    for (int i = 0; i < 256; ++i) { int64_t d = std::llabs(int64_t(gpu[i]) - int64_t(golden[i])); if (d) mism++; if (d > maxAbs) maxAbs = d; }
    std::printf("C[0..3] golden = %d %d %d %d   gpu = %d %d %d %d\n",
        golden[0], golden[1], golden[2], golden[3], gpu[0], gpu[1], gpu[2], gpu[3]);
    rb->Unmap(0, nullptr);

    std::printf("WMMA 16x16x16 INT8 GEMM: mismatches = %d / 256   max|err| = %lld\n", mism, (long long)maxAbs);
    std::printf("%s\n", mism == 0 ? "PASS" : "FAIL");
    return mism == 0 ? 0 : 2;
}
