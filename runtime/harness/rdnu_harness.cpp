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

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxcapi.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <fstream>

using Microsoft::WRL::ComPtr;

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

static void Check(HRESULT hr, const char* what)
{
    if (FAILED(hr)) { std::printf("FATAL: %s (hr=0x%08X)\n", what, (unsigned)hr); std::exit(1); }
}

// -------------------------------------------------------------- D3D12 helpers
static D3D12_HEAP_PROPERTIES HeapProps(D3D12_HEAP_TYPE t)
{
    D3D12_HEAP_PROPERTIES h{};
    h.Type = t;
    h.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    h.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    h.CreationNodeMask = 1;
    h.VisibleNodeMask = 1;
    return h;
}

static D3D12_RESOURCE_DESC BufferDesc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
{
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Alignment = 0;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN;
    d.SampleDesc.Count = 1;
    d.SampleDesc.Quality = 0;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = flags;
    return d;
}

// ------------------------------------------------------------------ .rdnut IO
struct Tensor
{
    std::vector<uint32_t> dims;
    std::vector<float>    data;
    size_t count() const { size_t n = 1; for (auto d : dims) n *= d; return n; }
};

static std::map<std::string, Tensor> LoadRdnut(const char* path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::printf("FATAL: cannot open %s\n", path); std::exit(1); }
    char magic[4]; f.read(magic, 4);
    if (std::memcmp(magic, "RDNT", 4) != 0) { std::printf("FATAL: bad magic in %s\n", path); std::exit(1); }
    uint32_t ver = 0, n = 0;
    f.read(reinterpret_cast<char*>(&ver), 4);
    f.read(reinterpret_cast<char*>(&n), 4);
    std::map<std::string, Tensor> out;
    for (uint32_t i = 0; i < n; ++i)
    {
        uint32_t nl = 0; f.read(reinterpret_cast<char*>(&nl), 4);
        std::string name(nl, '\0'); f.read(&name[0], nl);
        uint32_t nd = 0; f.read(reinterpret_cast<char*>(&nd), 4);
        Tensor t; t.dims.resize(nd); f.read(reinterpret_cast<char*>(t.dims.data()), std::streamsize(nd) * 4);
        t.data.resize(t.count()); f.read(reinterpret_cast<char*>(t.data.data()), std::streamsize(t.count()) * 4);
        out[name] = std::move(t);
    }
    return out;
}

// -------------------------------------------------------------- shader compile
static std::vector<uint8_t> CompileCS(const std::wstring& file, const wchar_t* entry, const wchar_t* profile)
{
    HMODULE lib = LoadLibraryW(L"dxcompiler.dll");
    if (!lib) { std::printf("FATAL: dxcompiler.dll not found next to the exe\n"); std::exit(1); }
    auto pCreate = reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(lib, "DxcCreateInstance"));

    ComPtr<IDxcUtils> utils; ComPtr<IDxcCompiler3> comp;
    Check(pCreate(CLSID_DxcUtils, IID_PPV_ARGS(&utils)), "DxcCreateInstance(Utils)");
    Check(pCreate(CLSID_DxcCompiler, IID_PPV_ARGS(&comp)), "DxcCreateInstance(Compiler)");

    ComPtr<IDxcBlobEncoding> src;
    Check(utils->LoadFile(file.c_str(), nullptr, &src), "LoadFile(shader)");
    DxcBuffer buf{ src->GetBufferPointer(), src->GetBufferSize(), DXC_CP_ACP };

    LPCWSTR args[] = { L"-T", profile, L"-E", entry, L"-O3", L"-enable-16bit-types" };
    ComPtr<IDxcResult> res;
    Check(comp->Compile(&buf, args, _countof(args), nullptr, IID_PPV_ARGS(&res)), "Compile");

    ComPtr<IDxcBlobUtf8> errs;
    res->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errs), nullptr);
    if (errs && errs->GetStringLength()) std::printf("DXC: %s\n", errs->GetStringPointer());
    HRESULT st = S_OK; res->GetStatus(&st);
    if (FAILED(st)) { std::printf("FATAL: shader compilation failed\n"); std::exit(1); }

    ComPtr<IDxcBlob> obj;
    res->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&obj), nullptr);
    auto* p = reinterpret_cast<uint8_t*>(obj->GetBufferPointer());
    return std::vector<uint8_t>(p, p + obj->GetBufferSize());
}

// ------------------------------------------------------------------- buffers
static ComPtr<ID3D12Resource> UploadBuffer(ID3D12Device* dev, const void* data, size_t bytes)
{
    ComPtr<ID3D12Resource> r;
    auto hp = HeapProps(D3D12_HEAP_TYPE_UPLOAD);
    auto rd = BufferDesc(bytes);
    Check(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&r)), "CreateCommittedResource(upload)");
    void* p = nullptr; D3D12_RANGE none{ 0, 0 };
    r->Map(0, &none, &p); std::memcpy(p, data, bytes); r->Unmap(0, nullptr);
    return r;
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("usage: rdnu_harness <bundle.rdnut> [shader.hlsl]\n"); return 1; }

    std::wstring shaderPath;
    if (argc >= 3)
    {
        std::string s(argv[2]); shaderPath.assign(s.begin(), s.end());
    }
    else
    {
        wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
        std::wstring dir(exe); dir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
        shaderPath = dir + L"first_conv.hlsl";
    }

    auto bundle = LoadRdnut(argv[1]);
    const Tensor& input  = bundle.at("input");       // (Cin, H, W)
    const Tensor& weight = bundle.at("weight");      // (Cout, Cin, 3, 3)
    const Tensor& bias   = bundle.at("bias");        // (Cout)
    const Tensor& golden = bundle.at("golden_out");  // (Cout, H, W)

    const uint32_t Cin = input.dims[0], H = input.dims[1], W = input.dims[2];
    const uint32_t Cout = weight.dims[0];
    std::printf("first_conv: Cin=%u H=%u W=%u Cout=%u\n", Cin, H, W, Cout);

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
    auto inBuf = UploadBuffer(dev.Get(), input.data.data(),  input.data.size()  * 4);
    auto wBuf  = UploadBuffer(dev.Get(), weight.data.data(), weight.data.size() * 4);
    auto bBuf  = UploadBuffer(dev.Get(), bias.data.data(),   bias.data.size()   * 4);

    const size_t outBytes = size_t(Cout) * H * W * 4;
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
    params[0].Constants.Num32BitValues = 4;
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

    auto dxil = CompileCS(shaderPath, L"first_conv_CS", L"cs_6_2");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = rootSig.Get();
    pd.CS = { dxil.data(), dxil.size() };
    ComPtr<ID3D12PipelineState> pso;
    Check(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)), "CreateComputePipelineState");

    // ---- record ----
    cl->SetPipelineState(pso.Get());
    cl->SetComputeRootSignature(rootSig.Get());
    uint32_t dims[4] = { Cin, H, W, Cout };
    cl->SetComputeRoot32BitConstants(0, 4, dims, 0);
    cl->SetComputeRootShaderResourceView(1, inBuf->GetGPUVirtualAddress());
    cl->SetComputeRootShaderResourceView(2, wBuf->GetGPUVirtualAddress());
    cl->SetComputeRootShaderResourceView(3, bBuf->GetGPUVirtualAddress());
    cl->SetComputeRootUnorderedAccessView(4, outBuf->GetGPUVirtualAddress());
    cl->Dispatch((W + 7) / 8, (H + 7) / 8, Cout);

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
