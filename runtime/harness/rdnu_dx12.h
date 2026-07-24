#pragma once
// rdnu_dx12.h - shared DX12 + .rdnut helpers for the RDNU validation harness/engine.
//
// Plain Windows-SDK D3D12 (no d3dx12.h, to stay independent of the newer DirectX-Headers
// submodule / Agility SDK). All helpers are `inline` so this header can be included by
// multiple translation units (rdnu_harness.cpp, rdnu_engine.cpp) without ODR clashes.

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

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;

inline void Check(HRESULT hr, const char* what)
{
    if (FAILED(hr)) { std::printf("FATAL: %s (hr=0x%08X)\n", what, (unsigned)hr); std::exit(1); }
}

// -------------------------------------------------------------- D3D12 helpers
inline D3D12_HEAP_PROPERTIES HeapProps(D3D12_HEAP_TYPE t)
{
    D3D12_HEAP_PROPERTIES h{};
    h.Type = t;
    h.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    h.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    h.CreationNodeMask = 1;
    h.VisibleNodeMask = 1;
    return h;
}

inline D3D12_RESOURCE_DESC BufferDesc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
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

inline std::map<std::string, Tensor> LoadRdnut(const char* path)
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
inline std::vector<uint8_t> CompileCS(const std::wstring& file, const wchar_t* entry, const wchar_t* profile,
                                     const std::wstring& includeDir = L"")
{
    HMODULE lib = LoadLibraryW(L"dxcompiler.dll");
    if (!lib) { std::printf("FATAL: dxcompiler.dll not found next to the exe\n"); std::exit(1); }
    auto pCreate = reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(lib, "DxcCreateInstance"));

    ComPtr<IDxcUtils> utils; ComPtr<IDxcCompiler3> comp;
    Check(pCreate(CLSID_DxcUtils, IID_PPV_ARGS(&utils)), "DxcCreateInstance(Utils)");
    Check(pCreate(CLSID_DxcCompiler, IID_PPV_ARGS(&comp)), "DxcCreateInstance(Compiler)");
    ComPtr<IDxcIncludeHandler> incl;
    utils->CreateDefaultIncludeHandler(&incl);   // resolves #include (e.g. AMD AGS wave-matrix header)

    ComPtr<IDxcBlobEncoding> src;
    Check(utils->LoadFile(file.c_str(), nullptr, &src), "LoadFile(shader)");
    DxcBuffer buf{ src->GetBufferPointer(), src->GetBufferSize(), DXC_CP_ACP };

    std::wstring inc = L"-I" + includeDir;
    std::vector<LPCWSTR> args = { L"-T", profile, L"-E", entry, L"-O3", L"-enable-16bit-types" };
    if (!includeDir.empty()) args.push_back(inc.c_str());
    ComPtr<IDxcResult> res;
    Check(comp->Compile(&buf, args.data(), UINT(args.size()), incl.Get(), IID_PPV_ARGS(&res)), "Compile");

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
inline ComPtr<ID3D12Resource> UploadBuffer(ID3D12Device* dev, const void* data, size_t bytes)
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

// DEFAULT-heap, UAV-capable buffer (feature maps / op outputs).
inline ComPtr<ID3D12Resource> DefaultBuffer(ID3D12Device* dev, size_t bytes, D3D12_RESOURCE_STATES state)
{
    ComPtr<ID3D12Resource> r;
    auto hp = HeapProps(D3D12_HEAP_TYPE_DEFAULT);
    auto rd = BufferDesc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    Check(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
        state, nullptr, IID_PPV_ARGS(&r)), "CreateCommittedResource(default)");
    return r;
}

inline ComPtr<ID3D12Resource> ReadbackBuffer(ID3D12Device* dev, size_t bytes)
{
    ComPtr<ID3D12Resource> r;
    auto hp = HeapProps(D3D12_HEAP_TYPE_READBACK);
    auto rd = BufferDesc(bytes);
    Check(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&r)), "CreateCommittedResource(readback)");
    return r;
}

// Transition barrier helper.
inline void Transition(ID3D12GraphicsCommandList* cl, ID3D12Resource* res,
                       D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    b.Transition.pResource = res;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cl->ResourceBarrier(1, &b);
}

// Picks the high-performance (discrete) adapter — the RX 7900 XTX on this box — and
// prints its description. Feature level 12_0 is enough for our compute path.
inline ComPtr<ID3D12Device> CreateHighPerfDevice()
{
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
    return dev;
}
