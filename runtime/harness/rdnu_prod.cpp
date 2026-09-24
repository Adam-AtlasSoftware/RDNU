// rdnu_prod.cpp - runs the production NSS network (EngineDx12 with the embedded DXIL) on the
// default D3D12 adapter and checks it against the exporter's integer golden.
//
//   rdnu_prod <golden dir> [--force-dp4a] [--drs] [--time N] [--size WxH]
//
// <golden dir>: nss.rdnm, nss_w8.bin, nss_prod.rdnut, nss_prod_drs.rdnut (nss_export.py).
// Compares every arena tensor, the KPN buffer and the temporal texture byte for byte.
// --time N records N network runs between timestamp queries at the golden size, or at
// --size (random input, no comparison). WMMA is used when the driver enables INT8 wave
// matrix (RDNA3+) unless --force-dp4a. Needs no DXGI, so it also runs under WSL2.
#include "../src/engine/rdnu_amd_ext.h"
#include "../src/engine/rdnu_engine_dx12.h"
#include "../tests/common/rdnut.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

using namespace rdnu;

namespace
{
struct Gpu
{
    ID3D12Device*              device = nullptr;
    ID3D12CommandQueue*        queue  = nullptr;
    ID3D12CommandAllocator*    alloc  = nullptr;
    ID3D12GraphicsCommandList* list   = nullptr;
    ID3D12Fence*               fence  = nullptr;
    uint64_t                   value  = 0;

    bool Init()
    {
        if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))))
            return false;
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        return SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) &&
               SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) &&
               SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&list))) &&
               SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    }

    // Executes the open list, waits, and reopens it.
    bool Submit()
    {
        if (FAILED(list->Close()))
            return false;
        ID3D12CommandList* l[] = {list};
        queue->ExecuteCommandLists(1, l);
        queue->Signal(fence, ++value);
        fence->SetEventOnCompletion(value, nullptr);  // blocks
        alloc->Reset();
        return SUCCEEDED(list->Reset(alloc, nullptr));
    }

    ID3D12Resource* Buffer(uint64_t bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state)
    {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = heap;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width            = bytes;
        rd.Height           = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels        = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags            = heap == D3D12_HEAP_TYPE_DEFAULT ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        ID3D12Resource* r   = nullptr;
        device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&r));
        return r;
    }

    void Barrier(ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b)
    {
        D3D12_RESOURCE_BARRIER x{};
        x.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        x.Transition.pResource   = r;
        x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        x.Transition.StateBefore = a;
        x.Transition.StateAfter  = b;
        list->ResourceBarrier(1, &x);
    }

    // Copies a UAV-state buffer to host memory.
    std::vector<int8_t> Read(ID3D12Resource* r, uint64_t bytes)
    {
        ID3D12Resource* rb = Buffer(bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        Barrier(r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(rb, 0, r, 0, bytes);
        Barrier(r, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Submit();
        std::vector<int8_t> out(bytes);
        void* p = nullptr;
        rb->Map(0, nullptr, &p);
        std::memcpy(out.data(), p, bytes);
        rb->Unmap(0, nullptr);
        rb->Release();
        return out;
    }

    // Copies a UAV-state 2D texture to host memory, rows packed.
    std::vector<int8_t> ReadTexture(ID3D12Resource* t, uint32_t texelBytes)
    {
        D3D12_RESOURCE_DESC                d = t->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT64                             total = 0;
        device->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, &total);
        ID3D12Resource* rb = Buffer(total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource       = rb;
        dst.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = fp;
        src.pResource       = t;
        src.Type            = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        Barrier(t, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        Barrier(t, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Submit();
        std::vector<int8_t> out(size_t(d.Width) * d.Height * texelBytes);
        uint8_t* p = nullptr;
        rb->Map(0, nullptr, reinterpret_cast<void**>(&p));
        for (uint32_t y = 0; y < d.Height; ++y)
            std::memcpy(&out[size_t(y) * d.Width * texelBytes], p + size_t(y) * fp.Footprint.RowPitch, size_t(d.Width) * texelBytes);
        rb->Unmap(0, nullptr);
        rb->Release();
        return out;
    }
};

template <typename Get>
size_t Diff(const char* name, const rdnut::Tensor& want, Get get)
{
    const uint32_t h = want.dims[0], w = want.dims[1], c = want.dims[2];
    size_t         bad = 0;
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (uint32_t k = 0; k < c; ++k)
                bad += get(x, y, k) != int(want.data[(size_t(y) * w + x) * c + k]);
    std::printf("  %-10s %s (%zu of %zu codes differ)\n", name, bad ? "FAIL" : "OK  ", bad, size_t(w) * h * c);
    return bad;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
        return std::printf("usage: rdnu_prod <golden dir> [--force-dp4a] [--drs] [--time N] [--size WxH]\n"), 1;
    std::string dir = argv[1], err;
    bool        forceDp4a = false, drs = false;
    int         timeRuns = 0;
    uint32_t    sizeW = 0, sizeH = 0;
    for (int i = 2; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--force-dp4a"))
            forceDp4a = true;
        else if (!std::strcmp(argv[i], "--drs"))
            drs = true;
        else if (!std::strcmp(argv[i], "--time") && i + 1 < argc)
            timeRuns = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--size") && i + 1 < argc)
            std::sscanf(argv[++i], "%ux%u", &sizeW, &sizeH);
    }

    std::vector<uint8_t> manifestBytes, blob;
    std::map<std::string, rdnut::Tensor> golden, goldenDrs;
    if (!rdnut::ReadBytes(dir + "/nss.rdnm", manifestBytes) || !rdnut::ReadBytes(dir + "/nss_w8.bin", blob) ||
        !rdnut::Load(dir + "/nss_prod.rdnut", golden, err) || (drs && !rdnut::Load(dir + "/nss_prod_drs.rdnut", goldenDrs, err)))
        return std::printf("cannot read the golden in %s %s\n", dir.c_str(), err.c_str()), 1;
    const rdnut::Tensor& in = golden.at("input_nhwc");
    const bool           compare = sizeW == 0;
    const uint32_t       W = compare ? in.dims[1] : RoundUp(sizeW, 8), H = compare ? in.dims[0] : RoundUp(sizeH, 8);

    Gpu gpu;
    if (!gpu.Init())
        return std::printf("no D3D12 device\n"), 1;
    const bool wmma = !forceDp4a && EnableAmdWaveMatrixInt8(gpu.device);

    EngineDx12     engine;
    EngineDx12Desc desc;
    desc.device        = gpu.device;
    desc.manifest      = manifestBytes.data();
    desc.manifestBytes = manifestBytes.size();
    desc.weights       = blob.data();
    desc.weightBytes   = blob.size();
    desc.maxWidth      = W;
    desc.maxHeight     = H;
    desc.useWmma       = wmma;
    if (!engine.Create(desc, err))
        return std::printf("engine: %s\n", err.c_str()), 1;
    const Plan& plan = engine.GetPlan();
    Manifest    man;
    ParseManifest(manifestBytes.data(), manifestBytes.size(), man, err);
    std::printf("network %ux%u, %s, %.1f MB\n", W, H, wmma ? "DP4a + WMMA" : "DP4a", engine.MemoryBytes() / 1048576.0);

    ID3D12Resource* input  = gpu.Buffer(plan.InputBytes(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* upload = gpu.Buffer(plan.InputBytes(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource* kpn    = gpu.Buffer(plan.KpnBytes(), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12Resource* temporal = nullptr;
    {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC td{};
        td.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width            = W;
        td.Height           = H;
        td.DepthOrArraySize = 1;
        td.MipLevels        = 1;
        td.Format           = DXGI_FORMAT_R8G8B8A8_SNORM;
        td.SampleDesc.Count = 1;
        td.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        gpu.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                            IID_PPV_ARGS(&temporal));
    }
    ID3D12DescriptorHeap*      heap = nullptr;
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    gpu.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap));
    D3D12_UNORDERED_ACCESS_VIEW_DESC ud{};
    ud.Format        = DXGI_FORMAT_R8G8B8A8_SNORM;
    ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    gpu.device->CreateUnorderedAccessView(temporal, nullptr, &ud, heap->GetCPUDescriptorHandleForHeapStart());

    auto writeInput = [&](const rdnut::Tensor* t) {
        int8_t* p = nullptr;
        upload->Map(0, nullptr, reinterpret_cast<void**>(&p));
        const uint32_t pitch = plan.TensorOfKind(TensorKind::Input).pitchPixels;
        std::srand(1);
        for (size_t i = 0; i < plan.InputBytes(); ++i)
            p[i] = int8_t(std::rand() & 0xff);
        if (t)
            for (uint32_t y = 0; y < t->dims[0]; ++y)
                for (uint32_t x = 0; x < t->dims[1]; ++x)
                    for (uint32_t c = 0; c < 12; ++c)
                        p[(size_t(y) * pitch + x) * 12 + c] = int8_t(int(t->data[(size_t(y) * t->dims[1] + x) * 12 + c]));
        upload->Unmap(0, nullptr);
    };
    bool inputUav = false;
    auto uploadInput = [&]() {
        if (inputUav)
            gpu.Barrier(input, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        gpu.list->CopyBufferRegion(input, 0, upload, 0, plan.InputBytes());
        gpu.Barrier(input, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        inputUav = true;
    };
    auto record = [&](uint32_t w, uint32_t h) {
        gpu.list->SetDescriptorHeaps(1, &heap);
        return engine.Record(gpu.list, input->GetGPUVirtualAddress(), kpn->GetGPUVirtualAddress(),
                             heap->GetGPUDescriptorHandleForHeapStart(), w, h, err);
    };
    auto check = [&](const std::map<std::string, rdnut::Tensor>& g) {
        size_t              bad   = 0;
        std::vector<int8_t> arena = gpu.Read(engine.Arena(), plan.ArenaBytes());
        std::vector<int8_t> k     = gpu.Read(kpn, plan.KpnBytes());
        std::vector<int8_t> t     = gpu.ReadTexture(temporal, 4);
        for (size_t i = 0; i < man.tensors.size(); ++i)
        {
            const TensorLayout& L = plan.Tensors()[i];
            if (L.kind != TensorKind::Arena)
                continue;
            bad += Diff(man.tensors[i].name, g.at(std::string("chk.") + man.tensors[i].name), [&](uint32_t x, uint32_t y, uint32_t c) {
                return int(arena[L.offset + (size_t(y + 1) * L.pitchPixels + x + 1) * L.channels + c]);
            });
        }
        const TensorLayout& KL = plan.TensorOfKind(TensorKind::Kpn);
        bad += Diff("kpn", g.at("golden.kpn"), [&](uint32_t x, uint32_t y, uint32_t c) { return int(k[(size_t(y) * KL.pitchPixels + x) * 36 + c]); });
        bad += Diff("temporal", g.at("golden.temporal"), [&](uint32_t x, uint32_t y, uint32_t c) { return int(t[(size_t(y) * W + x) * 4 + c]); });
        return bad;
    };

    size_t bad = 0;
    writeInput(compare ? &in : nullptr);
    uploadInput();
    if (!record(W, H) || !gpu.Submit())
        return std::printf("record: %s\n", err.c_str()), 1;
    if (compare)
    {
        std::printf("frame %ux%u\n", W, H);
        bad += check(golden);
        if (drs)
        {
            const rdnut::Tensor& in2 = goldenDrs.at("input_nhwc");
            writeInput(&in2);
            uploadInput();
            if (!record(in2.dims[1], in2.dims[0]) || !gpu.Submit())
                return std::printf("record: %s\n", err.c_str()), 1;
            std::printf("frame %ux%u (same arena)\n", in2.dims[1], in2.dims[0]);
            bad += check(goldenDrs);
        }
    }

    if (timeRuns > 0)
    {
        ID3D12QueryHeap*      qh = nullptr;
        D3D12_QUERY_HEAP_DESC qd{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2, 0};
        gpu.device->CreateQueryHeap(&qd, IID_PPV_ARGS(&qh));
        ID3D12Resource* ts = gpu.Buffer(16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        gpu.list->EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        for (int i = 0; i < timeRuns; ++i)
            record(W, H);
        gpu.list->EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, 1);
        gpu.list->ResolveQueryData(qh, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, ts, 0);
        gpu.Submit();
        uint64_t* q = nullptr;
        ts->Map(0, nullptr, reinterpret_cast<void**>(&q));
        UINT64 freq = 1;
        gpu.queue->GetTimestampFrequency(&freq);
        std::printf("network %ux%u: %.3f ms per run over %d runs\n", W, H, double(q[1] - q[0]) * 1000.0 / double(freq) / timeRuns, timeRuns);
        ts->Unmap(0, nullptr);
        ts->Release();
        qh->Release();
    }

    engine.Destroy();
    input->Release(), upload->Release(), kpn->Release(), temporal->Release(), heap->Release();
    std::printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 2 : 0;
}
