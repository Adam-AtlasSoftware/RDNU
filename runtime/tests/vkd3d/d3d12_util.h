// d3d12_util.h - device, upload and readback helpers for the D3D12 tests.
#pragma once

#include <d3d12.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

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
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        return SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))) &&
               SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) &&
               SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) &&
               SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&list))) &&
               SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    }

    void Submit()
    {
        list->Close();
        ID3D12CommandList* l[] = {list};
        queue->ExecuteCommandLists(1, l);
        queue->Signal(fence, ++value);
        fence->SetEventOnCompletion(value, nullptr);
        alloc->Reset();
        list->Reset(alloc, nullptr);
    }

    ID3D12Resource* Texture(uint32_t w, uint32_t h, DXGI_FORMAT f, bool uav, D3D12_RESOURCE_STATES state)
    {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC d{};
        d.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width            = w;
        d.Height           = h;
        d.DepthOrArraySize = 1;
        d.MipLevels        = 1;
        d.Format           = f;
        d.SampleDesc.Count = 1;
        d.Flags            = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        ID3D12Resource* r  = nullptr;
        device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r));
        return r;
    }

    ID3D12Resource* Buffer(uint64_t bytes, D3D12_HEAP_TYPE heap)
    {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = heap;
        D3D12_RESOURCE_DESC d{};
        d.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width            = bytes;
        d.Height           = 1;
        d.DepthOrArraySize = 1;
        d.MipLevels        = 1;
        d.SampleDesc.Count = 1;
        d.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ID3D12Resource* r  = nullptr;
        device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d,
                                        heap == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST,
                                        nullptr, IID_PPV_ARGS(&r));
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

    // Uploads tightly packed texels to a texture in `state`; the upload buffer is returned for release after Submit.
    ID3D12Resource* Upload(ID3D12Resource* t, const void* texels, uint32_t texelBytes, D3D12_RESOURCE_STATES state)
    {
        D3D12_RESOURCE_DESC                d = t->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT64                             total = 0;
        device->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, &total);
        ID3D12Resource* up = Buffer(total, D3D12_HEAP_TYPE_UPLOAD);
        uint8_t*        p  = nullptr;
        up->Map(0, nullptr, reinterpret_cast<void**>(&p));
        for (uint32_t y = 0; y < d.Height; ++y)
            std::memcpy(p + size_t(y) * fp.Footprint.RowPitch, static_cast<const uint8_t*>(texels) + size_t(y) * d.Width * texelBytes,
                        size_t(d.Width) * texelBytes);
        up->Unmap(0, nullptr);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource       = t;
        dst.Type            = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.pResource       = up;
        src.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = fp;
        Barrier(t, state, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        Barrier(t, D3D12_RESOURCE_STATE_COPY_DEST, state);
        return up;
    }

    std::vector<uint8_t> Read(ID3D12Resource* t, uint32_t texelBytes, D3D12_RESOURCE_STATES state)
    {
        D3D12_RESOURCE_DESC                d = t->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT64                             total = 0;
        device->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, &total);
        ID3D12Resource*             rb = Buffer(total, D3D12_HEAP_TYPE_READBACK);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource       = rb;
        dst.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = fp;
        src.pResource       = t;
        src.Type            = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        Barrier(t, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        Barrier(t, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
        Submit();
        std::vector<uint8_t> out(size_t(d.Width) * d.Height * texelBytes);
        uint8_t*             p = nullptr;
        rb->Map(0, nullptr, reinterpret_cast<void**>(&p));
        for (uint32_t y = 0; y < d.Height; ++y)
            std::memcpy(&out[size_t(y) * d.Width * texelBytes], p + size_t(y) * fp.Footprint.RowPitch, size_t(d.Width) * texelBytes);
        rb->Unmap(0, nullptr);
        rb->Release();
        return out;
    }
};

inline uint16_t ToHalf(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000;
    const int      e = int((x >> 23) & 0xff) - 112;
    uint32_t       m = x & 0x7fffff;
    if (e < -10)
        return uint16_t(s);
    if (e <= 0)
    {
        const uint32_t M = m | 0x800000, sh = uint32_t(14 - e), h = M >> sh, r = M & ((1u << sh) - 1), half = 1u << (sh - 1);
        return uint16_t(s | (h + (r > half || (r == half && (h & 1)))));
    }
    if (e >= 31)
        return uint16_t(s | 0x7c00);
    uint32_t h = (uint32_t(e) << 10) | (m >> 13), r = m & 0x1fff;
    return uint16_t(s | (h + (r > 0x1000 || (r == 0x1000 && (h & 1)))));
}

inline float FromHalf(uint16_t h)
{
    const uint32_t s = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t       x = e == 0 ? s : e == 31 ? s | 0x7f800000 | (m << 13) : s | ((e + 112) << 23) | (m << 13);
    if (e == 0 && m)
    {
        float f = std::ldexp(float(m), -24);
        return h & 0x8000 ? -f : f;
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}
}  // namespace
