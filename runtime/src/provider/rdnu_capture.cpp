// rdnu_capture.cpp - see rdnu_capture.h
#include "rdnu_capture.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace rdnu
{
namespace
{
float Half(uint16_t h)
{
    const int e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float     v = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(float(m | 0x400), e - 25);
    return h & 0x8000 ? -v : v;
}

// the unsigned 11 and 10 bit floats of R11G11B10: 5 exponent bits, 6 or 5 mantissa bits
float Small(uint32_t v, int mantissa)
{
    const int e = int(v >> mantissa), m = int(v & ((1u << mantissa) - 1));
    if (e == 31)
        return m ? NAN : INFINITY;
    return e == 0 ? std::ldexp(float(m), -14 - mantissa) : std::ldexp(float(m | (1 << mantissa)), e - 15 - mantissa);
}

float Srgb(float c)
{
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

struct Layout
{
    uint32_t bytes    = 0;  // per texel; 0 for formats not read here
    uint32_t channels = 0;
};

Layout LayoutOf(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return {16, 4};
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM: return {8, 4};
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return {4, 4};
    case DXGI_FORMAT_R11G11B10_FLOAT: return {4, 3};
    case DXGI_FORMAT_R32G32_TYPELESS:
    case DXGI_FORMAT_R32G32_FLOAT: return {8, 2};
    case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R16G16_FLOAT: return {4, 2};
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: return {4, 1};
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_D16_UNORM: return {2, 1};
    default: return {};
    }
}

// One texel as float; typeless formats read as the float or unorm view games use.
void Texel(DXGI_FORMAT f, const uint8_t* p, float* out)
{
    uint16_t h[4];
    uint32_t w;
    switch (f)
    {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32A32_FLOAT: std::memcpy(out, p, 16); break;
    case DXGI_FORMAT_R32G32_TYPELESS:
    case DXGI_FORMAT_R32G32_FLOAT: std::memcpy(out, p, 8); break;
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_D32_FLOAT: std::memcpy(out, p, 4); break;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        std::memcpy(h, p, 8);
        for (int c = 0; c < 4; ++c)
            out[c] = Half(h[c]);
        break;
    case DXGI_FORMAT_R16G16B16A16_UNORM:
        std::memcpy(h, p, 8);
        for (int c = 0; c < 4; ++c)
            out[c] = h[c] / 65535.0f;
        break;
    case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R16G16_FLOAT:
        std::memcpy(h, p, 4);
        out[0] = Half(h[0]), out[1] = Half(h[1]);
        break;
    case DXGI_FORMAT_R16_FLOAT: std::memcpy(h, p, 2), out[0] = Half(h[0]); break;
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_D16_UNORM: std::memcpy(h, p, 2), out[0] = h[0] / 65535.0f; break;
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: std::memcpy(&w, p, 4), out[0] = (w & 0xffffff) / 16777215.0f; break;
    case DXGI_FORMAT_R11G11B10_FLOAT:
        std::memcpy(&w, p, 4);
        out[0] = Small(w & 0x7ff, 6), out[1] = Small((w >> 11) & 0x7ff, 6), out[2] = Small(w >> 22, 5);
        break;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
        std::memcpy(&w, p, 4);
        for (int c = 0; c < 3; ++c)
            out[c] = ((w >> (10 * c)) & 0x3ff) / 1023.0f;
        out[3] = (w >> 30) / 3.0f;
        break;
    default:
    {
        const bool bgra = f == DXGI_FORMAT_B8G8R8A8_TYPELESS || f == DXGI_FORMAT_B8G8R8A8_UNORM || f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        const bool srgb = f == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        for (int c = 0; c < 4; ++c)
            out[c] = p[bgra && c < 3 ? 2 - c : c] / 255.0f;
        if (srgb)
            for (int c = 0; c < 3; ++c)
                out[c] = Srgb(out[c]);
    }
    }
}

struct Tensor
{
    std::string           name;
    std::vector<uint32_t> dims;
    std::vector<float>    data;
};

// runtime/tests/common/rdnut.h
bool WriteRdnut(const std::string& path, const std::vector<Tensor>& tensors)
{
    std::ofstream f(path, std::ios::binary);
    auto          u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    f.write("RDNT", 4);
    u32(1);
    u32(uint32_t(tensors.size()));
    for (const Tensor& t : tensors)
    {
        u32(uint32_t(t.name.size()));
        f.write(t.name.data(), std::streamsize(t.name.size()));
        u32(uint32_t(t.dims.size()));
        for (uint32_t d : t.dims)
            u32(d);
        f.write(reinterpret_cast<const char*>(t.data.data()), std::streamsize(t.data.size() * 4));
    }
    return bool(f);
}
}  // namespace

std::unique_ptr<Capture> Capture::FromEnvironment(ID3D12Device* device)
{
    static std::atomic<uint32_t> contexts{0};
    const char*                  dir = std::getenv("RDNU_CAPTURE");
    if (!dir || !*dir)
        return nullptr;
    std::error_code e;
    std::filesystem::create_directories(dir, e);
    std::unique_ptr<Capture> c(new Capture());
    c->device_           = device;
    c->dir_              = dir;
    c->context_          = contexts++;
    const char* frames   = std::getenv("RDNU_CAPTURE_FRAMES");
    c->limit_            = frames ? std::strtoull(frames, nullptr, 10) : 300;
    return c;
}

Capture::~Capture()
{
    Write(0, true);
}

void Capture::Record(ID3D12GraphicsCommandList* cl, uint64_t frame, const CaptureImage* images, size_t count, const CaptureParams& params)
{
    if (frame >= limit_)
        return;
    Frame f;
    f.index  = frame;
    f.params = params;
    for (size_t i = 0; i < count; ++i)
    {
        const CaptureImage& in = images[i];
        if (!in.resource)
            continue;
        const D3D12_RESOURCE_DESC d = in.resource->GetDesc();
        Image                     im;
        UINT64                    bytes = 0;
        device_->GetCopyableFootprints(&d, 0, 1, 0, &im.footprint, nullptr, nullptr, &bytes);
        im.name   = in.name;
        im.format = im.footprint.Footprint.Format;
        im.width  = uint32_t(d.Width);
        im.height = d.Height;
        if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || !LayoutOf(im.format).bytes)
            continue;
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC b{};
        b.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        b.Width            = bytes;
        b.Height           = 1;
        b.DepthOrArraySize = 1;
        b.MipLevels        = 1;
        b.SampleDesc.Count = 1;
        b.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &b, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&im.buffer))))
            continue;
        D3D12_RESOURCE_BARRIER x{};
        x.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        x.Transition.pResource   = in.resource;
        x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        x.Transition.StateBefore = in.state;
        x.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
        if (in.state != D3D12_RESOURCE_STATE_COPY_SOURCE)
            cl->ResourceBarrier(1, &x);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource        = im.buffer;
        dst.Type             = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint  = im.footprint;
        src.pResource        = in.resource;
        src.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        std::swap(x.Transition.StateBefore, x.Transition.StateAfter);
        if (in.state != D3D12_RESOURCE_STATE_COPY_SOURCE)
            cl->ResourceBarrier(1, &x);
        f.images.push_back(im);
    }
    pending_.push_back(std::move(f));
}

void Capture::Write(uint64_t completed, bool idle)
{
    for (size_t k = 0; k < pending_.size();)
    {
        Frame& f = pending_[k];
        if (!idle && f.index >= completed)
        {
            ++k;
            continue;
        }
        const CaptureParams& p = f.params;
        std::vector<Tensor>  t = {
            {"frame", {1}, {float(f.index)}},
            {"jitter", {2}, {p.jitter[0], p.jitter[1]}},
            {"mv_scale", {2}, {p.mvScale[0], p.mvScale[1]}},
            {"camera", {3}, {p.camera[0], p.camera[1], p.camera[2]}},
            {"pre_exposure", {1}, {p.preExposure}},
            {"sharpness", {1}, {p.sharpness}},
            {"render", {2}, {float(p.render[0]), float(p.render[1])}},
            {"upscale", {2}, {float(p.upscale[0]), float(p.upscale[1])}},
            {"reset", {1}, {float(p.reset)}},
            {"flags", {1}, {float(p.flags)}},
        };
        for (Image& im : f.images)
        {
            const Layout   l = LayoutOf(im.format);
            const uint32_t c = l.channels;
            Tensor         x{im.name, {im.height, im.width, c}, std::vector<float>(size_t(im.width) * im.height * c)};
            uint8_t*       data = nullptr;
            if (SUCCEEDED(im.buffer->Map(0, nullptr, reinterpret_cast<void**>(&data))))
            {
                for (uint32_t y = 0; y < im.height; ++y)
                {
                    const uint8_t* row = data + im.footprint.Offset + size_t(y) * im.footprint.Footprint.RowPitch;
                    for (uint32_t i = 0; i < im.width; ++i)
                    {
                        float v[4];
                        Texel(im.format, row + size_t(i) * l.bytes, v);
                        std::memcpy(&x.data[(size_t(y) * im.width + i) * c], v, size_t(c) * 4);
                    }
                }
                im.buffer->Unmap(0, nullptr);
            }
            t.push_back(std::move(x));
            im.buffer->Release();
        }
        char name[64];
        std::snprintf(name, sizeof(name), "/c%02u_f%05llu.rdnut", context_, static_cast<unsigned long long>(f.index));
        WriteRdnut(dir_ + name, t);
        pending_.erase(pending_.begin() + long(k));
    }
}

}  // namespace rdnu
