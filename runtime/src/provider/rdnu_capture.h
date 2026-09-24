// rdnu_capture.h - per-dispatch dumps of the upscaler's inputs and outputs, for
// runtime/tools/eval/compare.py. Enabled by RDNU_CAPTURE=<directory>; RDNU_CAPTURE_FRAMES caps
// the frames per context (default 300). Each frame becomes <dir>/c<context>_f<frame>.rdnut
// with float tensors (H, W, C) and the dispatch parameters.
#pragma once

#include <d3d12.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rdnu
{

struct CaptureImage
{
    const char*           name     = nullptr;  // color, depth, motion, exposure, output, reference
    ID3D12Resource*       resource = nullptr;
    D3D12_RESOURCE_STATES state    = D3D12_RESOURCE_STATE_COMMON;  // state at the copy
};

struct CaptureParams
{
    float    jitter[2]   = {};  // as the host passed it (FSR's convention)
    float    mvScale[2]  = {};
    float    camera[3]   = {};  // near, far, vertical field of view
    float    preExposure = 1.0f;
    float    sharpness   = 0.0f;
    uint32_t render[2]   = {};
    uint32_t upscale[2]  = {};
    uint32_t reset       = 0;
    uint32_t flags       = 0;  // FfxApiCreateContextUpscaleFlags
};

class Capture
{
public:
    // A capture for one context when RDNU_CAPTURE is set, else null.
    static std::unique_ptr<Capture> FromEnvironment(ID3D12Device* device);
    ~Capture();

    // Copies the images to readback memory on cl, after the dispatch recorded on it.
    void Record(ID3D12GraphicsCommandList* cl, uint64_t frame, const CaptureImage* images, size_t count, const CaptureParams& params);
    // Writes the frames the GPU has finished (completed dispatches); all of them when idle.
    void Write(uint64_t completed, bool idle = false);

private:
    struct Image
    {
        std::string                        name;
        ID3D12Resource*                    buffer = nullptr;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        DXGI_FORMAT                        format = DXGI_FORMAT_UNKNOWN;
        uint32_t                           width = 0, height = 0;
    };
    struct Frame
    {
        uint64_t           index = 0;
        CaptureParams      params;
        std::vector<Image> images;
    };

    ID3D12Device*      device_ = nullptr;
    std::string        dir_;
    uint32_t           context_ = 0;
    uint64_t           limit_   = 0;
    std::vector<Frame> pending_;
};

}  // namespace rdnu
