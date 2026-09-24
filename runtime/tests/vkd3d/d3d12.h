// d3d12.h for Linux test builds: vkd3d-proton's native D3D12 (DXIL through dxil-spirv) on any
// Vulkan driver, Mesa lavapipe included. Needs -std=gnu++17 for its __uuidof emulation.
#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <vkd3d_windows.h>
#include <vkd3d_d3d12.h>

#ifndef IID_PPV_ARGS
#define IID_PPV_ARGS(pp) __uuidof(**(pp)), reinterpret_cast<void**>(pp)
#endif
