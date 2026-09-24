// rdnu_amd_ext.h - AMD DX12 driver extensions for the wave-matrix network kernels.
#pragma once

#include <d3d12.h>

namespace rdnu
{

// Enables AMD shader intrinsics on an existing device (no AGS device creation needed, so it
// works inside a game's device) and reports whether INT8 16x16x16 wave-matrix multiply with
// INT32 accumulation is available (RDNA3 and later). False on any other driver or build.
bool EnableAmdWaveMatrixInt8(ID3D12Device* device);

}  // namespace rdnu
