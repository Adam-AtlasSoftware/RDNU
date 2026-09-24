// Per-phase KPN tap offsets for non-2x scale factors.
#define NSS_BIND_UAV_OFFSET_LUT 0
#define NSS_BIND_CB_NSS         11

#include "generated/ffx_nss_generate_offset_lut.h"

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    GenerateOffsetLut(int(id.x));
}
