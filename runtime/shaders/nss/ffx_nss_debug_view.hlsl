// Debug view: 4x4 grid of the pass inputs, intermediates and network inputs.
#define NSS_BIND_SRV_INPUT_COLOR_JITTERED   0
#define NSS_BIND_SRV_INPUT_MOTION_VECTORS   1
#define NSS_BIND_SRV_HISTORY_UPSCALED_COLOR 2
#define NSS_BIND_SRV_INPUT_DEPTH            3
#define NSS_BIND_SRV_INPUT_DEPTH_TM1        4
#define NSS_BIND_SRV_LUMA_DERIV_TM1         5
#define NSS_BIND_SRV_NEAREST_DEPTH_COORD    6
#define NSS_BIND_SRV_FEEDBACK_TENSOR        7
#define NSS_BIND_SRV_DISOCCLUSION_MASK_LQ   8
#define NSS_BIND_SRV_LUMA_DERIV             9
#define NSS_BIND_PREPROCESS_INPUT_TENSOR    10
#define NSS_BIND_UAV_DEBUG_VIEWS            11
#define NSS_BIND_CB_NSS                     12

#include "generated/ffx_nss_debug_view.h"

[numthreads(FFX_NSS_THREAD_GROUP_WIDTH, FFX_NSS_THREAD_GROUP_HEIGHT, FFX_NSS_THREAD_GROUP_DEPTH)]
void main(uint3 id : SV_DispatchThreadID)
{
    DebugViewCS(int2(id.xy));
}
