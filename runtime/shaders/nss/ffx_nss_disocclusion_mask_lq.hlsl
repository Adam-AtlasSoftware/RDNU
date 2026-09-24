// Low-resolution disocclusion mask (balanced and performance modes).
#define NSS_BIND_SRV_INPUT_MOTION_VECTORS 0
#define NSS_BIND_SRV_INPUT_DEPTH          1
#define NSS_BIND_SRV_INPUT_DEPTH_TM1      2
#define NSS_BIND_UAV_DISOCCLUSION_MASK_LQ 3
#define NSS_BIND_CB_NSS                   4

#include "generated/ffx_nss_disocclusion_mask_lq.h"

[numthreads(FFX_NSS_THREAD_GROUP_WIDTH, FFX_NSS_THREAD_GROUP_HEIGHT, FFX_NSS_THREAD_GROUP_DEPTH)]
void main(uint3 id : SV_DispatchThreadID)
{
    DisocclusionMaskLq(int2(id.xy));
}
