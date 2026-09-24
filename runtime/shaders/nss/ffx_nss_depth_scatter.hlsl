// Depth scatter: reprojects dilated depth into the previous frame (atomic min/max).
#define NSS_BIND_SRV_INPUT_DEPTH          1
#define NSS_BIND_SRV_INPUT_MOTION_VECTORS 2
#define NSS_BIND_UAV_DEPTH_TM1            3
#define NSS_BIND_CB_NSS                   4

#include "generated/ffx_nss_depth_scatter.h"

[numthreads(FFX_NSS_THREAD_GROUP_WIDTH, FFX_NSS_THREAD_GROUP_HEIGHT, FFX_NSS_THREAD_GROUP_DEPTH)]
void main(uint3 id : SV_DispatchThreadID)
{
    DepthScatter(int2(id.xy));
}
