// Post-process: KPN filter, history rectification and temporal accumulation.
#define NSS_BIND_SRV_INPUT_COLOR_JITTERED   0
#define NSS_BIND_SRV_INPUT_MOTION_VECTORS   1
#define NSS_BIND_SRV_HISTORY_UPSCALED_COLOR 2
#define NSS_BIND_KPN_TENSOR                 3
#define NSS_BIND_SRV_FEEDBACK_TENSOR        4
#define NSS_BIND_SRV_NEAREST_DEPTH_COORD    5
#define NSS_BIND_SRV_OFFSET_LUT             6
#define NSS_BIND_UAV_UPSCALED_OUTPUT        7
#define NSS_BIND_UAV_HISTORY_UPSCALED_COLOR 8
#define NSS_BIND_CB_NSS                     9
#define NSS_POSTPROCESS                     1

#include "generated/ffx_nss_postprocess.h"

[numthreads(FFX_NSS_THREAD_GROUP_WIDTH, FFX_NSS_THREAD_GROUP_HEIGHT, FFX_NSS_THREAD_GROUP_DEPTH)]
void main(uint3 id : SV_DispatchThreadID)
{
    Postprocess(int2(id.xy));
}
