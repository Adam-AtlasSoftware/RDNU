// Pre-process: builds the 12-channel int8 network input and the per-frame side outputs.
#define NSS_BIND_SRV_INPUT_COLOR_JITTERED   0
#define NSS_BIND_SRV_INPUT_DEPTH            1
#define NSS_BIND_SRV_INPUT_MOTION_VECTORS   2
#define NSS_BIND_SRV_HISTORY_UPSCALED_COLOR 3
#define NSS_BIND_SRV_FEEDBACK_TENSOR        4
#define NSS_BIND_SRV_INPUT_DEPTH_TM1        5
#define NSS_BIND_SRV_LUMA_DERIV_TM1         6
#define NSS_BIND_SRV_DISOCCLUSION_MASK_LQ   7
#define NSS_BIND_PREPROCESS_INPUT_TENSOR    8
#define NSS_BIND_UAV_LUMA_DERIV             9
#define NSS_BIND_UAV_NEAREST_DEPTH_COORD    10
#define NSS_BIND_UAV_DEPTH_TM1              11
#define NSS_BIND_CB_NSS                     12
#define NSS_PREPROCESS                      1

#include "generated/ffx_nss_preprocess.h"

[numthreads(FFX_NSS_THREAD_GROUP_WIDTH, FFX_NSS_THREAD_GROUP_HEIGHT, FFX_NSS_THREAD_GROUP_DEPTH)]
void main(uint3 id : SV_DispatchThreadID)
{
    Preprocess(int2(id.xy));
}
