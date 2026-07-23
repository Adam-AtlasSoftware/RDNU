// rdg_ctr_block.hlsl
// RDG Cross-Temporal Rendering (CTR) Block

// Note: These registers should match your root signature layout
Texture2D<float2> MotionVectors : register(t1);
Texture2DArray<min16float4> TemporalHistory : register(t3); // Persistent UAV read
SamplerState LinearSampler : register(s0);

RWTexture2DArray<min16float4> CurrentHiddenState : register(u0);
RWTexture2DArray<min16float4> NewTemporalHistory : register(u1); // Persistent UAV write

#ifndef NUM_SLICES
#define NUM_SLICES 12 // Default to 48 channels (12 * 4) for State 0
#endif

// Skip Connections
Texture2DArray<min16float4> SkipConnections : register(t4);
cbuffer CTRConstants : register(b0) {
    min16float SkipAlpha;
};

// Performs the warping of the previous hidden state using current MVs
void ApplyCTR_Warp(int2 pixelCoord, float2 uv, out min16float4 warpedHistory[NUM_SLICES])
{
    float2 mv = MotionVectors.Load(int3(pixelCoord, 0)).xy;
    float2 warpedUV = uv + mv;
    for (int i = 0; i < NUM_SLICES; i++) {
        warpedHistory[i] = TemporalHistory.SampleLevel(LinearSampler, float3(warpedUV, i), 0);
    }
}

// Performs the Per-Pixel Cross-Attention
min16float4 ApplyCTR_CrossAttention(min16float4 qPixel, min16float4 kPixel, min16float4 vPixel)
{
    min16float attentionScore = dot(qPixel, kPixel);
    min16float scaledScore = attentionScore * (min16float)0.125f; // scaling factor
    min16float attentionWeight = (min16float)1.0f / ((min16float)1.0f + exp(-scaledScore));
    return attentionWeight * vPixel;
}

[numthreads(8, 8, 1)]
void RDG_CTR_Block_CS(uint3 DispatchThreadId : SV_DispatchThreadID)
{
    int2 pixelCoord = int2(DispatchThreadId.xy);
    // Approximate UV
    float2 uv = float2((float)pixelCoord.x / 1920.0f, (float)pixelCoord.y / 1080.0f);

    min16float4 warpedHistory[NUM_SLICES];
    ApplyCTR_Warp(pixelCoord, uv, warpedHistory);

    // Process and store
    for (int i = 0; i < NUM_SLICES; i++) {
        min16float4 skipVal = SkipConnections.Load(int4(pixelCoord, i, 0));
        min16float4 mergedVal = warpedHistory[i] + skipVal * SkipAlpha;

        CurrentHiddenState[int3(pixelCoord, i)] = mergedVal;
        NewTemporalHistory[int3(pixelCoord, i)] = mergedVal; // Save for next frame
    }
}
