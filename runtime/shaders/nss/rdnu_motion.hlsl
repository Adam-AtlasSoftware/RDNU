// rdnu_motion.hlsl - FSR 3.1 motion vectors NSS cannot read directly, as render pixels.
// Display resolution vectors are sampled where FSR 3 samples them (ComputeHrPosFromLrPos), and
// jitter carried in the vectors is cancelled as FSR 3 does (MotionVectorJitterCancellation).
cbuffer MotionConstants : register(b0)
{
    int2   RenderSize;
    int2   SourceSize;  // size the vectors are defined at
    float2 Scale;       // motionVectorScale
    float2 Cancel;      // previous minus current jitter when the vectors carry it
    float2 Jitter;
    uint   Display;     // defined at the upscaled size
    uint   Pad;
};

Texture2D<float2>   Source : register(t0);
RWTexture2D<float2> Output : register(u0);

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(RenderSize)))
        return;
    int2 p = int2(id.xy);
    if (Display)
        p = clamp(int2(floor((float2(p) + 0.5 - Jitter) / float2(RenderSize) * float2(SourceSize))), 0, SourceSize - 1);
    Output[id.xy] = (Source.Load(int3(p, 0)) * Scale - Cancel) * float2(RenderSize) / float2(SourceSize);
}
