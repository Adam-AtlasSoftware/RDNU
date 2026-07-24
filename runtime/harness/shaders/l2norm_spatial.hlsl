// l2norm_spatial.hlsl - per-channel L2 normalize over the spatial (H*W) dimension, in FP32.
// Matches F.normalize(x, p=2, dim=<spatial>, eps=1e-12): out = x / max(||x||_2, eps), where the
// norm is taken over H*W independently for each channel. Used for CTR's q1 and k normalization
// (both reduce to per-channel spatial normalize once the head reshape is accounted for).
//
// cbuffer slots used: Cout = channel count, H, W = spatial dims. One thread per channel.

StructuredBuffer<float>   In  : register(t0);   // [C * H * W]
RWStructuredBuffer<float>  Out : register(u0);   // [C * H * W]

cbuffer Dims : register(b0)
{
    uint Cin; uint H; uint W; uint Cout;
    uint KH; uint KW; uint PadH; uint PadW;
    uint StrideH; uint StrideW; uint Groups; uint OH; uint OW;
};

[numthreads(1, 1, 64)]
void l2norm_spatial_CS(uint3 tid : SV_DispatchThreadID)
{
    uint c = tid.z;
    if (c >= Cout)
        return;

    uint hw = H * W;
    uint base = c * hw;

    float sumsq = 0.0f;
    for (uint s = 0; s < hw; ++s)
    {
        float v = In[base + s];
        sumsq += v * v;
    }
    float denom = sqrt(sumsq);
    denom = max(denom, 1e-12f);

    float inv = 1.0f / denom;
    for (uint s2 = 0; s2 < hw; ++s2)
        Out[base + s2] = In[base + s2] * inv;
}
