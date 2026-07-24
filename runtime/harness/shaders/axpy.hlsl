// axpy.hlsl - channel-broadcast scaled add: Out = A + Alpha[c] * B, where Alpha is a per-channel
// (C,) vector. Used for the decoder skip-merge `x = x + alpha * enc_skip` (skip_alphas is (1,C,1,1)).
//
// cbuffer slots used: Cout = channels, OH, OW = spatial.

StructuredBuffer<float>   A     : register(t0);   // [C * OH * OW]  (upsampled x)
StructuredBuffer<float>   B     : register(t1);   // [C * OH * OW]  (encoder skip)
StructuredBuffer<float>   Alpha : register(t2);   // [C]            (per-channel scale)
RWStructuredBuffer<float>  Out   : register(u0);   // [C * OH * OW]

cbuffer Dims : register(b0)
{
    uint Cin; uint H; uint W; uint Cout;
    uint KH; uint KW; uint PadH; uint PadW;
    uint StrideH; uint StrideW; uint Groups; uint OH; uint OW;
};

[numthreads(8, 8, 1)]
void axpy_CS(uint3 tid : SV_DispatchThreadID)
{
    uint ox = tid.x, oy = tid.y, oc = tid.z;
    if (ox >= OW || oy >= OH || oc >= Cout)
        return;

    uint idx = (oc * OH + oy) * OW + ox;
    Out[idx] = A[idx] + Alpha[oc] * B[idx];
}
