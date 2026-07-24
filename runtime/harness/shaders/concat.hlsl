// concat.hlsl - channel-concatenate two NCHW tensors: Out = cat([A, B], dim=channels).
// A supplies channels [0, Cin); B supplies channels [Cin, Cout). Spatial dims are shared.
// Used for the DFM merge input: cat([x, c], dim=1). Cin = A's channel count.

StructuredBuffer<float>   A   : register(t0);   // [Cin * OH * OW]
StructuredBuffer<float>   B   : register(t1);   // [(Cout-Cin) * OH * OW]
RWStructuredBuffer<float>  Out : register(u0);   // [Cout * OH * OW]

cbuffer Dims : register(b0)
{
    uint Cin;       // channels taken from A (the concat split point)
    uint H;
    uint W;
    uint Cout;      // total output channels
    uint KH;
    uint KW;
    uint PadH;
    uint PadW;
    uint StrideH;
    uint StrideW;
    uint Groups;
    uint OH;
    uint OW;
};

[numthreads(8, 8, 1)]
void concat_CS(uint3 tid : SV_DispatchThreadID)
{
    uint ox = tid.x;
    uint oy = tid.y;
    uint oc = tid.z;
    if (ox >= OW || oy >= OH || oc >= Cout)
        return;

    uint o = (oc * OH + oy) * OW + ox;
    if (oc < Cin)
        Out[o] = A[(oc * OH + oy) * OW + ox];
    else
        Out[o] = B[((oc - Cin) * OH + oy) * OW + ox];
}
