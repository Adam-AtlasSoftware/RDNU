// relu.hlsl - elementwise max(x, 0). Pointwise op: same NCHW layout in and out.
// cbuffer slots used: Cout = channels, OH, OW = spatial dims (same as gelu.hlsl).

StructuredBuffer<float>   Input  : register(t0);   // [Cout * OH * OW]
RWStructuredBuffer<float>  Output : register(u0);   // [Cout * OH * OW]

cbuffer Dims : register(b0)
{
    uint Cin; uint H; uint W; uint Cout;
    uint KH; uint KW; uint PadH; uint PadW;
    uint StrideH; uint StrideW; uint Groups; uint OH; uint OW;
};

[numthreads(8, 8, 1)]
void relu_CS(uint3 tid : SV_DispatchThreadID)
{
    uint ox = tid.x, oy = tid.y, oc = tid.z;
    if (ox >= OW || oy >= OH || oc >= Cout)
        return;
    uint i = (oc * OH + oy) * OW + ox;
    Output[i] = max(Input[i], 0.0f);
}
