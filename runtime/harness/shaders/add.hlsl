// add.hlsl - elementwise add of two equal-shape NCHW tensors: Out = A + B.
// Used for the residual connections (x = block(x) + x). Pointwise; no weights.

StructuredBuffer<float>   A   : register(t0);   // [Cout * OH * OW]
StructuredBuffer<float>   B   : register(t1);   // [Cout * OH * OW]
RWStructuredBuffer<float>  Out : register(u0);   // [Cout * OH * OW]

cbuffer Dims : register(b0)
{
    uint Cin;
    uint H;
    uint W;
    uint Cout;
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
void add_CS(uint3 tid : SV_DispatchThreadID)
{
    uint ox = tid.x;
    uint oy = tid.y;
    uint oc = tid.z;
    if (ox >= OW || oy >= OH || oc >= Cout)
        return;

    uint idx = (oc * OH + oy) * OW + ox;
    Out[idx] = A[idx] + B[idx];
}
