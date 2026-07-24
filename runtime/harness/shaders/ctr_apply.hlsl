// ctr_apply.hlsl - apply CTR attention to values: out[hd,i,s] = sum_j Attn[hd,i,j] * V[hd,j,s].
// Attn is the softmaxed score matrix (head, Cq, Ck); V is (head*Ck channels, H*W). The output is
// (head*Cq channels, H*W) = the attended feature map, channel = hd*Cq + i, spatial s = y*W + x.
//
// cbuffer slots used: Cout = head*Cq (out channels), H, W = spatial, Groups = head, KH = Cq, KW = Ck.

StructuredBuffer<float>   Attn : register(t0);   // [head * Cq * Ck]
StructuredBuffer<float>   V    : register(t1);   // [head*Ck * H*W]
RWStructuredBuffer<float>  Out  : register(u0);   // [head*Cq * H*W]

cbuffer Dims : register(b0)
{
    uint Cin; uint H; uint W; uint Cout;
    uint KH; uint KW; uint PadH; uint PadW;
    uint StrideH; uint StrideW; uint Groups; uint OH; uint OW;
};

[numthreads(8, 8, 1)]
void ctr_apply_CS(uint3 tid : SV_DispatchThreadID)
{
    uint Cq = KH, Ck = KW;
    uint x = tid.x, y = tid.y, oc = tid.z;
    if (x >= W || y >= H || oc >= Cout)
        return;

    uint hd = oc / Cq;
    uint i  = oc % Cq;
    uint hw = H * W;
    uint s  = y * W + x;

    uint abase = (hd * Cq + i) * Ck;
    float acc = 0.0f;
    for (uint j = 0; j < Ck; ++j)
        acc += Attn[abase + j] * V[(hd * Ck + j) * hw + s];

    Out[oc * hw + s] = acc;
}
