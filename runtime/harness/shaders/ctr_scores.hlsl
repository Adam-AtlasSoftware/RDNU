// ctr_scores.hlsl - CTR channel-attention scores: attn[hd,i,j] = sum_s Q[hd,i,s] * K[hd,j,s].
// Q is normalized q1 (head*Cq channels), K is normalized k (head*Ck channels); both laid out
// (channel, H*W) with channel = head*C + idx. Contraction is over the spatial dim s in [0,hw).
// Output is the tiny per-head score matrix, laid out (head, Cq, Ck): idx = (hd*Cq+i)*Ck + j.
//
// cbuffer slots used: Cin = head, H = Cq, W = Ck, Cout = hw.

StructuredBuffer<float>   Q   : register(t0);   // [head*Cq * hw]  (normalized q1)
StructuredBuffer<float>   K   : register(t1);   // [head*Ck * hw]  (normalized k)
RWStructuredBuffer<float>  Out : register(u0);   // [head * Cq * Ck]

cbuffer Dims : register(b0)
{
    uint Cin; uint H; uint W; uint Cout;
    uint KH; uint KW; uint PadH; uint PadW;
    uint StrideH; uint StrideW; uint Groups; uint OH; uint OW;
};

[numthreads(8, 8, 1)]
void ctr_scores_CS(uint3 tid : SV_DispatchThreadID)
{
    uint head = Cin, Cq = H, Ck = W, hw = Cout;
    uint j = tid.x, i = tid.y, hd = tid.z;
    if (j >= Ck || i >= Cq || hd >= head)
        return;

    uint qbase = (hd * Cq + i) * hw;
    uint kbase = (hd * Ck + j) * hw;
    float acc = 0.0f;
    for (uint s = 0; s < hw; ++s)
        acc += Q[qbase + s] * K[kbase + s];

    Out[(hd * Cq + i) * Ck + j] = acc;
}
