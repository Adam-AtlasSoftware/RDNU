// softmax_rows.hlsl - numerically-stable softmax over the last axis (Ck) of the CTR score
// matrix, laid out (head, Cq, Ck). Matches F.softmax(attn, dim=-1). One thread per (head, Cq) row.
//
// cbuffer slots used: Cin = head, H = Cq, W = Ck.

StructuredBuffer<float>   In  : register(t0);   // [head * Cq * Ck]
RWStructuredBuffer<float>  Out : register(u0);   // [head * Cq * Ck]

cbuffer Dims : register(b0)
{
    uint Cin; uint H; uint W; uint Cout;
    uint KH; uint KW; uint PadH; uint PadW;
    uint StrideH; uint StrideW; uint Groups; uint OH; uint OW;
};

[numthreads(8, 8, 1)]
void softmax_rows_CS(uint3 tid : SV_DispatchThreadID)
{
    uint head = Cin, Cq = H, Ck = W;
    uint i = tid.x, hd = tid.y;
    if (i >= Cq || hd >= head)
        return;

    uint base = (hd * Cq + i) * Ck;

    float m = -3.402823466e38f;
    for (uint j = 0; j < Ck; ++j)
        m = max(m, In[base + j]);

    float sum = 0.0f;
    for (uint j2 = 0; j2 < Ck; ++j2)
        sum += exp(In[base + j2] - m);

    float inv = 1.0f / sum;
    for (uint j3 = 0; j3 < Ck; ++j3)
        Out[base + j3] = exp(In[base + j3] - m) * inv;
}
