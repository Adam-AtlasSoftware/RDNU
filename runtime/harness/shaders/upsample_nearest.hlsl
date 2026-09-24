// upsample_nearest.hlsl - nearest-neighbour 2x upsample (nn.UpsamplingNearest2d(scale_factor=2)).
// out[c, y, x] = in[c, y/2, x/2]. cbuffer slots used: Cout = channels, H, W = input, OH, OW = output.

StructuredBuffer<float>   In  : register(t0);   // [C * H * W]
RWStructuredBuffer<float>  Out : register(u0);   // [C * OH * OW]

cbuffer Dims : register(b0)
{
    uint Cin; uint H; uint W; uint Cout;
    uint KH; uint KW; uint PadH; uint PadW;
    uint StrideH; uint StrideW; uint Groups; uint OH; uint OW;
};

[numthreads(8, 8, 1)]
void upsample_nearest_CS(uint3 tid : SV_DispatchThreadID)
{
    uint ox = tid.x, oy = tid.y, c = tid.z;
    if (ox >= OW || oy >= OH || c >= Cout)
        return;
    Out[(c * OH + oy) * OW + ox] = In[(c * H + oy / 2) * W + ox / 2];
}
