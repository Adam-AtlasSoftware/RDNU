// pixelshuffle.hlsl - torch nn.PixelShuffle(r): (C*r*r, H, W) -> (C, H*r, W*r).
// out[c, h*r+i, w*r+j] = in[c*r*r + i*r + j, h, w]  (PyTorch channel ordering).
//
// cbuffer slots used: Cin, H, W = input; Cout, OH, OW = output; StrideH = r (upscale factor).

StructuredBuffer<float>   In  : register(t0);   // [Cin * H * W],  Cin = Cout * r * r
RWStructuredBuffer<float>  Out : register(u0);   // [Cout * OH * OW], OH = H*r, OW = W*r

cbuffer Dims : register(b0)
{
    uint Cin; uint H; uint W; uint Cout;
    uint KH; uint KW; uint PadH; uint PadW;
    uint StrideH; uint StrideW; uint Groups; uint OH; uint OW;
};

[numthreads(8, 8, 1)]
void pixelshuffle_CS(uint3 tid : SV_DispatchThreadID)
{
    uint ox = tid.x, oy = tid.y, oc = tid.z;
    if (ox >= OW || oy >= OH || oc >= Cout)
        return;

    uint r = StrideH;
    uint h = oy / r, i = oy % r;
    uint w = ox / r, j = ox % r;
    uint inC = oc * (r * r) + i * r + j;

    Out[(oc * OH + oy) * OW + ox] = In[(inC * H + h) * W + w];
}
