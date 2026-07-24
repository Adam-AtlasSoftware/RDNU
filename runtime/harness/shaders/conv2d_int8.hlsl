// conv2d_int8.hlsl - correctness-first W8A8 convolution (the RDNU shipping precision).
//
// Per-layer scheme (verified against the exported INT8 model): the FP32 input activation is
// quantized on the fly with the layer's scalar activation scale, an INT8xINT8 -> INT32 conv is
// accumulated, then dequantized:
//     xq[c]      = clamp(round(x[c] / a_scale), -127, 127)
//     acc[oc]    = sum_over(ic,ky,kx) xq * w_int8[oc,ky,kx,ic]           (INT32)
//     out[oc]    = acc[oc] * w_scale[oc] * a_scale + bias[oc]            (FP32)
//
// Weights are OHWI (Cout, KH, KW, Cin/groups), stored int8-as-float here (exact; the WMMA path
// will pack them). w_scale/bias are per-output-channel FP32; a_scale is a scalar FP32.
// This is the plain reference the WMMA INT8 kernel will be validated against.

StructuredBuffer<float>   Input  : register(t0);   // fp32 activation [Cin * H * W]
StructuredBuffer<float>   Wi8    : register(t1);   // int8-as-fp32 weights [Cout * KH * KW * (Cin/groups)]
StructuredBuffer<float>   WScale : register(t2);   // per-oc fp32 [Cout]
StructuredBuffer<float>   Bias   : register(t3);   // per-oc fp32 [Cout]
StructuredBuffer<float>   AScale : register(t4);   // scalar fp32 [1]
RWStructuredBuffer<float>  Output : register(u0);   // fp32 [Cout * OH * OW]

cbuffer Dims : register(b0)
{
    uint Cin; uint H; uint W; uint Cout;
    uint KH; uint KW; uint PadH; uint PadW;
    uint StrideH; uint StrideW; uint Groups; uint OH; uint OW;
};

int quant(float v, float invA)
{
    // round-half-up (floor(x+0.5)); the numpy reference uses the identical formula so GPU == ref.
    return clamp(int(floor(v * invA + 0.5f)), -127, 127);
}

[numthreads(8, 8, 1)]
void conv2d_int8_CS(uint3 tid : SV_DispatchThreadID)
{
    uint ox = tid.x, oy = tid.y, oc = tid.z;
    if (ox >= OW || oy >= OH || oc >= Cout)
        return;

    float aScale = AScale[0];
    float invA   = 1.0f / aScale;

    uint inPerGroup  = Cin / Groups;
    uint outPerGroup = Cout / Groups;
    uint icBase      = (oc / outPerGroup) * inPerGroup;

    int acc = 0;
    for (uint icl = 0; icl < inPerGroup; ++icl)
    {
        uint ic = icBase + icl;
        for (uint ky = 0; ky < KH; ++ky)
        {
            int iy = int(oy * StrideH + ky) - int(PadH);
            if (iy < 0 || iy >= int(H)) continue;
            for (uint kx = 0; kx < KW; ++kx)
            {
                int ix = int(ox * StrideW + kx) - int(PadW);
                if (ix < 0 || ix >= int(W)) continue;
                int xq = quant(Input[(ic * H + uint(iy)) * W + uint(ix)], invA);
                int wq = int(Wi8[((oc * KH + ky) * KW + kx) * inPerGroup + icl]);
                acc += xq * wq;
            }
        }
    }
    Output[(oc * OH + oy) * OW + ox] = float(acc) * WScale[oc] * aScale + Bias[oc];
}
