// conv2d.hlsl - general direct 2D convolution (correctness-first reference, FP32).
//
// One thread per output element. Handles arbitrary kernel size, stride, padding and
// grouping, so it covers the whole family of RDG convs with a single kernel:
//   * first_conv / ffn.fn.0  : 3x3, stride 1, pad 1, groups 1
//   * ffn.fn.2 / 1x1 mixers  : 1x1, stride 1, pad 0, groups 1
//   * downs.*                : 3x3, stride 2, pad 1, groups 1
//   * DFM depthwise / CycleFC shift : groups = Cin
//
// Buffers are NCHW, row-major, float32. Weight is OIHW: (Cout, Cin/groups, KH, KW).
// This is the plain reference the WMMA/INT8 kernels are validated against; no tiling.

StructuredBuffer<float>   Input  : register(t0);   // [Cin * H * W]
StructuredBuffer<float>   Weight : register(t1);   // [Cout * (Cin/groups) * KH * KW]
StructuredBuffer<float>   Bias   : register(t2);   // [Cout]
RWStructuredBuffer<float>  Output : register(u0);   // [Cout * OH * OW]

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
    uint OH;        // output height  (passed in so indexing matches the golden layout)
    uint OW;        // output width
};

[numthreads(8, 8, 1)]
void conv2d_CS(uint3 tid : SV_DispatchThreadID)
{
    uint ox = tid.x;
    uint oy = tid.y;
    uint oc = tid.z;
    if (ox >= OW || oy >= OH || oc >= Cout)
        return;

    // Grouped conv: output channel oc reads only its group's slice of input channels.
    uint inPerGroup  = Cin / Groups;
    uint outPerGroup = Cout / Groups;
    uint group       = oc / outPerGroup;
    uint icBase      = group * inPerGroup;

    float acc = Bias[oc];
    for (uint icl = 0; icl < inPerGroup; ++icl)
    {
        uint ic = icBase + icl;
        for (uint ky = 0; ky < KH; ++ky)
        {
            int iy = int(oy * StrideH + ky) - int(PadH);
            if (iy < 0 || iy >= int(H))
                continue;
            for (uint kx = 0; kx < KW; ++kx)
            {
                int ix = int(ox * StrideW + kx) - int(PadW);
                if (ix < 0 || ix >= int(W))
                    continue;
                float v = Input[(ic * H + uint(iy)) * W + uint(ix)];
                float w = Weight[((oc * inPerGroup + icl) * KH + ky) * KW + kx];
                acc += v * w;
            }
        }
    }
    Output[(oc * OH + oy) * OW + ox] = acc;
}
