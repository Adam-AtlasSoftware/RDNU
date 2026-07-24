// first_conv.hlsl - RDG first_conv: Conv2d(Cin=3 -> Cout=36, 3x3, stride 1, pad 1) + bias.
// Correctness-first reference: plain FP32, one thread per output element. No WMMA yet.
// Buffers are NCHW, row-major, float32. Weight is OIHW (out, in, kh, kw).

StructuredBuffer<float>   Input  : register(t0);   // [Cin * H * W]
StructuredBuffer<float>   Weight : register(t1);   // [Cout * Cin * 3 * 3]
StructuredBuffer<float>   Bias   : register(t2);   // [Cout]
RWStructuredBuffer<float>  Output : register(u0);   // [Cout * H * W]

cbuffer Dims : register(b0)
{
    uint Cin;
    uint H;
    uint W;
    uint Cout;
};

[numthreads(8, 8, 1)]
void first_conv_CS(uint3 tid : SV_DispatchThreadID)
{
    uint x  = tid.x;
    uint y  = tid.y;
    uint oc = tid.z;
    if (x >= W || y >= H || oc >= Cout)
        return;

    float acc = Bias[oc];
    for (uint ic = 0; ic < Cin; ++ic)
    {
        for (int ky = 0; ky < 3; ++ky)
        {
            int iy = int(y) + ky - 1;               // pad = 1
            if (iy < 0 || iy >= int(H))
                continue;
            for (int kx = 0; kx < 3; ++kx)
            {
                int ix = int(x) + kx - 1;
                if (ix < 0 || ix >= int(W))
                    continue;
                float v = Input[(ic * H + uint(iy)) * W + uint(ix)];
                float w = Weight[((oc * Cin + ic) * 3 + uint(ky)) * 3 + uint(kx)];
                acc += v * w;
            }
        }
    }
    Output[(oc * H + y) * W + x] = acc;
}
