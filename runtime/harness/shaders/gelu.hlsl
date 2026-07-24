// gelu.hlsl - exact (erf-based) GELU, elementwise. Matches PyTorch nn.GELU() default
// (approximate='none'): gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2))).
//
// HLSL has no erf intrinsic, so erf is the Abramowitz & Stegun 7.1.26 rational
// approximation (max abs error ~1.5e-7 — far inside the 1e-3 validation tolerance).
// Pointwise op: no weight/bias buffers; input and output share the same NCHW layout.

StructuredBuffer<float>   Input  : register(t0);   // [Cout * OH * OW]
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
    uint OH;
    uint OW;
};

float erf_approx(float x)
{
    float s  = sign(x);
    float ax = abs(x);
    float t  = 1.0f / (1.0f + 0.3275911f * ax);
    // Horner form of (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5).
    float poly = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t
                   - 0.284496736f) * t + 0.254829592f) * t;
    return s * (1.0f - poly * exp(-ax * ax));
}

[numthreads(8, 8, 1)]
void gelu_CS(uint3 tid : SV_DispatchThreadID)
{
    uint ox = tid.x;
    uint oy = tid.y;
    uint oc = tid.z;
    if (ox >= OW || oy >= OH || oc >= Cout)
        return;

    uint idx = (oc * OH + oy) * OW + ox;
    float x  = Input[idx];
    Output[idx] = 0.5f * x * (1.0f + erf_approx(x * 0.70710678118654752440f));
}
