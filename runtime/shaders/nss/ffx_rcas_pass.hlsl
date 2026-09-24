// ffx_rcas_pass.hlsl - optional sharpening of the upscaled output.
// Robust contrast-adaptive sharpening from AMD FidelityFX FSR 1 (FsrRcasF, MIT), with the
// denoise term on, applied to exposure-scaled colour as the FSR 3 upscaler does. Edge taps
// clamp to the image instead of reading zeros.
cbuffer RcasConstants : register(b0)
{
    uint2 Size;
    float Exposure;
    float Sharpness;  // exp2(-stops): 1 is the strongest
};

Texture2D<float4>   Input  : register(t0);
RWTexture2D<float4> Output : register(u0);

#define RCAS_LIMIT (0.25 - 1.0 / 16.0)

float3 Tap(int2 p)
{
    return Input.Load(int3(clamp(p, int2(0, 0), int2(Size) - 1), 0)).rgb * Exposure;
}

float Luma(float3 c)
{
    return c.b * 0.5 + (c.r * 0.5 + c.g);
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const int2 p = int2(id.xy);
    if (any(id.xy >= Size))
        return;

    const float3 b = Tap(p + int2(0, -1));
    const float3 d = Tap(p + int2(-1, 0));
    const float3 e = Tap(p);
    const float3 f = Tap(p + int2(1, 0));
    const float3 h = Tap(p + int2(0, 1));

    const float bL = Luma(b), dL = Luma(d), eL = Luma(e), fL = Luma(f), hL = Luma(h);
    float nz = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;
    nz = saturate(abs(nz) / (max(max(max(bL, dL), max(eL, fL)), hL) - min(min(min(bL, dL), min(eL, fL)), hL)));
    nz = -0.5 * nz + 1.0;

    const float3 mn4 = min(min(b, d), min(f, h));
    const float3 mx4 = max(max(b, d), max(f, h));
    const float  lowerLimiter = saturate(eL / min(min(bL, dL), min(fL, hL)));
    const float3 hitMin = mn4 / (4.0 * mx4) * lowerLimiter;
    const float3 hitMax = (1.0 - mx4) / (4.0 * mn4 - 4.0);
    const float3 lobe3  = max(-hitMin, hitMax);
    float lobe = max(-RCAS_LIMIT, min(max(max(lobe3.r, lobe3.g), lobe3.b), 0.0)) * Sharpness;
    lobe *= nz;

    const float3 c = (lobe * (b + d + h + f) + e) / (4.0 * lobe + 1.0);
    Output[p] = float4(c / Exposure, 1.0);
}
