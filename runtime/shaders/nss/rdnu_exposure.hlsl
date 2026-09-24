// rdnu_exposure.hlsl - the exposure NSS applies before tonemapping, with FSR 3.1 semantics.
//
// Default (prepare): writes a 1x1 exposure texture. Mode 0 takes the game's exposure texture
// times Scale (1 / preExposure). Mode 1 is FSR 3 auto exposure (ComputeAutoExposureFromLavg,
// AMD FidelityFX, MIT) from the log-average luminance of a 64x64 grid of the colour input.
// RDNU_EXPOSURE_PATCH: stores (e, 1/e) into an NSS constant buffer at byte offset Offset.
cbuffer ExposureConstants : register(b0)
{
    uint  Mode;  // patch: byte offset of NssConstants::_Exposure
    float Scale;
    uint  Width;
    uint  Height;
};

#if RDNU_EXPOSURE_PATCH

Texture2D<float>    ExposureTex : register(t0);
RWByteAddressBuffer Constants   : register(u0);

[numthreads(1, 1, 1)]
void main()
{
    float e = ExposureTex.Load(int3(0, 0, 0));
    e = e > 0.0 && e < 65504.0 ? e : 1.0;
    Constants.Store2(Mode, asuint(float2(e, 1.0 / e)));
}

#else

Texture2D<float4>  Source   : register(t0);
RWTexture2D<float> Exposure : register(u0);

groupshared float gSum[256];

[numthreads(256, 1, 1)]
void main(uint gi : SV_GroupIndex)
{
    if (Mode == 0)
    {
        if (gi == 0)
            Exposure[uint2(0, 0)] = Source.Load(int3(0, 0, 0)).r * Scale;
        return;
    }

    float sum = 0.0;
    for (uint i = gi; i < 64 * 64; i += 256)
    {
        const uint2  cell = uint2(i % 64, i / 64);
        const uint2  p    = (cell * uint2(Width, Height) + uint2(Width, Height) / 2) / 64;
        const float3 c    = min(Source.Load(int3(p, 0)).rgb, 65504.0);
        sum += log(max(dot(c, float3(0.2126, 0.7152, 0.0722)), 1e-6));
    }
    gSum[gi] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (gi < s)
            gSum[gi] += gSum[gi + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (gi == 0)
    {
        const float lavg  = exp(gSum[0] / 4096.0);
        const float ev100 = log2(lavg * 100.0 / 12.5);
        const float lmax  = (78.0 / (0.65 * 100.0)) * exp2(ev100);
        Exposure[uint2(0, 0)] = 1.0 / lmax;
    }
}

#endif
