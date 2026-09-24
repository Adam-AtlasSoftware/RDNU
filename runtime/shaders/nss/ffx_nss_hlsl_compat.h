// ffx_nss_hlsl_compat.h - the GLSL built-ins Arm's NSS pass headers use, in HLSL.
// Compile with -enable-16bit-types -HV 2021 (half is fp16, int8_t lives in int16_t registers).
#ifndef FFX_NSS_HLSL_COMPAT_H
#define FFX_NSS_HLSL_COMPAT_H

#define FFX_GPU  1
#define FFX_HLSL 1

#ifndef FFX_HALF
#define FFX_HALF 1
#endif
#ifndef QUANTIZED
#define QUANTIZED 1
#endif
#undef NSS_SUPPORT_TENSOR
#define NSS_SUPPORT_TENSOR 0

#define RDNU_REG_(c, n) register(c##n)
#define RDNU_REG_T(n)   RDNU_REG_(t, n)
#define RDNU_REG_U(n)   RDNU_REG_(u, n)
#define RDNU_REG_B(n)   RDNU_REG_(b, n)

#define vec2      float2
#define vec3      float3
#define vec4      float4
#define ivec2     int2
#define ivec3     int3
#define ivec4     int4
#define uvec2     uint2
#define uvec3     uint3
#define uvec4     uint4
#define f16mat4x4 half4x4
#define int8_t    int16_t
#define int8_t2   int16_t2
#define int8_t3   int16_t3
#define int8_t4   int16_t4
#define uint8_t   uint16_t

#define fract          frac
#define uintBitsToFloat asfloat
#define floatBitsToUint asuint

#define lessThan(a, b)         ((a) < (b))
#define lessThanEqual(a, b)    ((a) <= (b))
#define greaterThan(a, b)      ((a) > (b))
#define greaterThanEqual(a, b) ((a) >= (b))
#define equal(a, b)            ((a) == (b))
#define notEqual(a, b)         ((a) != (b))

// GLSL combined samplers: sampler2D(tex, smp) becomes the two arguments of the shims below.
#define sampler2D(t, s)  t, s
#define usampler2D(t, s) t, s

// GLSL mix is defined as x * (1 - a) + y * a; a in {0, 1} selects exactly.
template <typename T, typename A>
T mix(T x, T y, A a)
{
    return x * (1 - a) + y * a;
}

float4 textureLod(Texture2D<float4> t, SamplerState s, float2 uv, float lod)
{
    return t.SampleLevel(s, uv, lod);
}

float4 textureSample(Texture2D<float4> t, SamplerState s, float2 uv)
{
    return t.SampleLevel(s, uv, 0);
}

float4 texelFetch(Texture2D<float4> t, SamplerState s, int2 p, int lod)
{
    return t.Load(int3(p, lod));
}

uint4 texelFetch(Texture2D<uint4> t, SamplerState s, int2 p, int lod)
{
    return t.Load(int3(p, lod));
}

float4 textureGather(Texture2D<float4> t, SamplerState s, float2 uv, int comp)
{
    if (comp == 0)
        return t.GatherRed(s, uv);
    return t.GatherGreen(s, uv);
}

int2 textureSize(Texture2D<float4> t, int lod)
{
    uint w, h, n;
    t.GetDimensions(lod, w, h, n);
    return int2(w, h);
}

int2 textureSize(Texture2D<uint4> t, int lod)
{
    uint w, h, n;
    t.GetDimensions(lod, w, h, n);
    return int2(w, h);
}

void imageStore(RWTexture2D<float4> img, int2 p, float4 v)
{
    img[p] = v;
}

void imageStore(RWTexture2D<uint4> img, int2 p, uint4 v)
{
    img[p] = v;
}

void imageAtomicMin(RWTexture2D<uint> img, int2 p, uint v)
{
    InterlockedMin(img[p], v);
}

void imageAtomicMax(RWTexture2D<uint> img, int2 p, uint v)
{
    InterlockedMax(img[p], v);
}

int bitfieldExtract(int v, int offset, int bits)
{
    return (v << (32 - offset - bits)) >> (32 - bits);
}

int4 bitfieldExtract(int4 v, int offset, int bits)
{
    return (v << (32 - offset - bits)) >> (32 - bits);
}

// int8 tensors in byte address buffers, four channels per dword (NHWC).
uint PackI8x4(int16_t4 v)
{
    uint4 u = uint4(v) & 0xFFu;
    return u.x | (u.y << 8) | (u.z << 16) | (u.w << 24);
}

int16_t4 UnpackI8x4(uint w)
{
    return int16_t4(int4(uint4(w, w, w, w) << uint4(24, 16, 8, 0)) >> 24);
}

#endif  // FFX_NSS_HLSL_COMPAT_H
