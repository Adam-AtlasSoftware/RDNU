// wmma_conv1x1.hlsl - WMMA-accelerated INT8 (W8A8) 1x1 convolution = a pure GEMM.
//   C[Cout, N] = W[Cout, Cin] @ Xq[Cin, N],  N = OH*OW (each pixel a column).
// Three passes (separate entry points sharing the constants + buffers below):
//   quant_CS    : FP32 activation (Cin,H,W) -> INT8 (Cin_pad, N), packed 4 pixels/uint
//   gemm_CS     : WMMA 16x16x16 INT8->INT32 tiled GEMM  W(Cout_pad,Cin_pad) @ Xq -> C(Cout_pad,N)
//   dequant_CS  : INT32 C -> FP32 out[oc] = C*w_scale[oc]*a_scale + bias[oc]
// Dims are padded to multiples of 16 for the fixed WMMA tile; padding channels are zero (contribute
// 0). Qact/Cacc stay UAVs across passes (UAV barriers, no state transitions). Validated vs conv2d_int8.

#define AmdExtD3DShaderIntrinsics_EnableWaveMatrix
#include "AmdExtD3DShaderIntrinsicsMatrixOps.hlsl"    // declares the AGS mailbox UAV at u0/magic-space

cbuffer Dims : register(b0)
{
    uint Cin; uint N; uint Cin_pad; uint Cout_pad;
    uint Cout; uint OH; uint OW; uint AScaleBits;      // AScaleBits = asuint(a_scale)
};

StructuredBuffer<float>   Act     : register(t0);   // [Cin * H * W]   (H*W == N)   -- quant in
ByteAddressBuffer         Weights : register(t1);   // [Cout_pad * Cin_pad] int8    -- gemm A
StructuredBuffer<float>   WScale  : register(t2);   // [Cout]                       -- dequant
StructuredBuffer<float>   Bias    : register(t3);   // [Cout]                       -- dequant
RWByteAddressBuffer        QAct    : register(u1);   // [Cin_pad * N] int8
RWByteAddressBuffer        Cacc    : register(u2);   // [Cout_pad * N] int32
RWStructuredBuffer<float>  OutF    : register(u3);   // [Cout * OH * OW]

int q1(float v, float invA) { return clamp(int(floor(v * invA + 0.5f)), -127, 127); }

// ---- pass 1: quantize activation to INT8 (Cin_pad, N), packed 4 pixels per uint ----
[numthreads(64, 1, 1)]
void quant_CS(uint3 tid : SV_DispatchThreadID)
{
    uint n4 = tid.x, ic = tid.y;               // n4 = pixel/4
    if (n4 * 4 >= N || ic >= Cin_pad) return;
    float invA = 1.0f / asfloat(AScaleBits);
    uint packed = 0;
    [unroll] for (uint e = 0; e < 4; ++e)
    {
        uint n = n4 * 4 + e;
        int q = 0;
        if (ic < Cin && n < N) q = q1(Act[ic * N + n], invA);
        packed |= (uint(q) & 0xFF) << (e * 8);
    }
    QAct.Store(ic * N + n4 * 4, packed);
}

// ---- pass 2: WMMA tiled INT8 GEMM (one Wave32 per 16x16 output tile) ----
[WaveSize(32)]
[numthreads(32, 1, 1)]
void gemm_CS(uint3 gid : SV_GroupID)
{
    uint ocTile = gid.y, nTile = gid.x;
    AmdWaveMatrixAccumulator<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I32, 16, 16> acc;
    acc.Fill(0);
    for (uint kt = 0; kt < Cin_pad / 16; ++kt)
    {
        AmdWaveMatrixA<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8, 16, 16> a;
        AmdWaveMatrixB<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8, 16, 16> b;
        a.Load(Weights, (ocTile * 16) * Cin_pad + kt * 16, Cin_pad, false);   // 16 oc rows, stride Cin_pad
        b.Load(QAct,    (kt * 16) * N + nTile * 16,        N,       false);   // 16 ic rows, stride N
        acc = AmdWaveMatrixMultiply(a, b, acc);
    }
    acc.Store(Cacc, (ocTile * 16) * N * 4 + (nTile * 16) * 4, N * 4, false);  // int32, stride N*4 bytes
}

// ---- pass 3: dequantize INT32 -> FP32 ----
[numthreads(64, 1, 1)]
void dequant_CS(uint3 tid : SV_DispatchThreadID)
{
    uint n = tid.x, oc = tid.y;
    if (n >= N || oc >= Cout) return;
    int c = asint(Cacc.Load(((oc * N) + n) * 4));
    OutF[oc * N + n] = float(c) * WScale[oc] * asfloat(AScaleBits) + Bias[oc];
}
