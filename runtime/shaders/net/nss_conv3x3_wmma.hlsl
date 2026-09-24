// nss_conv3x3_wmma.hlsl - INT8 NHWC 3x3 conv on RDNA3 wave-matrix units (stride 1, arena in/out).
//
// Permutation: CIN (multiple of 16), COUT (16, 32 or 64).
//
// Per wave: one output row segment of 16 * NT_PX pixels and all COUT channels, as NT_OC x NT_PX
// 16x16 accumulators (8 in every configuration). C^T[oc][px] = sum over taps and 16-channel
// slices of W[oc][c] (row-major OHWI rows, stride 9 * CIN) times X^T[c][px] (the NHWC tensor
// read column-major with the pixel stride). The -128 border of arena tensors makes every tap
// address valid, so the K loop has no branches. Accumulators start from the int32 bias
// (stride-0 load), go through LDS column-major (pixel rows, channels contiguous) and are
// requantised by the same scalar epilogue as the DP4a kernel.

#ifndef CIN
#define CIN 32
#endif
#ifndef COUT
#define COUT 32
#endif
#define OUT_KIND 0
#define ACT 0

#include "nss_net_common.hlsli"
#include "rdnu_wmma.hlsli"

#define NT_OC (COUT / 16)
#define NT_PX (8 / NT_OC)
#define KT (CIN / 16)
#define WROW (9 * CIN)

groupshared uint gAcc[NT_OC * NT_PX * 256];

WM_WAVE_ATTR
[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    gWmLane = gi;
    const int y  = int(gid.y);
    const int x0 = int(gid.x) * 16 * NT_PX;
    // runs that start beyond the 16-aligned row end are skipped (wave-uniform)
    const int np = min(NT_PX, (int(OutW + 15) / 16 * 16 - x0) / 16);

    WmAcc acc[NT_OC][NT_PX];
    [unroll] for (uint o = 0; o < NT_OC; ++o)
    {
        WmLoadBias(acc[o][0], Weights, BiasOff + o * 64);
        [unroll] for (uint p = 1; p < NT_PX; ++p)
            acc[o][p] = acc[o][0];
    }

    [unroll] for (uint tap = 0; tap < 9; ++tap)
    {
        const int dy = int(tap / 3) - 1;
        const int dx = int(tap % 3) - 1;
        [unroll] for (uint kt = 0; kt < KT; ++kt)
        {
            WmB b[NT_PX];
            [unroll] for (uint p = 0; p < NT_PX; ++p)
                if (int(p) < np)
                    WmLoadB(b[p], Input, uint(InAddr(x0 + int(p) * 16 + dx, y + dy)) + kt * 16, InPix);
            [unroll] for (uint o = 0; o < NT_OC; ++o)
            {
                WmA a;
                WmLoadA(a, Weights, WOff + o * 16 * WROW + tap * CIN + kt * 16, WROW);
                [unroll] for (uint p = 0; p < NT_PX; ++p)
                    if (int(p) < np)
                        acc[o][p] = WmMad(a, b[p], acc[o][p]);
            }
        }
    }

    [unroll] for (uint o = 0; o < NT_OC; ++o)
        [unroll] for (uint p = 0; p < NT_PX; ++p)
            if (int(p) < np)
                WmStoreLds(acc[o][p], gAcc, (o * NT_PX + p) * 256);
    GroupMemoryBarrierWithGroupSync();

    // lane -> pixel n of a run and 8 of its 16 channels
    const uint n    = gi & 15;
    const uint half = gi >> 4;
    [unroll] for (uint p = 0; p < NT_PX; ++p)
    {
        const int x = x0 + int(p) * 16 + int(n);
        if (int(p) >= np || x >= int(OutW) || y >= int(OutH))
            continue;
        const uint outAddr = uint(OutAddr(x, y));
        [unroll] for (uint o = 0; o < NT_OC; ++o)
        {
            const uint src = (o * NT_PX + p) * 256 + n * 16 + half * 8;
            const uint oc  = o * 16 + half * 8;
            uint4 a0 = uint4(gAcc[src + 0], gAcc[src + 1], gAcc[src + 2], gAcc[src + 3]);
            uint4 a1 = uint4(gAcc[src + 4], gAcc[src + 5], gAcc[src + 6], gAcc[src + 7]);
            float4 r0 = asfloat(Weights.Load4(ScaleOff + oc * 4));
            float4 r1 = asfloat(Weights.Load4(ScaleOff + oc * 4 + 16));
            int4 q0 = int4(RequantRelu(int(a0.x), r0.x), RequantRelu(int(a0.y), r0.y), RequantRelu(int(a0.z), r0.z), RequantRelu(int(a0.w), r0.w));
            int4 q1 = int4(RequantRelu(int(a1.x), r1.x), RequantRelu(int(a1.y), r1.y), RequantRelu(int(a1.z), r1.z), RequantRelu(int(a1.w), r1.w));
            Output.Store2(outAddr + oc, uint2(PackQ4(q0), PackQ4(q1)));
            StoreRingWords(x, y, oc, 2);
        }
    }
}
