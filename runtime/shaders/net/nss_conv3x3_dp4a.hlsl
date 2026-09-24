// nss_conv3x3_dp4a.hlsl - INT8 NHWC 3x3 conv with fused requantisation, SM 6.4 dot4add_i8packed.
// Runs every NSS layer on RDNA2 and any SM 6.4 GPU, and the stride-2, upsample and head layers
// on RDNA3.
//
// Permutation (compile-time): CIN (multiple of 4), OCB (output channels per pass, multiple of 4),
// STRIDE (1|2), UPSAMPLE (0|1: nearest 2x folded into the tap addressing), IN_BORDERED (0: the
// pre-process tensor, bounds-checked; 1: arena tensor with a -128 border), OUT_KIND, ACT.
//
// A group computes an 8x8 output tile. The input footprint is staged once in LDS; each thread
// owns one output pixel and loops over OCB-wide output-channel blocks. Weights are laid out
// [block][tap][cin/4][OCB] so each (tap, channel group) is one wave-uniform 16..64-byte load
// that the compiler issues on the scalar unit: activations come from LDS, weights from SGPRs,
// and every multiply is a v_dot4.

#ifndef CIN
#define CIN 32
#endif
#ifndef OCB
#define OCB 16
#endif
#ifndef STRIDE
#define STRIDE 1
#endif
#ifndef UPSAMPLE
#define UPSAMPLE 0
#endif
#ifndef IN_BORDERED
#define IN_BORDERED 1
#endif
#ifndef OUT_KIND
#define OUT_KIND 0
#endif
#ifndef ACT
#define ACT 0
#endif

#include "nss_net_common.hlsli"

#define TILE 8
#define CW (CIN / 4)
#if UPSAMPLE
#define TW (TILE / 2 + 2)
#elif STRIDE == 2
#define TW (2 * TILE + 1)
#else
#define TW (TILE + 2)
#endif
#define LP (CW | 1)                  // LDS words per pixel, odd: neighbouring pixels hit different banks
#define TILE_WORDS (TW * TW * CW)

groupshared uint gTile[TW * TW * LP];
#if ACT == ACT_SIGMOID
groupshared int gLut[256];
#endif

void LoadTile(uint gi, int2 origin)
{
    for (uint i = gi; i < TILE_WORDS; i += TILE * TILE)
    {
        uint pix = i / CW;
        uint g   = i - pix * CW;
        uint ty  = pix / TW;
        uint tx  = pix - ty * TW;
        int  sx  = origin.x + int(tx);
        int  sy  = origin.y + int(ty);
        uint w   = QZERO_WORD;
#if IN_BORDERED
        w = Input.Load(uint(InAddr(sx, sy)) + g * 4);
#else
        if (sx >= 0 && sy >= 0 && sx < int(InW) && sy < int(InH))
            w = Input.Load(uint(InAddr(sx, sy)) + g * 4);
#endif
        gTile[(ty * TW + tx) * LP + g] = w;
    }
}

[numthreads(TILE, TILE, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint gi : SV_GroupIndex)
{
    int2 o0 = int2(gid.xy) * TILE;
    int2 lo = int2(gtid.xy);
#if UPSAMPLE
    int2 origin = (o0 >> 1) - 1;
#elif STRIDE == 2
    int2 origin = o0 * 2 - 1;
#else
    int2 origin = o0 - 1;
#endif
    LoadTile(gi, origin);
#if ACT == ACT_SIGMOID
    for (uint i = gi; i < 256; i += TILE * TILE)
    {
        uint w = Weights.Load(LutOff + (i & ~3u));
        gLut[i] = int(w << (24 - 8 * (i & 3))) >> 24;
    }
#endif
    GroupMemoryBarrierWithGroupSync();

    int2 o = o0 + lo;
    if (o.x >= int(OutW) || o.y >= int(OutH))
        return;

    uint tb[9];
    [unroll] for (uint tap = 0; tap < 9; ++tap)
    {
        int dy = int(tap / 3);
        int dx = int(tap % 3);
#if UPSAMPLE
        // fine tap o + d - 1 reads source (o + d - 1) >> 1; the tile origin is (o0 >> 1) - 1
        int ty = ((lo.y + dy - 1) >> 1) + 1;
        int tx = ((lo.x + dx - 1) >> 1) + 1;
#else
        int ty = lo.y * STRIDE + dy;
        int tx = lo.x * STRIDE + dx;
#endif
        tb[tap] = uint(ty * TW + tx) * LP;
    }

    uint outAddr = uint(OutAddr(o.x, o.y));
    for (uint oc0 = 0; oc0 < Cout; oc0 += OCB)
    {
        int acc[OCB];
        [unroll] for (uint j4 = 0; j4 < OCB / 4; ++j4)
        {
            int4 c = asint(Weights.Load4(BiasOff + (oc0 + j4 * 4) * 4));
            acc[j4 * 4 + 0] = c.x;
            acc[j4 * 4 + 1] = c.y;
            acc[j4 * 4 + 2] = c.z;
            acc[j4 * 4 + 3] = c.w;
        }

        uint wBlock = WOff + (oc0 / OCB) * (9 * CW * OCB * 4);
        [unroll] for (uint t = 0; t < 9; ++t)
        {
            [unroll] for (uint g = 0; g < CW; ++g)
            {
                uint x  = gTile[tb[t] + g];
                uint wo = wBlock + (t * CW + g) * OCB * 4;
                [unroll] for (uint k = 0; k < OCB / 4; ++k)
                {
                    uint4 w = Weights.Load4(wo + k * 16);
                    acc[k * 4 + 0] = dot4add_i8packed(x, w.x, acc[k * 4 + 0]);
                    acc[k * 4 + 1] = dot4add_i8packed(x, w.y, acc[k * 4 + 1]);
                    acc[k * 4 + 2] = dot4add_i8packed(x, w.z, acc[k * 4 + 2]);
                    acc[k * 4 + 3] = dot4add_i8packed(x, w.w, acc[k * 4 + 3]);
                }
            }
        }

        uint packed[OCB / 4];
        [unroll] for (uint q4 = 0; q4 < OCB / 4; ++q4)
        {
            float4 r = asfloat(Weights.Load4(ScaleOff + (oc0 + q4 * 4) * 4));
            int4 q;
#if ACT == ACT_RELU
            q.x = RequantRelu(acc[q4 * 4 + 0], r.x);
            q.y = RequantRelu(acc[q4 * 4 + 1], r.y);
            q.z = RequantRelu(acc[q4 * 4 + 2], r.z);
            q.w = RequantRelu(acc[q4 * 4 + 3], r.w);
#else
            q.x = gLut[RequantLogit(acc[q4 * 4 + 0], r.x) + 128];
            q.y = gLut[RequantLogit(acc[q4 * 4 + 1], r.y) + 128];
            q.z = gLut[RequantLogit(acc[q4 * 4 + 2], r.z) + 128];
            q.w = gLut[RequantLogit(acc[q4 * 4 + 3], r.w) + 128];
#endif
#if OUT_KIND == OUT_TEMPORAL
            TemporalOut[uint2(o)] = float4(q) * (1.0f / 127.0f);
#endif
            packed[q4] = PackQ4(q);
        }

#if OUT_KIND == OUT_INT8
#if OCB == 16
        Output.Store4(outAddr + oc0, uint4(packed[0], packed[1], packed[2], packed[3]));
#elif OCB == 12
        Output.Store3(outAddr + oc0, uint3(packed[0], packed[1], packed[2]));
#elif OCB == 8
        Output.Store2(outAddr + oc0, uint2(packed[0], packed[1]));
#else
        [unroll] for (uint s = 0; s < OCB / 4; ++s)
            Output.Store(outAddr + oc0 + s * 4, packed[s]);
#endif
        StoreRingWords(o.x, o.y, oc0, OCB / 4);
#endif
    }
}
