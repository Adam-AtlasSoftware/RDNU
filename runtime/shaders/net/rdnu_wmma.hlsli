// rdnu_wmma.hlsli - the five wave-matrix operations the NSS conv kernel uses, 16x16x16 INT8 -> INT32.
//
// Default: AMD AGS wave-matrix intrinsics (RDNA3, SM 6.6, AMD driver), used the way FSR 4's INT8
// operators use them. With RDNU_WMMA_EMULATE=1 the same API is implemented in plain HLSL with
// every thread holding whole matrices, so the kernel's addressing and epilogue run on any
// device (Mesa lavapipe in the Linux tests). Semantics, element (row, col):
//   WmLoadA(a, buf, off, stride)   a(m, k) = int8 buf[off + m * stride + k]        (row major)
//   WmLoadB(b, buf, off, stride)   b(k, n) = int8 buf[off + n * stride + k]        (column major)
//   WmLoadBias(c, buf, off)        c(m, n) = int32 buf[off + m * 4]                (stride 0)
//   WmMad(a, b, c)                 c + a * b
//   WmStoreLds(c, lds, off)        lds[off + n * 16 + m] = c(m, n)                 (column major)
#ifndef RDNU_WMMA_HLSLI
#define RDNU_WMMA_HLSLI

#if RDNU_WMMA_EMULATE

struct WmA   { int v[256]; };
struct WmB   { int v[256]; };
struct WmAcc { int v[256]; };

static uint gWmLane;

int WmLoadS8(uint w, uint byteInWord) { return int(w << (24 - 8 * byteInWord)) >> 24; }

void WmLoadA(inout WmA a, ByteAddressBuffer buf, uint off, uint stride)
{
    [loop] for (uint m = 0; m < 16; ++m)
        [loop] for (uint k = 0; k < 16; ++k)
        {
            uint addr = off + m * stride + k;
            a.v[m * 16 + k] = WmLoadS8(buf.Load(addr & ~3u), addr & 3u);
        }
}

void WmLoadB(inout WmB b, RWByteAddressBuffer buf, uint off, uint stride)
{
    [loop] for (uint k = 0; k < 16; ++k)
        [loop] for (uint n = 0; n < 16; ++n)
        {
            uint addr = off + n * stride + k;
            b.v[k * 16 + n] = WmLoadS8(buf.Load(addr & ~3u), addr & 3u);
        }
}

void WmLoadBias(inout WmAcc c, ByteAddressBuffer buf, uint off)
{
    [loop] for (uint m = 0; m < 16; ++m)
    {
        int bias = asint(buf.Load(off + m * 4));
        [loop] for (uint n = 0; n < 16; ++n)
            c.v[m * 16 + n] = bias;
    }
}

WmAcc WmMad(WmA a, WmB b, WmAcc c)
{
    [loop] for (uint m = 0; m < 16; ++m)
        [loop] for (uint n = 0; n < 16; ++n)
        {
            int s = c.v[m * 16 + n];
            [loop] for (uint k = 0; k < 16; ++k)
                s += a.v[m * 16 + k] * b.v[k * 16 + n];
            c.v[m * 16 + n] = s;
        }
    return c;
}

#define WmStoreLds(acc, lds, off)                                    \
    do                                                               \
    {                                                                \
        if (gWmLane == 0)                                            \
            [loop] for (uint m_ = 0; m_ < 16; ++m_)                  \
                [loop] for (uint n_ = 0; n_ < 16; ++n_)              \
                    lds[(off) + n_ * 16 + m_] = uint(acc.v[m_ * 16 + n_]); \
    } while (0)

#define WM_WAVE_ATTR

#else  // AGS wave matrix

#define AmdExtD3DShaderIntrinsics_EnableWaveMatrix
#include "AmdExtD3DShaderIntrinsicsMatrixOps.hlsl"

typedef AmdWaveMatrixA<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8, 16, 16>            WmA;
typedef AmdWaveMatrixB<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8, 16, 16>            WmB;
typedef AmdWaveMatrixAccumulator<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I32, 16, 16> WmAcc;

static uint gWmLane;

void WmLoadA(inout WmA a, ByteAddressBuffer buf, uint off, uint stride) { a.Load(buf, off, stride, false); }
void WmLoadB(inout WmB b, RWByteAddressBuffer buf, uint off, uint stride) { b.Load(buf, off, stride, true); }
void WmLoadBias(inout WmAcc c, ByteAddressBuffer buf, uint off) { c.Load(buf, off, 0, true); }
WmAcc WmMad(WmA a, WmB b, WmAcc c) { return AmdWaveMatrixMultiply(a, b, c); }
#define WmStoreLds(acc, lds, off) AMD_GROUPSHARED_STORE(acc, lds, (off), 16, true)

#define WM_WAVE_ATTR [WaveSize(32)]

#endif

#endif  // RDNU_WMMA_HLSLI
