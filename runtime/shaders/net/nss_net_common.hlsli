// nss_net_common.hlsli - shared by the NSS INT8 conv kernels (DP4a and wave matrix).
//
// Tensors are NHWC int8 in byte-address buffers, 4 channels per 32-bit word. A tensor view is
// (base, pitch, pix): byte address of pixel (x, y) channel c is base + y * pitch + x * pix + c,
// where base already points at valid pixel (0, 0) and the view's channel offset. Arena tensors
// carry a border: one pixel left/top, and at least one pixel right/bottom, all holding -128
// (the code for 0.0), so every 3x3 tap is in bounds and zero padding costs nothing. The arena is
// cleared to 0x80 once; producers write the right/bottom ring each frame (StoreRing*), so a
// shrinking render size never exposes stale data.
//
// Requantisation is integer-exact and identical to runtime/tools/nss_export.py:
//   acc  = sum x * w + C[oc]                  (C folds the -128 zero point and the int32 bias)
//   relu : q = clamp(round(f32(acc) * r[oc]) - 128, -128, 127)
//   head : l = clamp(round(f32(acc) * r[oc]) + zp_logit, -128, 127); q = LUT[l + 128]
// One int->float convert and one multiply: no fused multiply-add, so every device rounds alike.

#ifndef NSS_NET_COMMON_HLSLI
#define NSS_NET_COMMON_HLSLI

#define ACT_RELU    0
#define ACT_SIGMOID 1

#define OUT_INT8     0   // NHWC int8 buffer (arena tensor or the KPN coefficient buffer)
#define OUT_TEMPORAL 1   // RGBA8_SNORM texture (the temporal/feedback tensor)

#define QZERO_WORD 0x80808080u   // four int8 -128 codes

cbuffer NetConstants : register(b0)
{
    uint  InBase;       // byte offset of input pixel (0,0), channel offset included
    uint  InPitch;      // bytes per input row
    uint  InPix;        // bytes per input pixel
    uint  InW;          // valid input size (bounds for unbordered input)
    uint  InH;
    uint  OutBase;
    uint  OutPitch;
    uint  OutPix;
    uint  OutW;         // valid output size
    uint  OutH;
    uint  OutRing;      // 1: write the -128 ring right/below the valid region
    uint  Cout;
    uint  WOff;         // DP4a layout [chunk][tap][cin/4][OCB] or OHWI [cout][3][3][cin]
    uint  BiasOff;      // int32 C[cout]
    uint  ScaleOff;     // float r[cout]
    uint  LutOff;       // int8 LUT[256] (sigmoid layers)
    int   ZpLogit;
    uint  GroupsX;      // grid width, for linearised dispatches
    uint  Pad0;
    uint  Pad1;
};

ByteAddressBuffer   Weights : register(t0);
RWByteAddressBuffer Output  : register(u0);
RWByteAddressBuffer Input   : register(u1);
#if OUT_KIND == OUT_TEMPORAL
RWTexture2D<snorm float4> TemporalOut : register(u2);
#endif

int InAddr(int x, int y)  { return int(InBase) + y * int(InPitch) + x * int(InPix); }
int OutAddr(int x, int y) { return int(OutBase) + y * int(OutPitch) + x * int(OutPix); }

int RequantRelu(int acc, float r)
{
    return clamp(int(round(float(acc) * r)) - 128, -128, 127);
}

int RequantLogit(int acc, float r)
{
    return clamp(int(round(float(acc) * r)) + ZpLogit, -128, 127);
}

uint PackQ4(int4 q)
{
    return (uint(q.x) & 0xFFu) | ((uint(q.y) & 0xFFu) << 8) | ((uint(q.z) & 0xFFu) << 16) | ((uint(q.w) & 0xFFu) << 24);
}

int4 UnpackQ4(uint w)
{
    return int4(int(w << 24) >> 24, int(w << 16) >> 24, int(w << 8) >> 24, int(w) >> 24);
}

// Border ring: the pixel right of, below, and diagonal to the last valid pixel must read as
// -128 for the next layer. Written by the threads that own the last column/row, for the
// producer's own channel range [0, nbytes) of the view.
void StoreRingWords(int x, int y, uint byteOff, uint nwords)
{
    if (OutRing == 0)
        return;
    bool lastX = x == int(OutW) - 1;
    bool lastY = y == int(OutH) - 1;
    for (uint i = 0; i < nwords; ++i)
    {
        if (lastX)
            Output.Store(uint(OutAddr(x + 1, y)) + byteOff + i * 4, QZERO_WORD);
        if (lastY)
            Output.Store(uint(OutAddr(x, y + 1)) + byteOff + i * 4, QZERO_WORD);
        if (lastX && lastY)
            Output.Store(uint(OutAddr(x + 1, y + 1)) + byteOff + i * 4, QZERO_WORD);
    }
}

#endif // NSS_NET_COMMON_HLSLI
