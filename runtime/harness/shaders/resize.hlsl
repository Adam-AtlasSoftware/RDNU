// resize.hlsl - bilinear resize matching torch F.interpolate(mode='bilinear', align_corners=False).
// Channels are preserved; only H,W change. Source index (align_corners=False, per PyTorch
// area_pixel_compute_source_index): src = scale*(dst+0.5) - 0.5, clamped to >= 0 for bilinear;
// integer indices clamped to [0, size-1]. Used for the decoder up-path and coarse-scale g/depth/flow.
//
// cbuffer slots used: Cout = channels, H, W = input dims, OH, OW = output dims.

StructuredBuffer<float>   In  : register(t0);   // [C * H * W]
RWStructuredBuffer<float>  Out : register(u0);   // [C * OH * OW]

cbuffer Dims : register(b0)
{
    uint Cin; uint H; uint W; uint Cout;
    uint KH; uint KW; uint PadH; uint PadW;
    uint StrideH; uint StrideW; uint Groups; uint OH; uint OW;
};

[numthreads(8, 8, 1)]
void resize_CS(uint3 tid : SV_DispatchThreadID)
{
    uint ox = tid.x, oy = tid.y, oc = tid.z;
    if (ox >= OW || oy >= OH || oc >= Cout)
        return;

    float scaleH = float(H) / float(OH);
    float scaleW = float(W) / float(OW);

    float sy = scaleH * (float(oy) + 0.5f) - 0.5f;  sy = max(sy, 0.0f);
    float sx = scaleW * (float(ox) + 0.5f) - 0.5f;  sx = max(sx, 0.0f);

    uint y0 = uint(floor(sy)); uint y1 = min(y0 + 1, H - 1); float fy = sy - float(y0);
    uint x0 = uint(floor(sx)); uint x1 = min(x0 + 1, W - 1); float fx = sx - float(x0);

    uint base = oc * H * W;
    float v00 = In[base + y0 * W + x0];
    float v01 = In[base + y0 * W + x1];
    float v10 = In[base + y1 * W + x0];
    float v11 = In[base + y1 * W + x1];

    float top = v00 * (1.0f - fx) + v01 * fx;
    float bot = v10 * (1.0f - fx) + v11 * fx;
    Out[(oc * OH + oy) * OW + ox] = top * (1.0f - fy) + bot * fy;
}
