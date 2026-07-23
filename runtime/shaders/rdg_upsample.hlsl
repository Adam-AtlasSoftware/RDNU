// rdg_upsample.hlsl
Texture2D<float4> InputColor : register(t0);
RWTexture2D<float4> UpscaledOutput : register(u1);

SamplerState LinearSampler : register(s0);

cbuffer Constants : register(b0)
{
    float frameDelta;
    float jitterX;
    float jitterY;
    uint renderWidth;
    uint renderHeight;
    uint upscaleWidth;
    uint upscaleHeight;
};

[numthreads(8, 8, 1)]
void RDG_Upsample_CS(uint3 DTid : SV_DispatchThreadID)
{
    // Prevent out of bounds writes
    if (DTid.x >= upscaleWidth || DTid.y >= upscaleHeight) return;

    // Calculate input pixel coordinate corresponding to this output pixel
    float inputX = (DTid.x + 0.5f) * ((float)renderWidth / (float)upscaleWidth);
    float inputY = (DTid.y + 0.5f) * ((float)renderHeight / (float)upscaleHeight);

    // Prevent bilinear footprint from bleeding into unrendered padding
    inputX = min(inputX, renderWidth - 0.5f);
    inputY = min(inputY, renderHeight - 0.5f);

    // Get physical dimensions of the input texture since it might be padded
    uint w, h;
    InputColor.GetDimensions(w, h);

    // Compute true UV coordinate for the input texture
    float2 uv = float2(inputX / (float)w, inputY / (float)h);

    // Sample and output
    UpscaledOutput[DTid.xy] = InputColor.SampleLevel(LinearSampler, uv, 0);
}
