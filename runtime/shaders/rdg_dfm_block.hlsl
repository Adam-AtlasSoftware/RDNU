// rdg_dfm_block.hlsl
// RDG Dynamic Feature Modulator (DFM) & CycleFC Block

Texture2D<float2> MotionVectors : register(t1);
Texture2D<float>  DepthBuffer : register(t2);
Texture2DArray<min16float4> PrevHiddenState : register(t3);
RWTexture2DArray<min16float4> NextHiddenState : register(u0);
RWTexture2D<float4> UpscaledOutput : register(u1);

Texture2D<float> ShiftWeightsX : register(t4); // 1x7 shifts
Texture2D<float> ShiftWeightsY : register(t5); // 7x1 shifts
SamplerState PointSampler : register(s0);

min16float4 FetchCycleFCPixel(Texture2DArray<min16float4> inputTex, int2 pixelCoord, int channelSlice)
{
    int2 offset = int2(0, 0);
    int shiftMode = channelSlice % 4;

    if (shiftMode == 0) offset = int2(-1, 0);
    else if (shiftMode == 1) offset = int2(1, 0);
    else if (shiftMode == 2) offset = int2(0, -1);
    else if (shiftMode == 3) offset = int2(0, 1);

    return inputTex.Load(int4(pixelCoord + offset, channelSlice, 0));
}

void ApplyDFM_Guidance(inout min16float4 featurePixel, float currentDepth, float2 currentMV)
{
    float mvMag = length(currentMV);
    min16float modulation = (min16float)(1.0f + 0.1f * currentDepth + 0.5f * mvMag);
    featurePixel = featurePixel * modulation;
}

float2 ApplySpatialShift(int2 inCoord, int channelSlice)
{
    float shiftX = ShiftWeightsX.Load(int3(0, channelSlice, 0));
    float shiftY = ShiftWeightsY.Load(int3(0, channelSlice, 0));
    return float2(inCoord.x + shiftX, inCoord.y + shiftY);
}

[numthreads(8, 8, 1)]
void RDG_DFM_Block_CS(uint3 DispatchThreadId : SV_DispatchThreadID)
{
    int2 outCoord = int2(DispatchThreadId.xy);

    // Pixel shuffle assumption: 2x scale
    int2 inCoord = outCoord / 2;
    int sliceOffset = (outCoord.y % 2) * 2 + (outCoord.x % 2);

    // Apply 1x7 / 7x1 spatial UV shifts instead of heavy convolutions
    float2 shiftedCoord = ApplySpatialShift(inCoord, sliceOffset);
    int2 finalSampleCoord = int2(shiftedCoord); // In a real implementation this would sample with interpolation

    float depth = DepthBuffer.Load(int3(inCoord, 0));
    float2 mv = MotionVectors.Load(int3(inCoord, 0)).xy;

    min16float4 features = FetchCycleFCPixel(PrevHiddenState, finalSampleCoord, sliceOffset);
    ApplyDFM_Guidance(features, depth, mv);

    // Output directly to the screen buffer
    UpscaledOutput[outCoord] = float4(features);
}
