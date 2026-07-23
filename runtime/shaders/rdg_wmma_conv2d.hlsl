// RDG FP16 WMMA Convolution Template (RDNA3 Optimized)
// Requires AMD AGS Intrinsics

#define AmdExtD3DShaderIntrinsics_EnableWaveMatrix
// AMD RDNA3 WaveMatrix (WMMA) intrinsics. The shader compiler must be given the include dir:
//   runtime/external/FidelityFX-SDK_WithFSR4/Kits/FidelityFX/api/internal/dx12/AmdExtD3D
#include "AmdExtD3DShaderIntrinsicsMatrixOps.hlsl"

Texture2D<float4> InputColor : register(t0);
Texture2D<float2> InputMV    : register(t1);
Texture2D<float>  InputDepth : register(t2);

// RDG uses packed FP16 arrays (e.g. 48 channels = 12x min16float4)
Texture2DArray<min16float4> PrevHiddenState : register(t3);
RWTexture2DArray<min16float4> NextHiddenState : register(u0);
RWTexture2D<float4> UpscaledOutput : register(u1);

// PyTorch Weights (Bound as Constant Buffers or SRVs)
// OHWI format: [Output, Height, Width, Input]
StructuredBuffer<int> QuantizedWeights : register(t4); // Packed INT8
StructuredBuffer<min16float> WeightScales : register(t5);
StructuredBuffer<min16float> Biases : register(t6);

cbuffer LayerConstants : register(b0) {
    min16float ActivationScale;
    uint OutChannelOffset;
}

// LDS (Local Data Share) for cooperative loading
groupshared min16float4 lds_features[16][16];
groupshared int lds_weights[16][16];

[numthreads(32, 1, 1)] // RDNA3 Wave32
void RDG_Conv2D_WMMA_CS(uint3 GroupId : SV_GroupID, uint3 DispatchThreadId : SV_DispatchThreadID, uint GroupIndex : SV_GroupIndex)
{
    // INT8 Native AMP Path
    AmdWaveMatrixB < AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_Int8, 16, 16 > inputMatrix8;
    AmdWaveMatrixB < AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_Int8, 16, 16 > weightMatrix8;
    AmdWaveMatrixAccumulator < AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_Int32, 16, 16 > accumulator32;

    // Cooperative Load into LDS
    uint2 localCoord = uint2(GroupIndex % 16, GroupIndex / 16);
    if (GroupIndex < 256) {
        // Load spatial tiles cooperatively to reduce VRAM bandwidth
        uint2 globalCoord = GroupId.xy * 16 + localCoord;
        lds_features[localCoord.y][localCoord.x] = PrevHiddenState.Load(int4(globalCoord.x, globalCoord.y, 0, 0));

        // Load specific layer weights from buffer
        lds_weights[localCoord.y][localCoord.x] = QuantizedWeights[localCoord.y * 16 + localCoord.x];
    }
    GroupMemoryBarrierWithGroupSync();

    // Load from LDS to WaveMatrix
    // inputMatrix8.Load(lds_features[0][0]); // Simplified load intrinsic conceptually
    // weightMatrix8.Load(lds_weights[0][0]);

    // Execute INT8 WMMA instruction (16x16x16 Matrix Multiply-Accumulate)
    // accumulator32.MultiplyAccumulate(inputMatrix8, weightMatrix8);

    // Dequantization: (INT32_Accum * Weight_Scale * Act_Scale) + Bias
    // Note: OutChannel is determined by the specific thread mapping in the wave
    uint outChannel = OutChannelOffset + localCoord.x; // Simplified channel mapping

    // Dequantize EXACTLY as specified: (INT32_Accum * Weight_Scale * Act_Scale) + Bias
    // min16float dequantized = (min16float)accumulator32.Result * WeightScales[outChannel] * ActivationScale + Biases[outChannel];

    // Apply Activation (GELU approximation)
    // float x = (float)dequantized;
    // float gelu = 0.5 * x * (1.0 + tanh(sqrt(2.0 / 3.14159) * (x + 0.044715 * pow(x, 3))));

    // Since we don't have the full accumulator macro ready,
    // let's pass-through some basic logic to ensure output is written
    min16float4 pixelData = lds_features[localCoord.y][localCoord.x];
    min16float4 result = pixelData; // Placeholder until Wave Matrix is fully wired

    // Output to NextHiddenState
    if (GroupIndex < 256) {
        NextHiddenState[int3(GroupId.xy * 16 + localCoord, 0)] = result;
    }
}