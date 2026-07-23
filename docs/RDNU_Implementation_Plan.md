# RDNU HLSL Shader Upscaler Implementation Plan

## Objective
Implement the RDNU INT8 model architecture in HLSL for integration into the Cauldron rendering framework (`rdg_dx12_backend.cpp` and `fsrapirendermodule.cpp`).

## Core Requirements & Specifications

1.  **INT8 Dequantization Math**:
    *   Formula: `(INT32_Accum * Weight_Scale * Act_Scale) + Bias`
    *   The `INT8` weights are packed in `OHWI` format, and the sequence of data in the `.bin` is:
        1. `[INT8]` Quantized Weights
        2. `[FP16]` Weight Scales (per-channel)
        3. `[FP16]` Activation Scale (scalar)
        4. `[FP16]` Bias (per-channel)
    *   This logic will be primarily contained within the `rdg_wmma_conv2d.hlsl` shader and any other convolution blocks.

2.  **CTR (Cross-Temporal Rendering) Layer**:
    *   **Persistent UAVs**: Allocate Read/Write UAVs to hold the temporal history (hidden states) across frames.
    *   **Motion Vector Warping**: Before applying attention, the temporal history must be warped using the current frame's motion vectors.
    *   This logic will go into `rdg_ctr_block.hlsl` and require C++ side updates to manage the persistent UAVs across frames.

3.  **U-Net Skip Connections**:
    *   **VRAM Management**: Store the intermediate outputs from the encoder layers.
    *   **Scaling and Merging**: During the decoder phase, multiply these stored skip connections by their respective `FP16` `skip_alphas` before adding/merging them with the decoder upsampled features.

4.  **DFM (Dynamic Feature Modulator) Blocks**:
    *   **Spatial UV Shifts**: Instead of using expensive 1x7 and 7x1 spatial convolutions, implement these as UV shifts when sampling the features. This acts as a highly optimized depthwise convolution approximation.
    *   This logic will reside in `rdg_dfm_block.hlsl`.

## Step-by-Step Implementation

### Step 1: `rdg_wmma_conv2d.hlsl` (Dequantization & Core Conv)
*   Update the shader to read the INT8 weights and FP16 scales/biases.
*   Implement the exact dequantization math: `Output = (Accum_Int32 * w_scale * a_scale) + bias`.
*   Ensure the cooperative load logic from LDS matches the OHWI format and padded dimensions (multiples of 16).

### Step 2: `rdg_ctr_block.hlsl` (Temporal History & Warping)
*   Ensure the shader correctly samples `PrevHiddenState` using a sampler and warped UV coordinates: `warpedUV = uv + mv`.
*   The `PrevHiddenState` must be bound as a readable texture, and `NextHiddenState` as a RWTexture.

### Step 3: `rdg_dfm_block.hlsl` (UV Shift Approximation)
*   Modify `RDG_DFM_Block_CS` to read the shift weights.
*   Implement the 1x7 and 7x1 convolutions as texture samples with offset UVs based on those weights, rather than a full MAC loop.

### Step 4: `rdg_dx12_backend.cpp` (Resource Management & Binding)
*   **Persistent UAVs**: Update the backend context creation to allocate textures that persist between frames to hold the CTR temporal history and the U-Net skip connections.
*   **Skip Alphas**: Read the `skip_alphas` from the `.bin` or metadata header and bind them appropriately so the shader can multiply the skip connections before merging.
*   **INT8 Pipeline**: Ensure the pipeline state objects (PSOs) and root signatures correctly define the buffers for INT8 weights, FP16 scales, and biases as required by the `.bin` layout.

### Step 5: `fsrapirendermodule.cpp` (Integration)
*   Ensure the FSR API module correctly routes the motion vectors and depth buffers to the RDNU backend.
*   Verify the upscale dispatch passes the correct resolution parameters.
