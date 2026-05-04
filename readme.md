# Project: RDNA3 AI Temporal Upscaler

## Overview

This project aims to engineer a high-performance, real-time spatial-temporal upscaling backend specifically optimized for the AMD Radeon RX 7900 XTX (GFX11/RDNA3 architecture). 

While traditional anti-aliasing resolves spatial jaggedness, modern video super-resolution (VSR) synthesizes high-frequency details by aggregating data across the temporal dimension. Although the RDNA3 architecture possesses 192 dedicated AI Accelerators designed for matrix operations, these are largely bypassed by current handwritten shader-based solutions like FidelityFX Super Resolution (FSR). 

This project bridges that gap by providing a DirectML-accelerated DLL-hijacking framework that utilizes Wave Matrix Multiply-Accumulate (WMMA) hardware to execute a complex neural network for temporal upscaling. The primary testing ground for this implementation is the tactical UE5 FPS **Ready or Not**, which utilizes a standard Unreal rendering pipeline and provides the high-quality motion vectors and depth metadata required for our Decoupled G-buffer Guidance (RDG) model.

## Architectural Analysis: RDNA3 and WMMA

The Radeon RX 7900 XTX features 96 Compute Units (CUs), each containing two AI Accelerators. These are specialized hardware blocks designed to perform WMMA operations, which are the fundamental primitives for neural network inference.

Achieving a sub-3ms per-frame inference budget at 1440p requires strict adherence to micro-architectural constraints:

*   **Wave32 Execution & Register Layout:** In Wave32 mode, data in the Vector General Purpose Registers (VGPRs) must be replicated between the two half-waves. Matrix data in lanes 0–15 must be identical to lanes 16–31 to satisfy the internal crossbar logic.
*   **Occupancy & Register Pressure:** If a kernel consumes too many VGPRs, the number of active wavefronts per CU drops, hindering the GPU's ability to hide memory latency.
*   **LDS Tiling:** Using Local Data Share (LDS) Tiling in custom pre-processing compute shaders is critical to prevent memory bandwidth bottlenecks.

## Model Theory: RDG (Decoupled G-buffer Guidance)

The "Brain" of the upscaler relies on the RDG (Decoupled G-buffer Guidance) framework, a highly efficient architecture designed for real-time rendering tasks.

*   **Decoupled G-buffer Paths:** Instead of computing expensive internal optical flow, the RDG model relies on engine-provided high-precision motion vectors (MVs) and depth to align features. This decouples the G-buffer guidance from the main reconstruction task, drastically reducing latency.
*   **Causal Recurrent Framework:** The model maintains a "hidden state" (a high-resolution feature map of temporal history) that is iteratively updated with new information from the current frame.
*   **Input Tensor Engineering:** The model ingests a 5-channel input alongside auxiliary buffers: Color (RGBA), Depth (Z, Reverse-Z format), and High-Precision Motion Vectors.

## Critical Performance Metrics for RX 7900 XTX

| Optimization Method | Implementation Detail | Expected Impact |
| :--- | :--- | :--- |
| **WMMA Mapping** | `__builtin_amdgcn_wmma` | ~2x throughput over standard vector paths. |
| **FP16 Precision** | Half-precision Tensors | 50% reduction in VRAM bandwidth pressure. |
| **Wave32 Alignment** | `numthreads[32, 1, 1]` | Maximize occupancy on RDNA3 CUs. |
| **G-Buffer Guidance** | Decoupled RDG Paths | Improved stability over "black box" spatial-only models. |

## Development Roadmap

### Phase I: Environment Setup & Data Pipeline (Linux Server)
The initial phase focuses on training the RDG model in a raw PyTorch environment on a 4x RTX 3090 cluster.
*   **RDG Framework Implementation:** Clone the RDG repository and configure a Python 3.8+ / PyTorch 1.11+ environment.
*   **Dataset Acquisition:** Utilize **GameIR** or **M3VIR** datasets. GameIR provides 19,200 LR-HR paired frames rendered in Unreal Engine with perfectly synchronized depth and motion vector maps, eliminating the need for manual game capture.
*   **Synthetic Degradation Logic:** Update the training script with a "gaming-native" degradation model:
    *   *Halton Jitter:* Apply a sub-pixel offset to the HR image before downsampling to simulate the engine’s TAA/DLSS jitter.
    *   *Reverse-Z:* Ensure depth maps are converted to the non-linear Reverse-Z format used by UE5.
*   **Training Routine:** Train the RDG-s (Small) variant using FP16 mixed precision, aiming for a target PSNR above 29 dB on the validation set.

### Phase II: The Inference Engine (RDNA3 Desktop)
Translate the trained PyTorch model into a high-performance Windows backend.
*   **ONNX Export:** Convert the `.pth` weights to `.onnx`. Ensure all operators are compatible with the DirectML operator set (avoiding custom Python-based layers).
*   **DirectML Custom Backend:**
    *   Initialize a standalone C++ project using the ONNX Runtime (ORT) with the DirectML Execution Provider.
    *   Use `SessionOptionsAppendExecutionProvider_DML1` to allow the ORT session to share the same `ID3D12Device` and `ID3D12CommandQueue` used by the game engine.
*   **WMMA Optimization (The "Secret Sauce"):**
    *   Verify that ORT is correctly mapping the MatMul operations to the `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32` intrinsic.
    *   Ensure data replication between lanes 0–15 and 16–31 in Wave32 mode.

### Phase III: The DLL Hijack Framework (Targeting Ready or Not)
Intercept the game's rendering data and redirect it to the DirectML backend.
*   **Targeting `nvngx.dll`:** Hook into *Ready or Not* by targeting its DLSS binaries located in `ReadyOrNot\Binaries\Win64`.
*   **OptiScaler Integration:** Use the OptiScaler source as the base to intercept `NVSDK_NGX_D3D12_EvaluateFeature`. Configure the proxy to spoof a VendorID of `0x10DE` (NVIDIA) to unlock the DLSS menu in the game.
*   **Resource Capture & Synchronization:** 
    *   Hook the `EvaluateFeature` call to grab `ID3D12Resource` pointers for the jittered color, depth, and motion vector buffers.
    *   Implement D3D12 Fences to ensure the AI backend does not process until the game has finished writing to those buffers.
*   **Backward Warping Compute Shader:** Write a custom HLSL compute shader that gathers pixels from the previous hidden state using motion vectors to align them with the current frame before feeding the tensor into the RDG network.

### Phase IV: POC Validation and Gray Zone Warfare Preparation
*   **Ready or Not Testing:** Enable DLSS in the menu. Use the OptiScaler overlay to verify the DirectML backend receives correct buffer dimensions. Address any "vibration" artifacts by implementing a Center-Shift layer in the model to account for Halton jitter.
*   **Anti-Cheat Strategy:** While local DLL replacement is acceptable for *Ready or Not*, prepare for multiplayer implementations (like *Gray Zone Warfare*) by transitioning to a Global Hook or the `DXGI_PROXY_PATH` environment variable method to avoid Easy Anti-Cheat (EAC) detection.
*   **Open Source & Collaboration:** Document the TFLOPS performance. Once stabilized, package the project as a **DirectML-VSR-AMD** plugin, providing a "plug-and-play" D3D12-native path for developers (like Madfinger Games) to support RDNA3 matrix accelerators without rewriting their post-processing stacks.

## Conclusion

This roadmap provides a concrete path from training a research-grade RDG model to deploying a high-performance gaming implementation that finally unlocks the 120+ TFLOPS of matrix-math potential in the RX 7900 XTX, directly benefiting Unreal Engine 5 titles.