# RDNA3 AI Temporal Upscaler (RDNU)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.11+](https://img.shields.io/badge/PyTorch-1.11+-ee4c2c.svg)](https://pytorch.org/)

## Overview

Welcome to the **RDNA3 AI Temporal Upscaler (RDNU)** project! This repository hosts a high-performance, real-time spatial-temporal upscaling backend specifically optimized for the AMD Radeon RX 7900 XTX (GFX11/RDNA3 architecture).

While traditional anti-aliasing resolves spatial jaggedness, modern video super-resolution (VSR) synthesizes high-frequency details by aggregating data across the temporal dimension. Although the RDNA3 architecture possesses 192 dedicated AI Accelerators designed for matrix operations, these are largely bypassed by current handwritten shader-based solutions like FidelityFX Super Resolution (FSR).

This project bridges that gap by providing a DirectML-accelerated DLL-hijacking framework that utilizes Wave Matrix Multiply-Accumulate (WMMA) hardware to execute a complex neural network for temporal upscaling. The primary testing ground for this implementation is the tactical UE5 FPS **Ready or Not**, which utilizes a standard Unreal rendering pipeline and provides the high-quality motion vectors and depth metadata required for our Decoupled G-buffer Guidance (RDG) model.

## Features

- **Decoupled G-buffer Guidance (RDG):** Utilizes engine-provided high-precision motion vectors and depth maps to align features, drastically reducing latency compared to internal optical flow computation.
- **RDNA3 AI Accelerator Optimization:** Leverages `__builtin_amdgcn_wmma` for ~2x throughput over standard vector paths.
- **FP16 Precision:** Reduces VRAM bandwidth pressure by 50% using Half-precision Tensors.
- **Wave32 Alignment:** Maximizes occupancy on RDNA3 Compute Units by strictly adhering to micro-architectural constraints.
- **DLL-Hijacking Framework:** Seamlessly intercepts rendering data from target games for injection into the DirectML backend.

## Architectural Analysis: RDNA3 and WMMA

Achieving a sub-3ms per-frame inference budget at 1440p requires strict adherence to micro-architectural constraints:

*   **Wave32 Execution & Register Layout:** In Wave32 mode, data in the Vector General Purpose Registers (VGPRs) must be replicated between the two half-waves. Matrix data in lanes 0–15 must be identical to lanes 16–31 to satisfy the internal crossbar logic.
*   **Occupancy & Register Pressure:** If a kernel consumes too many VGPRs, the number of active wavefronts per CU drops, hindering the GPU's ability to hide memory latency.
*   **LDS Tiling:** Using Local Data Share (LDS) Tiling in custom pre-processing compute shaders is critical to prevent memory bandwidth bottlenecks.

## Installation and Setup

### Prerequisites
- Python 3.8+ / PyTorch 1.11+ (for training)
- Windows 10/11 (for the DirectML inference backend) / Linux (for training)
- AMD Radeon RX 7000 Series GPU (Optimized for RX 7900 XTX)

### Cloning the Repository
```bash
git clone https://github.com/Adam-AtlasSoftware/RDNU.git
cd RDNU
```

### Environment Setup (Training)
We provide a setup script to initialize the training environment, download the required datasets (GameIR/M3VIR), and prepare the RDG framework dependencies:

```bash
./start_training.sh
```

## Model Theory & Training

The "Brain" of the upscaler relies on the **RDG (Decoupled G-buffer Guidance)** framework, a causal recurrent model maintaining a high-resolution feature map of temporal history.

We train this model using the **GameIR** or **M3VIR** datasets. To bridge the gap between dataset and actual game engine rendering, we inject a "gaming-native" degradation pipeline during data loading:
- **Halton Jitter:** Applies a sub-pixel offset to the HR image before downsampling to simulate the engine’s TAA/DLSS jitter.
- **Reverse-Z:** Ensures depth maps are converted to the non-linear Reverse-Z format used natively by UE5.

## The Inference Engine

The inference engine translates the trained PyTorch model into a high-performance Windows backend.
- **ONNX Export:** Converts `.pth` weights to `.onnx`.
- **DirectML Custom Backend:** Utilizes ONNX Runtime (ORT) with the DirectML Execution Provider, sharing the `ID3D12Device` and `ID3D12CommandQueue` used by the game engine.
- **WMMA Optimization:** Maps MatMul operations to the `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32` intrinsic for maximum throughput.

## Roadmap

- [x] **Phase I:** Environment Setup & Data Pipeline (RDG Framework Implementation, Dataset Acquisition, Synthetic Degradation Logic)
- [ ] **Phase II:** The Inference Engine (ONNX Export, DirectML Custom Backend Initialization, WMMA Optimization)
- [ ] **Phase III:** The DLL Hijack Framework (Targeting *Ready or Not*, OptiScaler Integration, Resource Capture & Backward Warping)
- [ ] **Phase IV:** POC Validation & Gray Zone Warfare Preparation (Artifact mitigation)
- [ ] **Phase V:** Open Source Plugin Release (DirectML-VSR-AMD plug-and-play D3D12-native path)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to discuss improvements, bug fixes, or optimizations.

## License

This project is licensed under the GPL3 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The [RDG (Efficient Video Super-Resolution for Real-time Rendering)](https://github.com/sunny2109/RDG) authors for the baseline architecture.
- Creators of the [GameIR](https://huggingface.co/datasets/LLLebin/GameIR) and [M3VIR](https://huggingface.co/datasets/guluthemonster/M3VIR) datasets.
- The Unreal Engine rendering community for continued insights into real-time rendering pipelines.