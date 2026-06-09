# RDNA3 AI Temporal Upscaler (RDNU)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)

## Overview

The RDNA3 AI Temporal Upscaler (RDNU) is a spatial-temporal upscaling backend optimized for the AMD Radeon RX 7900 XTX (GFX11/RDNA3 architecture).

While traditional anti-aliasing resolves spatial jaggedness, modern video super-resolution (VSR) algorithms synthesize high-frequency details by aggregating sub-pixel shifts across a temporal dimension. This project implements a recurrent neural network upscaler that evaluates native game engine G-Buffers to reconstruct high-resolution outputs.

The inference engine relies on a DirectML-accelerated DLL-hijacking framework designed to intercept rendering data from modern graphics APIs and execute the model using Wave Matrix Multiply-Accumulate (WMMA) hardware instructions. This interception occurs at the API level, providing compatibility across arbitrary rendering engines and pipelines. *Note: The C++/DirectML scaffolding for the rendering integration is maintained separately and will be merged into this repository in a future phase.*

## Features

- **Recurrent G-Buffer Guidance:** The model ingests multi-frame temporal histories (currently an 8-frame window) alongside native engine data, including Depth, Surface Normals, Albedo (BRDF), and High-Precision Optical Flow.
- **Fractional Scaling & RCAS:** Implements fractional scale handling (e.g., 1.5x downscaling to mimic the DLSS/FSR "Quality" preset) with frequency-preserving antialiased interpolation and Robust Contrast Adaptive Sharpening (RCAS) for micro-detail recovery.
- **RDNA3 AI Accelerator Optimization:** Targets `__builtin_amdgcn_wmma` for increased throughput over standard vector ALU paths.
- **FP16 Precision:** Operates on Half-precision tensors to reduce VRAM bandwidth utilization.
- **Wave32 Alignment:** Configured to respect RDNA3 Compute Unit micro-architectural constraints, maintaining optimal active wavefront occupancy.

## Architectural Constraints: RDNA3 and WMMA

Meeting strict per-frame inference budgets at high output resolutions necessitates specific hardware alignment:

*   **Wave32 Execution & Register Layout:** In Wave32 mode, data in the Vector General Purpose Registers (VGPRs) is replicated between half-waves. Matrix input mappings must satisfy internal crossbar logic.
*   **Occupancy & Register Pressure:** Kernel configurations must strictly manage VGPR consumption to prevent drops in active wavefronts per CU, which masks memory latency.
*   **LDS Tiling:** Custom pre-processing compute shaders utilize Local Data Share (LDS) tiling to avoid memory bandwidth saturation.

## Installation and Setup

### Prerequisites
- Python 3.8+ / PyTorch 2.0+
- Windows 10/11 (for the DirectML inference backend) / Linux (for training)
- AMD Radeon RX 7000 Series GPU (Targeting RX 7900 XTX)

### Cloning the Repository
```bash
git clone https://github.com/Adam-AtlasSoftware/RDNU.git
cd RDNU
```

### Environment Setup (Training)
Training is handled via distributed PyTorch (DDP). Execution scripts are provided to configure the environment, define GPU utilization, and launch the distributed loop.

```bash
# Launch the Base x2 architecture distributed training loop
./scripts/training/start_training_base_x2.sh
```

## Model Architecture & Training Pipeline

The core network is a causal recurrent model based on the Decoupled G-buffer Guidance (RDG) framework. The current iteration (Base variant) utilizes 36 feature channels to retain complex texture dictionaries and relies on SSIM, L1, FFT, and Temporal Consistency loss functions.

### Dataset Processing
The network is trained against the **MPI-Sintel** and **Virtual KITTI 2 (vKITTI)** datasets. The data loaders dynamically synthesize incomplete G-Buffers to ensure the model processes a pipeline identical to commercial game engines:
- **Surface Normal Derivation:** 3D Surface Normals are calculated analytically on the CPU using Sobel operators across the 16-bit depth maps.
- **Albedo/BRDF Mapping:** Material boundaries are derived using raw Class Segmentation maps.
- **Engine Flow:** The dataloaders parse raw `.flo` binaries and 16-bit vector maps, applying magnitude scaling appropriate for the targeted output resolution.
- **Degradation:** Halton sequence jitter is applied prior to downsampling to mimic TAA characteristics.

## The Inference Engine (Pending Integration)

The runtime environment handles the translation of PyTorch weights to a DirectML execution context.
- **ONNX Export:** Translates the `.pth` model state to `.onnx`.
- **DirectML Backend:** Uses ONNX Runtime (ORT) with the DirectML Execution Provider, attaching to the target engine's `ID3D12Device`.
- **DLL Interception:** Hooks into the render pipeline to extract matrices, depth, and motion vectors required by the model.

## Roadmap

- [x] **Phase I:** Environment Setup & Data Pipeline (Recurrent Network Implementation, Synthetic G-Buffer Derivation, RCAS Integration)
- [x] **Phase II:** Distributed Training (Net2Net inflation, SSIM tuning, Temporal alignment)
- [ ] **Phase III:** The Inference Engine (ONNX Export, DirectML Backend Initialization, WMMA Optimization)
- [ ] **Phase IV:** The DLL Hijack Framework (Universal API Hooks, Resource Capture & Backward Warping)
- [ ] **Phase V:** Artifact Mitigation & Open Source Plugin Release

## License

This project is licensed under the GPL3 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The [RDG (Efficient Video Super-Resolution for Real-time Rendering)](https://github.com/sunny2109/RDG) authors for the baseline architecture concept.
- Creators of the [MPI-Sintel](http://sintel.is.tue.mpg.de/) and [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) datasets.
- Creators of the [GameIR](https://huggingface.co/datasets/LLLebin/GameIR) and [M3VIR](https://huggingface.co/datasets/guluthemonster/M3VIR) datasets.
