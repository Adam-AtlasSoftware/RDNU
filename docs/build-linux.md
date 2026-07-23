# Linux / Nvidia — status and porting plan

> **TL;DR:** The *training* half of RDNU already runs on Linux. The *real-time upscaler* half
> is currently Windows + DX12 + AMD-only and does **not** yet run on Linux or Nvidia. This
> document is the plan for getting an inference/validation harness onto an Nvidia GPU on Linux.

## What already works on Linux

- **Training** (`RDG/`, `scripts/`) — PyTorch/BasicSR, distributed training, data synthesis.
- **Weight export** — `runtime/tools/export_weights.py` converts a trained `.pth` to the OHWI
  FP16/INT8 `.bin` layout the shaders consume. This is pure PyTorch/NumPy and is
  platform-independent.

## Why the runtime doesn't "just build" on Linux

The current runtime is DX12 + HLSL, integrated into AMD's FidelityFX/Cauldron sample:

- **DirectX 12** has no native Linux implementation (only translation layers such as
  vkd3d-proton).
- The HLSL compute shaders use **AMD RDNA3 wave-matrix (WMMA) intrinsics**
  (`AmdExtD3DShaderIntrinsicsMatrixOps.hlsl`), which are AMD-hardware-specific and won't run on
  Nvidia.
- AMD's own ML upscaler (FSR4) ships as a **signed binary backend**, not as reusable shaders —
  so it is not a drop-in path for a custom model, and its ML path is AMD-GPU-only regardless.

Getting onto **Nvidia + Linux** is therefore a *port*, not a build-configuration change.

## The plan: a standalone cross-platform inference harness

Rather than porting AMD's Windows sample, add a self-contained harness under `runtime/` that
runs the RDG graph directly on the GPU and validates it against a PyTorch reference. Two viable
backends:

1. **Vulkan compute + cooperative matrix** (recommended for portability)
   - Use `VK_KHR_cooperative_matrix` (or `VK_NV_cooperative_matrix`) for the INT8/FP16 GEMM tiles,
     the Nvidia analogue of the AMD WMMA path.
   - Compile the compute shaders with DXC (SPIR-V) or port them to GLSL/Slang.
   - Runs on Nvidia, and also on AMD/Intel where the extension is supported.

2. **CUDA** (fastest path to a working Nvidia reference)
   - Implement the conv/CTR/DFM/upsample stages with cuBLASLt / CUTLASS INT8 tensor-core GEMMs.
   - Least portable, but the quickest way to get a numerically-correct baseline on the RTX box.

### Suggested milestones

1. **Reference dump** — extend `export_weights.py` / a test script to dump per-layer input and
   output tensors from PyTorch for a fixed frame (golden values).
2. **Harness skeleton** — a CMake target (behind `RDNU_BACKEND_VULKAN` / a new `RDNU_BACKEND_CUDA`)
   that loads `runtime/models/**/*.bin`, allocates GPU buffers, and dispatches one layer.
3. **Layer-by-layer bring-up** — validate each stage (INT8 conv + dequant, CTR temporal warp,
   DFM UV-shift, U-Net skips, upsample) against the golden values.
4. **Full-frame** — chain the graph, compare the upscaled output (PSNR/SSIM) to the reference.

## Building the CMake project on Linux today

The `linux` preset configures the cross-platform bits (the Windows-only DX12 sample and MSVC
smoke tests are disabled):

```bash
git clone --recurse-submodules git@github.com:Adam-AtlasSoftware/RDNU.git
cd RDNU && git lfs pull
cmake --preset linux
cmake --build build/linux --config Release        # currently: shader validation only
```

`RDNU_BACKEND_VULKAN` is a placeholder option today — enabling it only prints a "not
implemented" notice. It is where the Vulkan harness from the plan above will hang off.

## Notes on hardware

- Testing on an **Nvidia** GPU means the AMD WMMA HLSL path is not reusable as-is; the harness
  must use Vulkan cooperative-matrix or CUDA tensor-core kernels instead.
- FSR4 / FSR4.1 (including the RDNA3 INT8 path) is AMD-hardware-only and does not apply to the
  Nvidia validation target — it is useful only as an A/B comparison on an AMD machine.
