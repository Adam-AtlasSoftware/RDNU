# Building on Windows (VS Code, no Visual Studio IDE)

The goal of this setup is to build, run and debug the RDNU DX12 upscaler entirely from
**VS Code** — the Visual Studio IDE is never needed, only its command-line build tools.

## Prerequisites

| Tool | Notes |
|------|-------|
| **MSVC build tools (v145)** | The FSR sample targets `PlatformToolset v145` (Visual Studio 2022 18.x / matching Build Tools) with the *Desktop development with C++* workload. If your install uses a different toolset, change `<PlatformToolset>` in `runtime/external/FidelityFX-SDK_WithFSR4/Samples/Upscalers/FidelityFX_FSR/dx12/FidelityFX_FSR_2022.vcxproj`. |
| **Windows 10/11 SDK** | Provides `d3d12`, `dxgi`, `dxcapi.h`, `d3dcompiler` (used by the sample and the smoke tests). |
| **Git + Git LFS** | Weights in `runtime/models/**/*.bin` are stored via LFS. |
| **CMake ≥ 3.21 + Ninja** | For the `runtime/` CMake project (shader validation + smoke tests). |
| **PowerShell 7 (`pwsh`)** | Used by the VS Code build task wrapper. |
| **VS Code extensions** | Accept the workspace recommendations (`.vscode/extensions.json`): C/C++, CMake Tools, HLSL Tools, Python. |

## First-time setup

```powershell
git clone --recurse-submodules git@github.com:Adam-AtlasSoftware/RDNU.git
cd RDNU
git lfs pull
```

The FidelityFX SDK fork already ships its resolved vcpkg dependencies
(`Kits/Cauldron2/dx12/vcpkg_installed/`), so no separate `vcpkg install` step is required to
build the sample.

## Building the FSR sample (the actual upscaler)

The RDNU backend (`runtime/src/rdg_dx12_backend.cpp`), shaders and model weights are compiled
into / copied by AMD's FSR DX12 sample. Build it from VS Code:

- **Terminal → Run Build Task** (`Ctrl+Shift+B`) → **Build FSR Sample (Release)** *(default)*.
- Or **Build FSR Sample (Debug)** / **Rebuild FSR Sample (Release)**.

These run [`runtime/tools/build_sample.ps1`](../runtime/tools/build_sample.ps1), which locates
MSBuild via `vswhere` and builds
`runtime/external/FidelityFX-SDK_WithFSR4/Samples/Upscalers/FidelityFX_FSR/dx12/FidelityFX_FSR_2022.sln`.

You can also run it directly:

```powershell
pwsh runtime/tools/build_sample.ps1 -Config Release          # or -Config Debug [-Rebuild]
```

The post-build step copies `runtime/shaders/rdg_*.hlsl` and `runtime/models/RDNU_{FP16,INT8}/`
next to the executable, so the backend finds them via relative paths at runtime.

## Running / debugging

Press **F5** and pick a configuration (`.vscode/launch.json`):
- **Debug FSR Sample (DX12, Debug)** — builds Debug first, launches under the MSVC debugger.
- **Run FSR Sample (DX12, Release)** — builds Release first, then runs.

The output executable is
`runtime/external/FidelityFX-SDK_WithFSR4/Samples/Upscalers/FidelityFX_FSR/dx12/x64/<Config>/FidelityFX_FSR.exe`.

## The CMake project (`runtime/`)

CMake here does **not** build the DX12 backend (that lives in the sample). It provides:

- `validate_shaders` — best-effort DXC validation of the HLSL compute shaders.
- `rdnu_test_dxc`, `rdnu_test_compile` — tiny toolchain smoke tests (opt-in).

```powershell
# Configure (CMake Tools does this automatically, or:)
cmake --preset windows

# Best-effort shader validation (needs dxc on PATH — e.g. from the Windows SDK bin)
cmake --build build/windows --config Release --target validate_shaders
```

> The WMMA convolution shader is a work-in-progress template that depends on AMD wave-matrix
> intrinsics; stock DXC will reject it until the kernel is finished. Validation failures for
> `rdg_wmma_conv2d.hlsl` are expected for now.

## Intellisense

`.vscode/c_cpp_properties.json` adds the FidelityFX SDK (`Kits`), the SDK's `vcpkg_installed`
include tree, and DirectX-Headers so headers like `<FidelityFX/upscalers/include/ffx_upscale.h>`
resolve in the editor.

## Troubleshooting

- **`vswhere not found` / `MSBuild not found`** — install Visual Studio 2022+ or the standalone
  *Build Tools for Visual Studio* with the C++ workload.
- **Toolset mismatch (`v145` not found)** — install the matching MSVC toolset or edit
  `<PlatformToolset>` in the sample `.vcxproj`.
- **Missing weights / shaders at runtime** — run `git lfs pull` and rebuild (the post-build copy
  stages assets next to the exe).
- **Submodule folder empty** — `git submodule update --init --recursive`.
