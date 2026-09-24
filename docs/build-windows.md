# Building and running on Windows

RDNU builds one DLL that speaks the FidelityFX API. It is installed under three names:

| File | Loaded by |
|---|---|
| `amd_fidelityfx_dx12.dll` | FSR 3.1 games, OptiScaler's FSR 3.1 path |
| `amd_fidelityfx_upscaler_dx12.dll` | FidelityFX SDK 2 games, through AMD's `amd_fidelityfx_loader_dx12.dll` |
| `rdnu_dx12.dll` | hosts that load RDNU explicitly |

Placed beside AMD's DLL renamed `<name>_original.dll`, RDNU serves upscaling and forwards
everything else to AMD: frame generation, swap chains, and hosts that pick another upscaler
version. Weights and DXIL are embedded; nothing is compiled at run time.

## Prerequisites

| Tool | Notes |
|---|---|
| Visual Studio 2022+ or Build Tools, C++ workload | MSVC, Windows SDK (`d3d12`, `dxc.exe`) |
| CMake 3.21+ and Ninja | the `windows` preset |
| Git LFS | model weights |
| Python 3 with numpy | goldens and `compare.py` |
| PowerShell 7 | the FSR sample build wrapper |

```powershell
git clone --recurse-submodules git@github.com:Adam-AtlasSoftware/RDNU.git
cd RDNU
git lfs pull
```

DXC is taken from `PATH`, then from the newest Windows SDK. The AMD wave-matrix headers come from
the FidelityFX SDK submodule (`Kits/FidelityFX/api/internal/dx12/AmdExtD3D`); without them RDNA3 runs
the DP4a kernels.

## Build

From a VS developer prompt, or VS Code's CMake Tools with the `windows` preset:

```powershell
cmake --preset windows
cmake --build build/windows --config Release
ctest --test-dir build/windows -C Release --output-on-failure
```

Outputs in `build/windows/runtime/Release/`:

| Target | What it is |
|---|---|
| `amd_fidelityfx_dx12.dll` and its two copies | the upscaler |
| `rdnu_prod` | the INT8 network against the exporter golden, `--time N` for timing |
| `rdnu_bench` | a whole upscale through the FidelityFX API, GPU timed (`runtime/tools/eval/bench.md`) |
| `test_nss_dx12`, `test_fsr_api` | Arm's component and the FidelityFX API on real frames |
| `rdnu_engine`, `run_engine` | the fp32/INT8 reference engine against the NSS goldens |

The tests need the goldens: `python runtime/tools/nss_export.py` and
`python runtime/tools/nss_pass_frames.py <sequence.safetensors> --start 60 --frames 16`
(a Bistro sequence from the Arm neural graphics dataset, see `docs/Datasets.md`).

## FSR sample

The SDK fork's FSR sample still carries the retired RDG backend. Apply the RDNU patch once, and
once per machine give Cauldron its vcpkg packages through vcpkg's MSBuild integration:

```powershell
git -C runtime/external/FidelityFX-SDK_WithFSR4 am ../../integration/0001-FSR-sample-run-RDNU-through-the-FidelityFX-API.patch
.\runtime\external\vcpkg\bootstrap-vcpkg.bat
.\runtime\external\vcpkg\vcpkg integrate install
cmake --build build/windows --config Release --target run_fsr_sample
```

The sample's post-build step copies RDNU's DLL from `build/windows` in as
`amd_fidelityfx_upscaler_dx12.dll` and keeps AMD's as the original. Method "RDNU (AI)" pins the
upscale context to RDNU; "FSR (ffxapi)" runs AMD's newest version. "Override FSR Version" picks any
version, and "Draw upscaler debug view" shows NSS's debug tiles.

## Games

1. Find the game's FidelityFX DLL next to its executable.
2. Rename it to `<name>_original.dll`, for example `amd_fidelityfx_dx12_original.dll`.
3. Copy RDNU's DLL in under the original name.

RDNU reports itself first in the version list, so it is the default upscaler; AMD's versions stay
selectable. DLSS and XeSS games go through OptiScaler: choose its FSR 3.1 (FidelityFX API)
upscaler for DX12 and put RDNU's DLL where OptiScaler loads `amd_fidelityfx_dx12.dll` (or
`amd_fidelityfx_upscaler_dx12.dll` for OptiScaler builds that use the SDK 2 DLLs); check
OptiScaler's documentation for the current option names. NSS reads colour, depth, motion vectors,
jitter, exposure and the camera; it ignores reactive and transparency masks.

Environment variables for testing:

| Variable | Effect |
|---|---|
| `RDNU_LOG=<file>` | appends RDNU's messages to a file; games rarely show the API's message callback |
| `RDNU_FORCE_DP4A=1` | the RDNA2 kernels on RDNA3, for A/B timing |
| `RDNU_CAPTURE=<dir>` | dumps every dispatch for `runtime/tools/eval/compare.py` |
| `RDNU_CAPTURE_FRAMES=N` | frames per context to capture (300) |
| `RDNU_COMPARE=<version>` | with a capture, also runs AMD's upscaler of that version on the same inputs |

## Troubleshooting

- **RDNU is not listed or the game falls back to FSR:** the device lacks SM 6.4 or native 16-bit
  types, or context creation failed. Set `RDNU_LOG` and read the file; each context logs its sizes
  and flags, and errors say what failed.
- **`dxc` not found:** pass `-DRDNU_DXC=<path to dxc.exe>`.
- **Sample build fails on `rdg_dx12_backend.cpp`:** the patch above is not applied.
- **Sample: cannot open `directx/d3d12.h` or `pix3.h`:** re-run `vcpkg integrate install`; the
  integration stores an absolute path.
- **Submodule folder empty:** `git submodule update --init --recursive`.
