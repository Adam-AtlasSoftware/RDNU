# Building and testing on Linux

Everything except AMD-specific timing runs on Linux. The D3D12 runtime runs through vkd3d-proton
on any Vulkan driver (Mesa lavapipe included). The HLSL passes and kernels are checked against
Arm's GLSL on Vulkan. The Windows DLL cross-builds with MinGW-w64 and runs under Wine behind AMD's
own loader.

## Packages

| | Arch | Ubuntu 24.04 |
|---|---|---|
| build | `cmake ninja gcc python-numpy` | `cmake ninja-build g++ python3-numpy` |
| Vulkan tests | `vulkan-icd-loader vulkan-headers vulkan-swrast glslang` | `libvulkan-dev mesa-vulkan-drivers glslang-tools` |
| vkd3d-proton | `meson mingw-w64-tools` (widl) | `meson mingw-w64-tools` |
| Windows DLL | `mingw-w64-gcc` | `g++-mingw-w64-x86-64-posix` |
| Wine run | `wine xorg-server-xvfb` | `wine64 xvfb` |

DXC: the Linux release from github.com/microsoft/DirectXShaderCompiler/releases, on `PATH` or
via `-DRDNU_DXC=`. The FidelityFX SDK fork is a submodule that needs SSH access; elsewhere pass
`-DRDNU_SDK_DIR=<clone>`.

## Build and test

```bash
runtime/tools/setup_vkd3d_proton.sh ~/vkd3d --windows   # Linux build, plus the Windows DLLs for Wine
cmake --preset linux -DRDNU_VKD3D_DIR=$HOME/vkd3d/install
cmake --build build/linux --config Release
ctest --preset linux-release
```

The tests need the goldens (`runtime/tools/nss_export.py`, `runtime/tools/nss_pass_frames.py`,
see `docs/build-windows.md`). What ctest runs:

| Test | Checks |
|---|---|
| `engine_core`, `upscale_map` | network planner (layouts, barriers, a CPU run against the golden), FSR-to-NSS mapping |
| `dx12_network`, `dx12_network_drs` | the INT8 network on D3D12, bit-exact to the golden, also at a smaller size |
| `dx12_nss`, `dx12_nss_1_5x` | Arm's NSS component through RDNU's backend on real frames |
| `fsr_api` | the FidelityFX API as a game drives it: exposure modes, sharpening, jittered and display-resolution motion vectors, size changes, the SDK 2 provider object |
| `fsr_capture`, `capture_compare` | provider captures, scored by `compare.py` against the ground truth |
| `vk_network`, `vk_passes` | the HLSL kernels and passes against Arm's GLSL on Vulkan |

vkd3d-proton needs `VKD3D_SHADER_MODEL=6_6` on lavapipe, which ctest sets. The setup script
also patches vkd3d-proton to report native 16-bit ops, which lavapipe runs but vkd3d otherwise
hides.

`cmake --build build/linux --target check_shaders` compiles every shader permutation and checks
that the generated NSS headers match Arm's GLSL.

## The Windows DLL from Linux

```bash
cmake --preset mingw          # uses build/linux's rdnu_shaderc
cmake --build build/mingw
runtime/tools/run_wine_tests.sh build/mingw ~/vkd3d runtime/external/FidelityFX-SDK_WithFSR4
```

`run_wine_tests.sh` runs the Windows binaries under Wine on lavapipe. It drives the DLL through
its own exports and through AMD's signed SDK 2 loader, with AMD's upscaler beside it as the
original. AMD's FSR 3.1 then runs on the same frames for comparison. The DLL lays out the
loader's provider interface itself, so MinGW builds match AMD's MSVC loader.

MinGW-w64 older than 12 lacks some D3D12 declarations; Arch's is current.
