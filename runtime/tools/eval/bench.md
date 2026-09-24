# Timing RDNU on RDNA2 and RDNA3

Goal: GPU time of the whole upscale and of each pass at 1080p, 1440p and 4K output on the RX 6800 XT
(DP4a) and the RX 7900 XTX (WMMA, and DP4a for comparison), next to FSR 3.1 and FSR 4.1 on the same
frames.

## Whole upscale: `rdnu_bench`

Built with the `windows` preset (`build/windows/runtime/Release`). It drives the FidelityFX API like a
game, one frame per submission, and reports GPU time between timestamps around `ffxDispatch`.

```
rdnu_bench --render 960x540   --scale 2        # 1080p Performance
rdnu_bench --render 1280x720  --scale 2        # 1440p Performance
rdnu_bench --render 1920x1080 --scale 2        # 4K Performance
rdnu_bench --render 2560x1440 --scale 1.5      # 4K Quality
rdnu_bench --render 1920x1080 --scale 2 --sharpen 0.5
set RDNU_FORCE_DP4A=1 && rdnu_bench --render 1920x1080 --scale 2   # RDNA3 without WMMA
```

Report the median. Close other GPU work, use a fixed power profile in Adrenalin, and run each size
three times.

## The network alone: `rdnu_prod`

```
rdnu_prod runtime\tools\golden --time 200 --size 1920x1088
rdnu_prod runtime\tools\golden --time 200 --size 1920x1088 --force-dp4a
```

The network runs at the render size rounded up to a multiple of 8.

## Per pass: Radeon GPU Profiler

1. Enable RGP capture in Radeon Developer Panel, start `rdnu_bench --frames 1000` (or a game), and
   capture a frame.
2. Every pass is a named event: `RDNU exposure`, the NSS passes (depth scatter, pre-process, offset
   LUT, post-process), `NSS network (WMMA)` or `NSS network (DP4a)`, `RDNU motion` and `RDNU RCAS`
   when used. The Events view gives each region's duration; the Wavefront occupancy view shows
   whether the network is bound by LDS, VGPRs or memory.
3. For the network, compare the per-layer dispatches with the arena layout in
   `runtime/src/engine/rdnu_engine_core.cpp`.

## Against FSR on the same frames

With AMD's DLL beside RDNU as `amd_fidelityfx_upscaler_dx12_original.dll` (see `docs/build-windows.md`):

```
set RDNU_CAPTURE=C:\captures\bistro
set RDNU_COMPARE=4.1
<game or FidelityFX_FSR.exe>
python runtime\tools\eval\compare.py C:\captures\bistro --html bistro.html
```

`RDNU_COMPARE` names AMD's version to run on the same inputs (any substring of its name); the
report scores RDNU against it. Time FSR itself by selecting its version in the FSR sample and
capturing with RGP.
