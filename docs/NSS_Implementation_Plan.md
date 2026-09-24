# NSS implementation plan: everything buildable without AMD hardware

Scope: all work that can be completed and verified on a Linux/Windows dev box without an RDNA GPU
and without retraining. Every package ends with (a) what is verified here and (b) the one-line GPU
gate that runs later on the RX 7900 XTX / RX 6800 XT. Quality and performance take priority over
ease: the network runs INT8 NHWC with fused epilogues on DP4a (RDNA2) and wave-matrix (RDNA3), the
Arm pass shaders are ported unchanged in algorithm, and the result ships as an FSR-3.1-compatible
DLL.

Read first: [NSS_Runtime_Plan.md](NSS_Runtime_Plan.md) (architecture, passes),
[Model_Candidates.md](Model_Candidates.md) §4.1 (NSS card), [NSS_Quality_Plan.md](NSS_Quality_Plan.md).
Reference-only material: the leaked FSR4 source (HLSL structure, INT8 conventions; copy nothing).

Tools available on the dev box: Linux DXC (`runtime/tools/check_shaders.sh`; SM 6.2–6.6, AGS wave
matrix needs the AMD DXC and `-I <FidelityFX-SDK>/Kits/FidelityFX/api/internal/dx12/AmdExtD3D`),
CPU PyTorch + the model gym (`nss_export.py` works), g++ for platform-free C++ unit tests. MSVC/D3D12
is not available: DX12-specific C++ is written against headers and first compiled on the Windows box.

---

## 0. Facts that fix the design (verified in this repo)

**Network I/O contract**, defined by Arm's pass shaders in their buffer-tensor variant
(`sdk/include/FidelityFX/gpu/nss/ffx_nss_preprocess.h` `WriteInputTensorPacked`,
`ffx_nss_postprocess.h` `ReadKpnParamsInt8FromBase` / `SampleTemporalParams`):

| Tensor | Resource on DX12 | Layout | Quantiser |
|---|---|---|---|
| input (`Resource_0_input`) | `RWByteAddressBuffer`, `Wp·Hp·12` bytes | NHWC int8: pixel `p = y·Wp + x`, 12 channels as 3 × int8x4 (`uint`): `[hist.rgb, col.r] [col.gb, motion_detector, fb.r] [fb.gba, luma_deriv]` | written already quantised by pre-process: `s = 0.003912`, `zp = −128` |
| kpn (`Resource_1_output`) | `RWByteAddressBuffer`, `Wk·Hk·36` bytes | NHWC int8: `((y·Wk + x)·9 + c/4)` uint, channel `c%4` | `q = round(sigmoid/0.003937) − 127`, `q ∈ [−127,127]` (`kKpnQuant`) |
| temporal (`Resource_2_output`, next frame's `FEEDBACK_TENSOR`) | `Texture2D R8G8B8A8_SNORM` at `Wp×Hp`, UAV + SRV, sampled bilinear-clamp | 4 ch per pixel | SNORM: `v = 2·sigmoid − 1`, stored as `round(v·127)` (`kTemporalQuant = (0.5, −1)`) |

`Wp×Hp` = `dataGraphSize` = render size padded up to a multiple of 8 (`FFX_NSS_RESOURCE_ALIGNMENT`);
`Wk = Wp/4`, `Hk = Hp/4`. Constants `_InputTensorSize`, `_PaddingScale`, `_KpnDimension`, `_KpnScale`
are already computed by `ffx_nss.cpp::setupConstantBuffer`.

**Backbone** (`programNSS`, 14 convs): channels 12→32(s2)→32→32(s2)→32→32→64(s2)→64→32 | up
→16 | cat48→32 | kpn 32→36 | up →16 | cat48→16→16 | up → temporal 16→4. 148,456 params,
11,304 MAC/input px. Every hidden activation is post-ReLU (non-negative) and quantised per-tensor
with `zp = −128`; weights per-channel symmetric. Layers with two consumers (`conv2d_1`, `conv2d_3`,
`conv2d_8`) have one shared output quantiser (asserted by the exporter).

**Arm SDK backend contract** (`sdk/include/FidelityFX/host/ffx_interface.h`, SDK 1.1.2 + Arm
extensions): `FFX_GPU_JOB_DATA_GRAPH = 6`, `FFX_RESOURCE_TYPE_TENSOR`, `fpCreateDataGraphPipeline`,
`FfxPipelineState::{srv,uav}TensorBindings`, `FfxDataGraphTensorInfo{resourceName, bufferAliased}`,
resource states `FFX_RESOURCE_STATE_DATA_GRAPH_{READ,WRITE}`, device caps `dataGraphSupported`,
`computeSupportTensor`. The component schedules the network as one job carrying the three tensors
by name.

**FFX API** (`ffx-api/`): providers are compiled into a table (`ffx_provider.cpp`); a provider
claims a create-desc type (`CanProvide`), has a version id/name, and implements
Create/Destroy/Configure/Query/Dispatch. `ffxDispatchDescUpscale` (FSR 3.1) carries colour, depth,
MVs, exposure, reactive, T&C, output, jitter, MV scale, render/upscale size, sharpening, frame time,
pre-exposure, reset, camera near/far/FoV, flags.

## Status (2026-09-24)

All nine packages are implemented. Everything below runs in `ctest` on Linux (lavapipe through
vkd3d-proton, and Vulkan for the GLSL cross-checks), and on the Windows binaries under Wine
(`runtime/tools/run_wine_tests.sh`).

| Package | Verified here |
|---|---|
| WP1, WP3, WP4 | INT8 network bit-exact to the exporter golden on D3D12, also at a smaller size in the same arena; plan invariants from 8×8 to 4K |
| WP5 | every pass against Arm's GLSL on real Bistro frames, 2× and 1.5× (codes ±1 at rounding ties) |
| WP6 | Arm's component end to end on D3D12; PSNR equal to the Vulkan run |
| WP7 | FidelityFX API as a game drives it: exposure modes, sharpening, motion vector flags, size changes, forwarding; AMD's signed SDK 2 loader gives identical output |
| WP8 | captures scored by `compare.py` reproduce the test's PSNR; AMD's FSR 3.1.5 captured on the same inputs |
| WP9 | CMake presets for Windows, Linux and MinGW; `check_shaders` over every permutation |

Where the build departs from the plan:

- **Backend:** RDNU's own D3D12 backend (`src/backend_dx12`) replaces the FidelityFX 1.1.3 one.
  It caches pipelines and shares one network engine across contexts, so a new render size only
  allocates textures.
- **Provider:** a standalone FidelityFX API DLL rather than a provider in the fork's table. It
  also serves FidelityFX SDK 2's loader, forwards to AMD's DLL, and cross-builds with MinGW.
- **Jitter:** NSS's sign is FSR's negated (measured, about 4 dB either way); the provider flips it.
- **Exposure:** computed on the GPU and patched into the NSS constants; Arm's shaders are unchanged.
- **Motion vectors:** jitter cancellation and display-resolution vectors are supported through a
  small pass that mirrors FSR 3, not rejected.
- **Capture:** in the provider (`RDNU_CAPTURE`, `RDNU_COMPARE`), so it works in games as well as the
  sample.
- **FSR sample:** a patch for the fork (`runtime/integration`); the fork is not modified here.
- **Not supported:** `GPU_MEMORY_USAGE_V2` (context-free memory estimate), reactive and
  transparency masks, non-linear colour input (warned). Dynamic resolution restarts the history
  at each new size.

---

## 1. Target layout

```
runtime/
├── shaders/nss/                      # WP5: HLSL pass shaders (replaces the placeholder rdg_*.hlsl)
│   ├── ffx_nss_common_hlsl.h         #   types, resource accessors, sampler, tensor buffer helpers
│   ├── ffx_nss_{depth_scatter,disocclusion_mask_lq,pre_process,generate_offset_lut,post_process,debug_view}.hlsl
│   └── ffx_rcas_pass.hlsl            #   sharpening (FidelityFX RCAS, MIT)
├── shaders/net/                      # WP3: production network kernels
│   ├── nss_conv3x3_dp4a.hlsl         #   SM 6.4, RDNA2 and fallback
│   ├── nss_conv3x3_wmma.hlsl         #   SM 6.6 + AGS wave matrix, RDNA3
│   └── nss_net_common.hlsli          #   layouts, epilogues, quant helpers shared by both
├── src/engine/                       # WP4
│   ├── rdnu_manifest.h               #   model manifest (layer table) parser, platform-free
│   ├── rdnu_engine_core.h/.cpp       #   arena planning, dispatch descriptors, platform-free
│   └── rdnu_engine_dx12.h/.cpp       #   PSOs, root signature, weight upload, recording
├── src/backend_dx12/                 # WP6: FidelityFX DX12 backend adapted to Arm's interface
├── src/provider/                     # WP7: ffx_provider_rdnu_upscale.cpp, DLL entry, RCAS, exposure
├── harness/                          # existing reference engine; gains --prod and pass harness
├── tools/nss_export.py               # WP1: extended (manifest, NHWC INT8 bundle)
├── tools/nss_pass_golden.py          # WP5: pre/post pass goldens from the torch pipelines
├── tools/eval/compare.py             # WP8
└── tests/core/                       # WP4: g++ unit tests for manifest/arena/layout math
```

Delete when WP7 lands: `runtime/shaders/rdg_*.hlsl`, `runtime/src/rdg_dx12_backend.cpp`,
`runtime/src/ffx_provider_rdg.h`.

---

## 2. Work packages

### WP1. Model manifest and NHWC INT8 export (`runtime/tools/nss_export.py`)

Goal: one artefact describes the network to the engine, so a widened/retrained variant is a data
change. Extend the exporter; keep the existing CHW bundles (the harness reference path).

Deliverables:
1. `nss.manifest.json` (also embedded in the bundle as a string tensor `manifest`):
   ```
   { "version": 1, "input": {"channels": 12, "quant": {"scale": 0.003912, "zp": -128}},
     "layers": [ { "name": "conv2d_0", "cin": 12, "cout": 32, "stride": 2, "act": "relu",
                   "in": "input", "out": "c0", "upsample_in": false,
                   "in_channel_offset": 0, "in_channel_stride": 12,
                   "out_channel_offset": 0, "out_channel_stride": 32,
                   "w_off": <bytes>, "w_bytes": 3456, "acc_bias_off": ..., "m_off": ..., "b_off": ...,
                   "in_quant": {"scale": s, "zp": -128}, "out_quant": {"scale": s', "zp": -128},
                   "out_kind": "int8" | "kpn_int8" | "temporal_snorm" }, ... ],
     "tensors": [ {"name": "c0", "channels": 32, "res_div": 2}, {"name": "cat7", "channels": 48, "res_div": 4}, ... ] }
   ```
   Concats are expressed as channel offsets into a wider tensor: `conv2d_7 → cat7[0:16]`,
   `conv2d_3 → cat7[16:48]` (and `conv2d_4` reads `cat7` with offset 16, stride 48);
   `conv2d_9 → cat9[0:16]`, `conv2d_1 → cat9[16:48]` (`conv2d_2` reads offset 16, stride 48).
   Nearest upsample is a flag on the consumer (`upsample_in: true` for `conv2d_7`, `conv2d_9`,
   `temporal_params_out_conv`): the kernel maps tap `(y+dy, x+dx)` to source `((y+dy)>>1, (x+dx)>>1)`.
   No concat or upsample passes exist.
2. Weight blob `nss_w8.bin`, per layer, in this order and 16-byte aligned:
   - `wq`: int8 OHWI `[cout][3][3][cin_pad]`, `cin_pad = cin` for DP4a (cin is 12/16/32/48/64, all
     multiples of 4) and the same blob serves WMMA for `cin ∈ {16,32,48,64}` (`conv2d_0` stays DP4a).
   - `acc_bias`: int32 `[cout]` = `128 · Σ_k wq[oc,k]` (folds the −zp of the input quantiser so the
     kernel accumulates raw `xq` without subtracting −128).
   - `m`: fp32 `[cout]` = `s_in · s_w[oc]`; `b`: fp32 `[cout]`.
   - `cout_pad` = round up to 16 for WMMA layers (36→48, 4→16 are DP4a layers, unpadded).
3. NHWC INT8 golden bundle `nss_prod.rdnut`: `input_nhwc` (uint8 bytes as float, `Wp·Hp·12`),
   `golden_kpn_nhwc` (int8, `Wk·Hk·36`), `golden_temporal` (int8 SNORM `Wp·Hp·4`), plus per-layer
   int8 activations (`chk.<tensor>`) computed by the exporter's integer-path reference **in the
   packed layout** so the production kernels are compared byte for byte. Input must be quantised
   exactly as the pre-process does: `q = clamp(round(x/s) − 128, −128, 127)`.
4. Assertions: shared quantiser equality for multi-consumer tensors; `acc_bias` fits int32; the
   sum `Σ|wq|·127·9·cin` < 2³¹ per layer (it is: max ≈ 127·127·9·64 ≈ 9.3M).

Verify here: run on the released checkpoints; `python -m json.tool` the manifest; the integer-path
reference reproduces the existing CHW INT8 golden after layout conversion (add that self-check).
GPU gate: none (data only).

### WP2. Reference path stays as is

`programNSS`/`programNSSInt8` in the harness are the correctness reference (already written,
compile clean). No changes except loading `nss_prod.rdnut` for the production comparison (WP4).

### WP3. Production network kernels (`runtime/shaders/net/`)

Shared (`nss_net_common.hlsli`): NHWC addressing `addr(p, c) = p·C_stride + c_off + c`, spatial
padding handled by a 1-pixel zero border inside `Wp×Hp`? No: the tensors are exactly `Wp×Hp` and the
pre-process writes valid data on every pixel; convs use zero padding at the image edge, so taps
outside `[0,W)×[0,H)` contribute 0. Implement by clamping in the tile loader (LDS staging), not by
padding memory: the loader writes zeros for out-of-range taps. Epilogue (both kernels):
```
v = float(acc + acc_bias[oc]) * m[oc] + b[oc]
relu:            q = clamp(round(v / s_out) - 128, -128, 127)          -> int8 NHWC, packed 4 ch / uint
kpn_int8:        q = clamp(round(sigmoid(v) * 254) - 127, -127, 127)   -> int8 NHWC
temporal_snorm:  RWTexture2D<snorm float4>[p] = 2*sigmoid(v) - 1       -> SNORM texture (hardware rounds)
```
`round` = `floor(x + 0.5)` (matches the reference). Sigmoid in fp32.

**A. `nss_conv3x3_dp4a.hlsl` (SM 6.4, `dot4add_i8packed`), RDNA2 and RDNA3 fallback.**
- Thread group 8×8 output pixels (64 threads, one wave on RDNA); each thread computes one pixel ×
  `OC_PER_THREAD` output channels, looped over cout in chunks (chunk = 16 for cout 32/64; 12 for 36;
  4 for the temporal head; 32 for cout 32 when cin ≤ 16 to raise arithmetic intensity).
- LDS tile: `(8·stride + 2) × (8·stride + 2)` input pixels × `cin` bytes (max: stride 2, cin 32 →
  18·18·32 = 10.4 KB; stride 1, cin 64 → 10·10·64 = 6.4 KB; upsample-in layers load the
  half-res source tile `(8/2+2)²`). Loaded cooperatively as `uint` (4 channels), zero-filled
  outside the image.
- Inner loop per tap (9) per channel group (`cin/4`): `acc[oc] = dot4add_i8packed(x_packed, w_packed[oc], acc[oc])`.
  Weights for the current oc chunk and all taps are loaded once per thread from the weight
  `ByteAddressBuffer` (K$-resident; ≤ 16·9·64 = 9.2 KB per chunk).
- Register budget: `acc[16]` + weights streamed per tap (16 × cin/4 uints for the current tap:
  ≤ 256 regs for cin 64 → stream per tap-and-group instead: load `w` for (tap, group) as 16 uints,
  do 16 dot4s). Target ≤ 96 VGPRs for occupancy ≥ 5 waves/SIMD; check with RGA offline
  (`rga -s dx12 --isa` runs on Linux against DXIL, no GPU needed) and record VGPR counts in the PR.
- Output packing: 4 consecutive channels per `uint` store; `out_channel_offset` must be a
  multiple of 4 (it is: 0 or 16).

**B. `nss_conv3x3_wmma.hlsl` (SM 6.6, AGS `AmdWaveMatrix*`), RDNA3 only, layers `conv2d_1..11`.**
- Formulation: `C[M = pixels, N = cout_pad] = Σ_tap A_tap[M, K = cin] · B_tap[K, N]`.
- A tiles: 16 consecutive **linear** pixels `p..p+15` of the NHWC input, `A.Load(buffer,
  base + (p + tapOffset(tap))·C_stride + c_off + kt·16, stride = C_stride, rowMajor)`; a run of 16
  pixels may wrap a row: those pixels are garbage and their outputs are discarded by the
  epilogue (`x ≥ W` or tap out of range). Edge zero-padding for WMMA: since taps outside the
  image must read zero, allocate every WMMA-consumed tensor with a 1-pixel border: `Wb = W+2`,
  `Hb = H+2`, valid pixels at `(x+1, y+1)`, borders zeroed once per frame by the producing
  kernel's epilogue (it writes zeros for border pixels). `conv2d_0` (DP4a) therefore writes `c0`
  with a border; every WMMA layer both reads and writes bordered tensors; the DP4a heads read
  bordered inputs (loader handles the offset). Upsample-in layers: `tapOffset` maps through the
  half-res source (`(y+dy)>>1`), which breaks the "16 consecutive pixels" contiguity for A; so
  `conv2d_7`, `conv2d_9` and the temporal head stay on DP4a (they are 3 of the 14 layers and
  ~9% of MACs). WMMA layers: `conv2d_1..6, conv2d_8, conv2d_10, conv2d_11` (cin ∈ {32,48,64,16},
  K multiple of 16 ✓; cout ∈ {32,64,16} ✓).
- B tiles: `B.Load(weights, w_off + tap·cin·cout_pad + kt·16·cout_pad + nt·16, stride = cout_pad,
  rowMajor)` with weights re-laid out per layer as `[tap][cin][cout_pad]` (K-major rows of N) — the
  exporter emits this second layout for WMMA layers (`w_off_wmma`).
- Wave32 per `16 px × 64 oc` (4 accumulators reusing each A tile 4×) or `64 px × 16 oc` (reusing
  B); pick per layer by cout: cout 64 → 16×64, cout 32 → 32×32 (2×2 accumulators), cout 16 → 64×16.
  Workgroup = 4 waves = 4 tiles along M. K loop: `for tap in 9: for kt in cin/16: acc = mul(A,B,acc)`.
- Epilogue from the accumulator: `acc.Store` to a small LDS scratch (16×16 int32 per wave), then
  the same epilogue code as DP4a reads LDS, applies bias/scale/act/quant and packs int8 NHWC.
- Stride-2 layers (`conv2d_2`, `conv2d_4`): A rows are every other pixel → not contiguous; use the
  DP4a kernel for those two as well, or run them as WMMA on a space-to-depth copy. Decision:
  DP4a first (they are 2 layers); revisit in the perf pass with measurements.
  Final WMMA set: `conv2d_1, conv2d_3, conv2d_5, conv2d_6, conv2d_8, conv2d_10, conv2d_11`
  (≈ 58% of MACs); DP4a: `conv2d_0, conv2d_2, conv2d_4, conv2d_7, conv2d_9, kpn, temporal`.

Verify here: `check_shaders.sh` compiles both (`-T cs_6_4` for DP4a; `-T cs_6_6 -I$AGS_INC` for
WMMA with the AMD DXC when available, otherwise the WMMA file is checked on Windows); RGA ISA
dumps for VGPR/LDS/occupancy; a C++ scalar model of both kernels' index math in `tests/core`
(the tile loader's clamp/zero logic and address formulas) run against the manifest.
GPU gate: `rdnu_engine --prod nss_prod.rdnut` matches every `chk.*` byte for byte on both paths.

### WP4. Engine library (`runtime/src/engine/`)

- `rdnu_manifest.h`: parse the manifest (no JSON dependency: emit the manifest also as a flat
  binary table `RDNM` from the exporter; parse that, keep JSON for humans).
- `rdnu_engine_core.{h,cpp}` (no D3D includes): given manifest + `Wp,Hp` → tensor sizes, arena
  offsets (one DEFAULT buffer, sub-allocated; 256-byte aligned; peak at 4K output ≈ 90 MB), per
  layer dispatch descriptors `{pso kind, root constants (dims, strides, offsets, quant, w offsets),
  grid}`, barrier list (UAV barriers between dependent layers only: `cat7`/`cat9` producers are
  independent and need no barrier between them).
- `rdnu_engine_dx12.{h,cpp}`: root signature = 32 root constants + weights SRV (`ByteAddressBuffer`)
  + input SRV + output UAV + (temporal) `RWTexture2D` UAV + AGS UAV slot; PSO per (kind, layer
  shape) compiled once at context creation from embedded DXIL (offline-compiled by CMake, no
  runtime DXC); `Record(cmdList, inputBuf, kpnBuf, temporalTex, Wp, Hp)`.
- Path selection: `RDNA3 && AGS wave matrix OK` → WMMA set; else DP4a for everything. Expose
  `--force-dp4a` for A/B timing.
- Harness: `rdnu_engine.cpp --prod <bundle>` loads `nss_prod.rdnut`, runs the production engine on
  the same input, compares `chk.*` and outputs; `--time N` loops N frames with timestamp queries.

Verify here: `tests/core/test_manifest_arena.cpp` (g++): sizes/offsets/alignment for 1080p, 1440p,
4K and odd sizes (e.g. 1290×722 → padded 1296×728); dispatch grids; barrier list; a scalar
implementation of the epilogue and tile loader validated against the exporter's `chk.*` for a
16×16 crop (pure C++, no GPU). GPU gate: as WP3.

### WP5. Pass shaders (`runtime/shaders/nss/`)

Port Arm's headers to HLSL without changing the algorithms. The headers are FFX-style: all
GLSL-specific code is in resource accessors (~100 sites) plus type names.

1. `ffx_nss_common_hlsl.h`, mirror of `ffx_nss_common_glsl.h`:
   - types: `vec2/3/4 → float2/3/4`, `int32_t2/4 → int2/4`, `uint32_t4 → uint4`, `int16_t4`
     (needs `-enable-16bit-types`), `int8_t4 → int16_t4` in registers with `pack_s8`/`unpack_s8s16`
     (SM 6.4) at the buffer boundary, `half` = 16-bit float (`FFX_HALF=1`), literals `1.0HF → 1.0h`.
   - constant buffer `NssConstants` (`cbuffer` at `b0`, same field order as `ffx_nss_private.h`),
     accessor functions unchanged.
   - resources: `Texture2D<float4>` / `RWTexture2D<...>` / `SamplerState s_LinearClamp` (static
     sampler in the root signature) / `ByteAddressBuffer` for tensors; `textureLod → SampleLevel`,
     `texelFetch → Load(int3)`, `textureGather → GatherRed/Green/…`, `imageStore → [] =`,
     `tensorReadARM/WriteARM` → never used (compile with `NSS_SUPPORT_TENSOR 0`), the std430
     buffer helpers become `Load`/`Store` on `ByteAddressBuffer` with the same linear index.
   - temporal tensor: `Texture2D<float4>` bound to the SNORM texture; `Dequantize(..., kTemporalQuant)` unchanged.
2. Per pass `.hlsl` wrapper (compute, `[numthreads(16,16,1)]`, entry `main`), the same
   `NSS_BIND_*` indices as the GLSL wrappers translated to `t#/u#/b#` registers, `#define`s
   (`NSS_PREPROCESS 1`, `NSS_FILTER_MODE`, quality mode, `NSS_V1_SHARP_THETA`) taken from Arm's
   `CMakeShadersNSS.txt` permutations; only the `high` quality permutations are built first.
3. `ffx_rcas_pass.hlsl`: FidelityFX RCAS (MIT) as the optional sharpening pass on the output.
4. Binding tables for the DX12 backend (name → register) generated from the same `NSS_BIND_*`
   macros into `ffx_nss_shaderblobs_dx12.cpp` (mirrors Arm's `ffx_nss_shaderblobs.cpp`).

Goldens (`runtime/tools/nss_pass_golden.py`): load 8 frames of `ml/data/arm-neural-graphics-dataset/nss/test`
(LFS pull that split), run `preprocess_torch` and `postprocess_torch` from the gym with the `high`
settings and the released fp32 backbone in between, and dump for each pass: every input resource
and every output resource as `.rdnut` (float32 planes) plus PNG previews, and the `NssConstants`
values used. The torch pipelines are Arm's own reference for the shaders (they assert parity with
Slang), so this is a legitimate golden. Harness: `rdnu_pass_harness.cpp` runs one HLSL pass on those
inputs (textures created from planes) and compares outputs (fp16 tolerance 1e-3 relative, int8
exact).

Verify here: all six passes compile with DXC (`cs_6_4`, 16-bit types); the golden tool runs on CPU;
a `diff` of the ported headers against the GLSL shows only accessor/type edits (keep the port as
a patch series per file for review). GPU gate: pass harness within tolerance on all 8 frames.

### WP6. DX12 backend for Arm's interface (`runtime/src/backend_dx12/`)

Base: FidelityFX SDK 1.1.x `sdk/src/backends/dx12/ffx_dx12.cpp` (+ `ffx_dx12.h`, shader-blob
accessors) from the fork; adapt to Arm's `ffx_interface.h`/`ffx_types.h`:
- New fps: `fpCreateDataGraphPipeline` → creates an `RdnuEngineDx12` from the manifest/weights
  embedded in the DLL, fills `FfxPipelineState::srvTensorBindings` = {`Resource_0_input`},
  `uavTensorBindings` = {`Resource_1_output`, `Resource_2_output`} by name from
  `dataGraphTensorInfo`; `fpCreateOpticalFlowPipeline` → `FFX_ERROR_INVALID_ARGUMENT` (NFRU later).
- `FFX_RESOURCE_TYPE_TENSOR` in `fpCreateResource`: `PREPROCESS_INPUT_TENSOR`, `KPN_TENSOR` →
  DEFAULT-heap buffers (`W·H·channels` bytes, UAV|SRV); `FEEDBACK_TENSOR` → `Texture2D
  R8G8B8A8_SNORM` (UAV|SRV). Note the component allocates the feedback tensor as ping-pong
  (`HISTORY_1/2`-style) itself; follow its resource table (`ffx_nss.cpp` ~L760–830).
- `FFX_GPU_JOB_DATA_GRAPH` in `fpExecuteGpuJobs`: transition the three resources
  (`DATA_GRAPH_READ` → `NON_PIXEL_SHADER_RESOURCE`, `DATA_GRAPH_WRITE` → `UNORDERED_ACCESS`), call
  `engine.Record(...)`, UAV barriers after.
- `fpGetDeviceCapabilities`: `dataGraphSupported = true`, `computeSupportTensor = false` (buffer
  path), fp16 = true; RDNA detection via `D3D12_FEATURE_DATA_ARCHITECTURE1` + AMD vendor id +
  AGS `agsGetGPUInfo` / wave-matrix probe (reuse `rdnu_wmma_probe` logic) to pick WMMA vs DP4a.
- Shader blobs: DXIL compiled at build time by CMake (`dxc.exe`, or Linux DXC for CI) into
  `.h` byte arrays per permutation; reflection replaced by the hand-written binding tables (WP5.4).
- Barriers: map Arm's resource states 1:1 (`FFX_RESOURCE_STATE_*` already exist in the DX12 backend
  except the two data-graph states).

Verify here: compiles nowhere on Linux; keep it small and mechanical; write it against the
headers with a checklist of every `fp*` symbol in Arm's `FfxInterface` (all must be assigned in
`ffxGetInterfaceDX12`). GPU gate: `ffxNssContextCreate/Dispatch` runs in the FSR sample and the
debug view (`FFX_API_NSS_DISPATCH_FLAG_DRAW_DEBUG_VIEW`) shows every intermediate.

### WP7. Provider, DLL, sample integration (`runtime/src/provider/`)

1. `ffx_provider_rdnu_upscale.cpp`: `CanProvide(FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE)`,
   `GetId() = 0x0001'0000'RDNU-versioned`, `GetVersionName() = "RDNU NSS 1.0"`. Mapping:

   | FSR 3.1 Upscale | NSS |
   |---|---|
   | create: `maxRenderSize`, `maxUpscaleSize`, flags `HDR`, `DEPTH_INVERTED`, `DEPTH_INFINITE`, `MOTION_VECTORS_JITTER_CANCELLATION` (not supported by NSS: reject with a clear error), `ENABLE_DEBUG_CHECKING` | `ffxApiCreateContextDescNss` with the same sizes; flags `QUANTIZED`, `HDR`, `DEPTH_INVERTED/INFINITE`, `MANAGE_HISTORY`, `ALLOW_16BIT`, quality mode `QUALITY` (= NSS high) |
   | dispatch: `color, depth, motionVectors, output, jitterOffset, motionVectorScale, renderSize, upscaleSize, reset, cameraNear/Far/FovAngleVertical, frameTimeDelta` | passed through 1:1 |
   | `exposure` resource / `preExposure` | NSS takes a float: a 1-thread pass copies `exposure[0,0] · preExposure` into a 4-byte buffer read by the pass shaders' `Exposure()` accessor (port the accessor to read from that buffer when bound); if no resource, run FidelityFX SPD auto-exposure (from the fork) into the same buffer |
   | `enableSharpening`, `sharpness` | RCAS pass on `output` after NSS |
   | `reactive`, `transparencyAndComposition` | ignored (v1); logged once |
   | queries: upscale ratio / render resolution from quality mode | FSR's table: Quality 1.5, Balanced 1.7, Performance 2.0, Ultra Performance 3.0 (NSS is trained at 2×; ratios ≠ 2 are configurable but unvalidated; report `FFX_API_RETURN_OK` and document) |
   | queries: jitter phase count / offset | `ffxNssGetJitterOffset` / NSS phase-count query |
   | `GPU_MEMORY_USAGE` | from the backend's effect memory query |
   | configure `KEYVALUE` (fVelocityFactor) | accepted, no-op |

2. Register the provider first in the table so it wins `CanProvide(UPSCALE)`; keep
   `ffxProvider_Nss` for hosts that speak NSS directly.
3. DLL: target `amd_fidelityfx_dx12.dll` exporting `ffxCreateContext`, `ffxDestroyContext`,
   `ffxConfigure`, `ffxQuery`, `ffxDispatch` (same `FFX_API_ENTRY` set); also build it under the
   name `rdnu_dx12.dll` for hosts that load it explicitly. Weights and DXIL embedded (no files on
   disk). No DXC at runtime.
4. FSR sample: in the fork's `fsrapirendermodule.cpp`, add an "RDNU" upscaler choice that loads
   `rdnu_dx12.dll` through `ffx_api_loader.h` and drives it with the same dispatch it already builds
   for FSR 3.1; keep a debug-view toggle.
5. Injection for games: drop-in DLL swap for FSR 3.1 titles, or OptiScaler with its FSR 3.1 DX12
   path pointed at the DLL (OptiScaler feeds DLSS/XeSS inputs through the same struct).

Verify here: the mapping table is implemented as a pure function with a g++ unit test
(`tests/core/test_upscale_mapping.cpp`) over representative descs (HDR, inverted depth, missing
exposure, each quality mode). GPU gate: the sample runs RDNU with the debug view; then a game via
DLL swap.

### WP8. Evaluation tooling

- Sample-side capture (GPU-gated): dump per frame `color, depth, mv, jitter, exposure, camera`
  and the RDNU output and, with the FSR 4.1 DLL selected, the FSR4 output, as EXR/PNG + JSON.
- `runtime/tools/eval/compare.py` (here): PSNR, SSIM, LPIPS(alex), FLIP, tPSNR (per Arm's
  definition), CGVQM if installed; per-sequence table + crops (thin geometry, foliage, text,
  specular under motion, disocclusion); HTML report. Inputs: two image sequences + optional GT.
- `runtime/tools/eval/bench.md`: procedure to time with RGP at 1080p/1440p/4K on the 6800 XT and
  7900 XTX, network and passes separately, DP4a vs WMMA (`--force-dp4a`).

### WP9. Build, docs, tests

- CMake: `rdnu_core` (platform-free, builds on Linux with tests), `rdnu_dxil` (DXC offline compile
  of `shaders/net` and `shaders/nss`, with the CI-checkable Linux path), `rdnu_dx12` (Windows),
  `rdnu_provider_dll` (Windows), `rdnu_harness`/`rdnu_engine`/`rdnu_pass_harness` (Windows).
- `check_shaders.sh` extended to the new directories and profiles.
- Docs: `build-windows.md` (DLL build, sample hookup), `NSS_Runtime_Plan.md` §6 updated as
  packages land; README status.

---

## 3. Order and dependencies

```
WP1 manifest/export ─┬─► WP3 kernels ─┬─► WP4 engine + core tests ─┐
                     │                │                            ├─► WP6 backend ─► WP7 provider/DLL ─► sample
WP5 pass port + goldens ──────────────┘                            │
WP8 eval tooling (independent) ────────────────────────────────────┘
WP9 build/docs alongside each
```

Sequence for the implementer: WP1 → WP3-A (DP4a) → WP4 (core + harness `--prod`) → WP5 → WP3-B
(WMMA) → WP6 → WP7 → WP8 → WP9. Each package is one PR with its verification output in the
description. Do not start WP6/WP7 before WP4's core tests pass: the backend is thin glue over the
engine and the mapping tests.

## 4. Hand-over checklist (first hour on the 7900 XTX)

1. `python runtime/tools/nss_export.py` and `python runtime/tools/nss_pass_frames.py <Bistro
   sequence> --start 60 --frames 16` for the goldens.
2. `cmake --preset windows`, build Release, `ctest --test-dir build/windows -C Release`: every
   test on the real GPU, WMMA on RDNA3.
3. `rdnu_prod runtime\tools\golden --time 200 --size 1920x1088`, with and without `--force-dp4a`;
   `rdnu_bench` at the sizes in `runtime/tools/eval/bench.md`.
4. `run_engine`: the fp32 reference within 1e-3 and INT8 exact.
5. The FSR sample with the patch, "RDNU (AI)" and the debug view; `low_res_color` must shake while
   `unjittered_color` stays still.
6. A game by DLL swap, captured with `RDNU_COMPARE=4.1`, scored with `compare.py`.
7. Then `NSS_Quality_Plan.md` §6 (LOD bias, jitter length, RCAS, evaluation against FSR 4.1).

## 5. Explicitly out of scope here

Retraining or fine-tuning (NSS-R2/R3 tiers in `NSS_Quality_Plan.md`), NFRU/frame generation,
ray-reconstruction, non-2× ratios beyond passing them through, reactive/T&C masks, and any
measurement.
