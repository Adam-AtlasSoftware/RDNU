# NSS runtime plan: an injectable DX12 upscaler built on Arm NSS

Supersedes the RDG-specific parts of [RDNU_Runtime_Plan.md](RDNU_Runtime_Plan.md); its engine
design (§2–3 there: padded tensor layout, root-descriptor binding, arena, WMMA 3×3 conv) still
applies and is reused unchanged.

## 1. Architecture

Arm ships NSS as a FidelityFX-style component in `runtime/external/neural-graphics-sdk-for-game-engines`
(MIT, derived from FidelityFX SDK 1.1.3): `sdk/src/components/nss/ffx_nss.cpp` schedules seven
passes through the backend interface, `ffx-api/src/ffx_provider_nss.cpp` exposes them through the
FFX API (`ffxApiCreateContextDescNss`, `ffxApiDispatchDescNss`). The SDK only has a Vulkan backend
and runs the network as a `VK_ARM_data_graph` (VGF) job.

RDNU keeps the component and supplies what is missing:

```
game / FSR sample / OptiScaler
        │  FidelityFX API, FSR 3.1 upscale descriptors
        ▼
rdnu_ffx_api.cpp: FSR-to-NSS mapping, exposure, motion vector   [RDNU, src/provider]
        │         and sharpening passes, AMD's DLL for the rest
        ▼
ffx_nss.cpp (pass scheduling, constants, resources)              [Arm, reused unchanged]
        │
        ▼
ffx_nss_dx12.cpp: FidelityFX backend interface on D3D12          [RDNU, src/backend_dx12]
   ├─ DepthScatter, DisocclusionMaskLQ, Preprocess, GenerateOffsetLut,
   │  Postprocess, DebugView ..... HLSL ports of Arm's GLSL        [RDNU, shaders/nss]
   └─ DataGraph ................. INT8 conv engine (14 convs)     [RDNU, src/engine, shaders/net]
```

The data-graph job becomes a sequence of compute dispatches over one arena shared by every
context; the pre-process pass writes the 12-channel input tensor and the post-process pass reads
the KPN and feedback tensors. Everything else (history ping-pong, feedback tensor, luma
derivative, depth tm1, constants, jitter phase queries) is the component's existing code.

## 2. Passes and resources (from `ffx_nss_private.h` and the pre-process bindings)

| Pass | Reads | Writes |
|---|---|---|
| DepthScatter | depth, MVs | nearest-depth coordinates (dilated MVs) |
| DisocclusionMaskLQ | depth, depth tm1, MVs, camera | disocclusion mask (LQ) |
| Preprocess | jittered colour, depth, MVs, history colour, feedback tensor, depth tm1, luma-derivative tm1, disocclusion mask | **input tensor** (12 ch), luma derivative, nearest-depth coord tm1, depth tm1 |
| DataGraph | input tensor | **kpn tensor** (36 ch at ¼ linear res), **temporal tensor** (4 ch at input res) |
| GenerateOffsetLut | jitter, scale | dynamic tap-offset LUT |
| Postprocess | jittered colour, history colour, kpn tensor, temporal tensor, LUT, disocclusion | output colour, next history, next feedback |
| DebugView | any of the above | debug output (optional) |

Constants: `NssConstants` (jitter, scale, dims, depth-to-view, exposure, KPN dims/scale, LUT
modulo/offset, history-reset flag). Shaders are thin wrappers around shared headers
(`ml/models/nss/scenario/*_shared.h`, 40 KB pre-process, 36 KB post-process, 12 KB depth scatter);
the port is those headers, mechanically, GLSL to HLSL, with the same bindings.

## 3. Network contract

Input tensor, channel order (`torch_preprocess/pipeline.py`): `[0–2]` warped history colour
(Karis tonemapped, exposure applied), `[3–5]` current colour unjittered and tonemapped, `[6]`
motion detector, `[7–10]` feedback tensor from the previous frame (zeroed on disocclusion),
`[11]` instability (luma derivative). Values in `[0, 1]`; dims padded to a multiple of 8.

Backbone (`AutoEncoderV1`, `ml/model-gym/.../model_blocks_v1.py`): 14 `3×3` ConvBlocks with
ReLU, sigmoid on the two heads, nearest 2× upsamples, two skip concats. 148,456 params,
11,304 MAC per input pixel (2,826 per output pixel at 2×). Exact op list and dispatch order are
in `programNSS()` (`runtime/harness/rdnu_engine.cpp`).

Outputs: `kpn` (36 = 6×6 taps per 4×4 input block, sigmoid) and `temporal` (4 ch at input res,
sigmoid: the post-process blend/rectification parameters, fed back as next frame's feedback
tensor; channel roles in `torch_postprocess/pipeline.py`). The VGF graph quantises both with a
fixed scale 1/254, zero-point −127; the runtime consumes the sigmoid outputs in FP16 instead.

## 4. INT8 scheme (implemented)

From the QAT checkpoint (`runtime/tools/nss_export.py` pins the fake-quant modules to layers and
asserts shapes and the model-card scales):

- activations: per-tensor affine, zero-point −128, `q ∈ [−128, 127]`; the input tensor scale is
  the model card's `0.003912`; each conv's input quantiser is the one after the previous ReLU.
- weights: per-output-channel symmetric, `q ∈ [−127, 127]`.
- conv: `acc = Σ (xq − zp)·wq` in int32; `y = acc·s_in·s_w[oc] + b[oc]`; ReLU or sigmoid.

`conv2d_int8.hlsl` takes `ascale = [scale, zp, qmin, qmax]`; RDG's symmetric layers use
`[s, 0, −127, 127]`, NSS uses `[s, −128, −128, 127]`. The exporter's integer-path reference is
the golden (`nss_backbone_int8.rdnut`); the GPU must match it bit-exactly in the FP32 build.

## 5. Injection

- **Games:** the DLL replaces `amd_fidelityfx_dx12.dll` (FSR 3.1) or
  `amd_fidelityfx_upscaler_dx12.dll` (FidelityFX SDK 2, behind AMD's loader), with AMD's DLL kept
  as `<name>_original.dll` for frame generation and the other upscaler versions. RDNU is listed
  first, so it is the default. For SDK 2 the context starts with a provider object laid out as
  AMD's MSVC loader expects, whatever compiler built the DLL.
- **OptiScaler:** its FSR 3.1 path feeds DLSS and XeSS inputs through the same descriptors.
- **FSR sample:** `runtime/integration/*.patch`; "RDNU (AI)" pins the context to RDNU.
- **Conventions:** NSS's jitter is FSR's negated; the provider flips it. Measured on the same
  frames, each upscaler loses about 4 dB with the other's sign. NSS reads no reactive or
  transparency masks. Each render size is its own NSS context, so dynamic resolution restarts the
  history.

## 6. Work list

Done (see `NSS_Implementation_Plan.md` for verification): export and goldens, DP4a and WMMA
kernels, engine, pass ports, D3D12 backend, FidelityFX API DLL, FSR sample patch, captures,
`compare.py`, bench, CMake with Linux and Wine test paths.

Next, on the RX 7900 XTX and RX 6800 XT:
1. `ctest`, then `rdnu_prod --time` and `rdnu_bench` at 1080p, 1440p and 4K
   (`runtime/tools/eval/bench.md`), WMMA against `RDNU_FORCE_DP4A=1`.
2. The FSR sample with RDNU and its debug view; check jitter with the `low_res_color` tile.
3. Games by DLL swap and through OptiScaler; captures with `RDNU_COMPARE` against FSR 4.1.
4. Decide on retraining per `NSS_Quality_Plan.md`.
