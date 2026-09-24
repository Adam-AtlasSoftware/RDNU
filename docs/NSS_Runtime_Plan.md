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

RDNU keeps the component and the API and supplies what is missing:

```
game / FSR sample / OptiScaler
        │  FFX API (ffx_nss.h): colour, depth, MVs, jitter, exposure, camera, reset
        ▼
ffx_provider_nss  ──►  ffx_nss.cpp (pass scheduling, constants, resources)   [Arm, reused]
        │
        ▼
DX12 backend (FidelityFX 1.1.3 ffx_dx12.cpp from the SDK fork)                [reused]
   ├─ DepthScatter, DisocclusionMaskLQ, Preprocess, GenerateOffsetLut,
   │  Postprocess, DebugView ............ HLSL ports of the GLSL passes        [RDNU]
   └─ DataGraph ........................ RDNU INT8 conv engine (14 convs)     [RDNU]
```

The data-graph job becomes a sequence of compute dispatches on the engine's buffers; the
pre-process pass writes the 12-channel input tensor in the engine's layout and the post-process
pass reads the two output tensors. Everything else (history ping-pong, feedback tensor, luma
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

- **FSR sample:** the provider registers `FFX_API_EFFECT_ID_NSS`; the sample's upscaler module
  gets an "RDNU" entry that fills `ffxApiDispatchDescNss` from the same inputs it gives FSR.
- **Games:** ship the provider as an FSR-3.1-API-compatible upscaler DLL. OptiScaler already
  bridges DLSS/XeSS/FSR inputs (colour, depth, MVs, jitter, exposure, reset, camera) to FSR-API
  providers, which is the whole `ffxApiDispatchDescNss` input set; no G-buffers are needed.
- Inputs NSS does not take: reactive/transparency masks (documented limitation).

## 6. Work list

Done in this repo:
- Submodules: model gym, capture plugin, NSS/NFRU weights, Arm SDK, datasets (`Datasets.md`).
- `runtime/tools/nss_export.py`: fp32 + INT8 weight bundles with per-layer goldens; counts verified.
- `programNSS` / `programNSSInt8` in the harness engine; `relu`, `sigmoid`, `upsample_nearest`
  kernels; zero-point support in `conv2d_int8.hlsl`; `run_engine` targets for both bundles.

Next, in order:
1. **GPU validation** (RX 7900 XTX): `nss_export.py` then `run_engine`; fp32 within 1e-3,
   INT8 bit-exact. Fix anything the checkpoints (`chk.conv2d_*`, `chk.kpn`) localise.
2. **Pre/post goldens:** extend the exporter to run `preprocess_torch` / `postprocess_torch` on
   a few frames of the Bistro test split and dump their inputs/outputs; these are the goldens for
   the HLSL pass ports.
3. **Pass ports:** GLSL shared headers → HLSL (`ffx_nss_*`), validated per pass against 2.
4. **DX12 backend for the Arm component:** compile `ffx_nss.cpp` + `ffx_provider_nss.cpp`
   against the FidelityFX 1.1.3 DX12 backend in the SDK fork; implement the DataGraph job as
   engine dispatches; the interface version mismatch, if any, is the first thing to check.
5. **Sample integration and first pictures:** RDNU selectable in the FSR sample; temporal on;
   `reset` on camera cut; RCAS after.
6. **Evaluation vs FSR 4.1** on captured sequences (FLIP, LPIPS, tPSNR, CGVQM, crops, RGP
   timings); then decide on retraining per `NSS_Quality_Plan.md`.
7. **Performance:** move the 14 convs onto the padded-layout WMMA path (RDNA3) and a DP4a path
   (RDNA2) from `RDNU_Runtime_Plan.md` §3; fuse ReLU/requantise into the conv epilogue.
