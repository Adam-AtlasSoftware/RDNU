> Superseded for the model-specific parts by [NSS_Runtime_Plan.md](NSS_Runtime_Plan.md) (2026-09-24): the network is now Arm NSS. §2–3 (engine architecture, tensor layout, binding model, WMMA conv) remain the engine design.

# RDNU runtime: state, findings, and the plan to a shippable upscaler

Supersedes the phase table in [WMMA_Implementation_Plan.md](WMMA_Implementation_Plan.md).
Everything below was derived from the code on `main` at `0c710ee`.

## 0. Where things stand

| Piece | State |
|---|---|
| `runtime/harness/shaders/*` (16 kernels) + `rdnu_engine.cpp` | Whole single-frame network on GPU. FP32 graph matches PyTorch (2.5e-5). INT8 W8A8 graph bit-exact vs the torch int8 reference. Frame 0 only (no temporal path). |
| `wmma_gemm.hlsl`, `wmma_conv1x1.hlsl` | WMMA INT8 works on the 7900 XTX. 1×1 conv = 3 passes (quant → GEMM → dequant), matches `conv2d_int8`. |
| `runtime/shaders/rdg_*.hlsl` + `rdg_dx12_backend.cpp` | Placeholders. Conv is a copy, CTR is a warp+add, upsample is bilinear. Not on the path forward. |
| `ffx_provider_rdg.h` | Hidden-state sizes are for RDG‑s (48/32/16 ch). The exported model is Base: 108/72/36 ch. |
| `dump_golden.py` | Random 64×64 inputs, 2 frames. Emits frame‑0 bundles only. |
| Dataset / preprocessing code | **Not in the repo.** `RDG/basicsr/data/` and `CombinedRecurrentDataset` are missing. Only `render_data_util.py` (upstream EXR loader) survives. |
| INT8 calibration script | Not in the repo. The W8A8 `.bin` cannot be regenerated. |

## 1. Findings that change the plan

### 1.1 The Base model costs ~293K MAC per render pixel

Per-pixel MACs of the exported network (dim 36, enc/dec [1,1], middle 1), counted from the metadata:

| Level | Resolution | MAC / level pixel | MAC / render pixel |
|---|---|---|---|
| 0 (36 ch) | full | 128K | 128K |
| 1 (72 ch) | ½ | 454K | 114K |
| 2 (108 ch) | ¼ | 819K | 51K |
| **total** | | | **~293K** |

The 3×3 convs (`ffn.fn.0`, `hfb.conv_x`, `ups.*`, `downs.*`) are ~75% of it.

| Render size | MAC / frame | Ops / frame |
|---|---|---|
| 1280×720 | 0.27 T | 0.54 T |
| 1920×1080 | 0.61 T | 1.21 T |
| 2560×1440 | 1.08 T | 2.16 T |

RDNA3 INT8 WMMA peak on the 7900 XTX is 123–246 TOPS depending on which AMD figure applies
to `V_WMMA_I32_16X16X16_IU8` (measure it: extend `rdnu_wmma_probe` into a throughput loop,
that is a one-hour task). At a realistic 40–50% of peak, 1080p render costs **10–25 ms of matrix
work per frame**, before memory traffic. That is a 60 fps budget, not a DLSS/FSR4 budget (1–3 ms).

Memory traffic is the same order: level‑0 tensors are 36–144 ch × 2.07 M px. In FP32 (what the
harness does today) the ~40 level‑0 tensor passes are ~25 GB/frame → ~30 ms at 800 GB/s. INT8
activations plus fused epilogues bring that to ~5 GB.

**Decision needed before Phase C (perf):** ship Base at 1080p‑render/60 fps, or switch to
RDNU‑S (16 ch ≈ 60K MAC/px, ~5× cheaper), or both behind a quality toggle. Nothing in Phases
A–B depends on this; the engine is model-agnostic.

### 1.2 The input conventions are unrecoverable from the repo

The runtime must reproduce the training preprocessing exactly. What is pinned today:

| Input | Pinned from | Convention |
|---|---|---|
| Motion | `CTR.forward`, `TimeLoss`, `motion_warp` | Sampled on the **current** frame's grid, in **render-res pixels**, `+y down`; `prev = sample(pre_h, p + mv[p])`. This is the FSR convention (`mv * motionVectorScale`), sign included. Flow is bilinearly resized to ½ and ¼ **without magnitude rescale** (`F.interpolate` on the values) — reproduce that, do not "fix" it. |
| Warp | `motion_warp` | `grid_sample(bilinear, padding=zeros, align_corners=True)`: pixel centres at integer coords, taps outside `[0,W-1]×[0,H-1]` read 0. |
| Padding | `check_image_size` | Render size padded right/bottom with zeros to a multiple of 4; output cropped to `2H×2W`. |
| Normal (upstream EXR path) | `load_exr` | `(n + 1) / 2` per channel. |
| Depth (upstream EXR path) | `load_exr` | Per-frame min–max normalised to `[0,1]`. |
| BRDF (upstream EXR path) | `load_exr` | Per-frame min–max normalised. |
| G-buffer order | `rdg_arch.forward` | `cat([normal(3), brdf(3)])` → 6 ch. |

What is **not** pinned, because the loader that trained the model is missing: colour space of
`Image` (sRGB `[0,1]` from PNG is the likely case), whether Depth was min–max per frame or raw
16‑bit / 65535, how the Sobel-from-depth normals were encoded, what "BRDF" actually was for
vKITTI/GameIR/M3VIR (class-segmentation colours vs real albedo), and the flow magnitude
scaling. **Action: commit `RDG/basicsr/data/*.py` from the training machine.** Until then,
Phase B's ingest kernel is a guess and quality cannot be judged.

Two consequences worth stating now:

- Training normals were derived from depth (Sobel). The runtime should derive them from the
  depth buffer the same way, not read the engine's normal RT. Same distribution, one small pass.
- The sample feeds FSR **HDR, pre-tonemap** colour. The model saw LDR. The ingest kernel needs a
  reversible tonemap + encode (`x/(1+x)` then sRGB, or whatever matches the loader), and the
  egress kernel the inverse.

### 1.3 Hidden-state semantics

- `pre_hs[k]` is the **DecodeLayer output** of the previous frame at each level (`cur_hs`):
  `mid_d` 108 ch @ ¼, `dec0` 72 ch @ ½, `dec1` 36 ch @ full. Not the CTR input, not padded
  channel counts from RDG‑s.
- Frame 0 / `reset`: `pre_h = x` (the current `d1`) and `flow = 0`. **Clearing the hidden state
  to zero is wrong**; the reset variant of the graph feeds `to_kv` from `d1` directly. Keep two
  op lists (frame‑0 and steady state) and pick per frame.
- Inside CTR, `x` is `d1` (after the HFB residual); `k, v = to_kv(warp(pre_h, flow))`.
- `ctr.alpha` is never used in `forward`. Ignore the three entries.
- Hidden states can be stored FP16: their only consumer is a quantized conv (`to_kv.0`), whose
  INT8 step dwarfs FP16 rounding.

### 1.4 CTR attention is channel attention

`attn = normalize(q1, over pixels) @ normalize(k, over pixels)ᵀ` is `(head, Cq, Ck)` with
`Cq = dim/4`, `Ck = 2·dim/4` (9×18, 18×36, 27×54). Cost is linear in pixels. What does **not**
scale is the current implementation: `l2norm_spatial` loops one thread over all pixels of a
channel, and `ctr_scores` loops one thread over all pixels per `(i, j)`. At 2 M pixels those are
serial loops of 2 M iterations. Both become parallel reductions (§3, K5). `apply` is a
per-pixel `(Cq×Ck)` matvec per head — fine as is, and equivalent to a block-diagonal 1×1 conv with
per-frame FP weights.

### 1.5 Where quantization happens

The int8 reference quantizes the **input of each conv with that conv's `a_scale`**. Tensors with
several quantized consumers use different scales per consumer (`x` → `proj_in.0` and `merge`;
`d1` → `to_q.0` and `to_kv.0` on frame 0). Rule for the engine:

- The master copy of every activation is FP16 (FP32 in validation builds).
- INT8 copies belong to the **consumer**: quantize in the conv's prologue with its own scale.
- Fuse "requantize with the next scale" into the producer's epilogue only when the producer's
  output has exactly one consumer and it is a quantized conv (the common case: `fn.0 → GELU → fn.2`,
  `proj_in.0 → proj_in.1`, `to_q.0 → to_q.1`, ...).
- FP16 storage changes low bits before requantization, so INT8 "bit-exact" is only achievable in
  the FP32 build. Shipping build gate is output PSNR vs golden (§5), not bit equality.

### 1.6 Stride-2 and shift convs

- `downs.*` are 3.5% of MACs. Keep them on `conv2d_int8` (ALU) initially. If needed later,
  space-to-depth the input (4·Cin channels at ½ res) and run them as stride‑1 2×2 WMMA convs.
- `mlp_h`/`mlp_w` shifts are per-channel pixel offsets `dx(i) = (i+3) mod 7 − 3` (resp. `dy`).
  Fold them into the quantize prologue of the following 1×1 conv: `Q[ic][p] = q(x[ic][p + shift(ic)])`
  with zero outside the image. Cost: nothing.

## 2. Target architecture

One engine, two hosts. `runtime/src/rdnu_engine.{h,cpp}` is a library; `rdnu_engine.cpp` in the
harness becomes a `main()` that loads a bundle and calls it; `rdg_dx12_backend.cpp` becomes a
thin FFX provider that calls it with Cauldron's resources. The placeholder shaders and the
descriptor-table root signature are deleted.

### 2.1 Tensor layout

Every activation is a buffer, channel-major, **spatially padded**:

```
elem(c, y, x) = c * Hp * Wp + (y + 1) * Wp + (x + 1)
Wp = roundup(W + 2, 16)   Hp = H + 2       W, H = level resolution (render size padded to ×4, then /1, /2, /4)
```

The 1‑pixel zero border makes every 3×3 tap a constant offset (`±Wp`, `±1`) and makes zero
padding implicit. `Wp` a multiple of 16 keeps a 16‑pixel WMMA tile inside one row. Border and pad
columns must be **written as zero by every producer** (the epilogue knows `x ≥ W`, `y ≥ H`).

Three storage types share this indexing: FP16 master (`RWBuffer<half>` / `RWByteAddressBuffer`),
FP32 (validation build, compile-time switch), INT8 (`RWByteAddressBuffer`, one byte per element;
16 consecutive pixels of one channel = 16 bytes = one WMMA row).

Channel counts are padded to 16 in the INT8 copies only (`Cin_pad`), with the pad channels zero.

### 2.2 Binding model

Copy the harness scheme, not the backend's: root constants (16 DWORDs) + root SRVs + root UAVs +
the AGS mailbox UAV at `u0, space 2147420894`. No descriptor heap for the engine. Textures appear
only in the ingest kernel (input SRVs) and the egress kernel (output UAV); those two use a small
descriptor table.

### 2.3 Memory

All buffers are allocated once per context for `maxRenderSize`. Per-frame `renderSize` goes in
root constants (dynamic resolution). A linear-scan liveness pass over the op list assigns
intermediates to a pool of slots; the whole-model graph needs ~6 live level‑0 tensors at peak.
Hidden states: 2 × 3 buffers, swapped per frame.

### 2.4 Graph

Keep the `Op` list from `rdnu_engine.cpp` (it is already the metadata-driven core). Add op kinds
`warp`, `quant` (with optional shift), `wmma_conv` (1×1 / 3×3, epilogue flags), `l2norm2`,
`scores_splitk`, `ingest`, `egress`. Add a `frame0` bool to the graph builder that selects the
`to_kv` input. Weight offsets come straight from `RDNU_WEIGHTS_W8A8` (upload the whole `.bin`
once as a `ByteAddressBuffer`; no repacking, OHWI padded is already the tile layout).

## 3. Kernel specifications

Each kernel lists the exact math, layout, dispatch and its validation gate. Tolerances: FP32 build
1e‑3 abs on block outputs (as today); INT8 convs bit-exact in the FP32 build.

### K1 `wmma_conv.hlsl` — INT8 1×1 and 3×3 conv, one kernel

```
C[oc, p] = Σ_tap Σ_ic W[oc, tap, ic] · Q[ic, p + off(tap)]        int32 accumulate
out[oc, p] = C · w_scale[oc] · a_scale + bias[oc]                  fp32
```

- One Wave32 per `16 oc × 16 px` tile (start there; then 4 accumulators per wave, `16 oc × 64 px`,
  reusing each weight tile 4× — the main occupancy/bandwidth win).
- Weight tile for tap `t = ky*3+kx`, oc tile `ot`, ic tile `kt`:
  `a.Load(Weights, w_off + (ot*16)*(9*Cin_pad) + t*Cin_pad + kt*16, 9*Cin_pad, false)`.
  For 1×1, `9 → 1`. Stride is the OHWI row length, so the exported `.bin` is used as is.
- Activation tile: `b.Load(QAct, (kt*16)*Hp*Wp + (row*Wp + col0) + off(t), Hp*Wp, false)`,
  `off(t) = (ky−1)*Wp + (kx−1)`. The spatial border makes this correct at image edges without
  branches.
- K loop: `for t in taps: for kt in Cin_pad/16: acc = WMMA(a, b, acc)`.
- Epilogue (per lane, after `acc.Store` to an int32 scratch or directly from the accumulator
  fragment if the AGS header exposes element access): dequant → `+bias` → optional GELU (erf,
  existing A&S 7.1.26) → optional residual add (`+R[oc,p]`) → optional gate (`×G[oc,p]`) → write FP16
  master; optional requantize with `next_a_scale` → write INT8 for the single consumer. Zero
  when `x ≥ W || y ≥ H`.
- Root constants: `w_off, s_off, b_off, a_scale_bits, next_a_scale_bits, Cin_pad, Cout, Cout_pad, W, H, Wp, Hp, taps, flags`.
- Dispatch: `(Wp/16, H_tiles, Cout_pad/16)`; rows can be batched per group.
- Gate: `int8_ffn_3x3`, `int8_ffn_1x1`, `int8_first_conv` bundles bit-exact vs `conv2d_int8`
  (already validated). Then `full_model_int8` with every 1×1/3×3 on K1.

### K2 `quant.hlsl` — FP→INT8 prologue, optional shift

```
Q[ic, p] = clamp(floor(x[ic, p + shift(ic)] / a_scale + 0.5), -127, 127)     ic < Cin
Q[ic, p] = 0                                                                  Cin ≤ ic < Cin_pad, or p in border
```

`shift(ic)` from the CycleFC formula (`mlp_h`: `dx = (ic+3) mod 7 − 3`, `mlp_w`: `dy`), zero
source when the shifted tap leaves the image. Only emitted when the producer could not fuse
requantization (§1.5). Gate: `cyclefc_shift_*` bundles, then the DFM block bundle.

### K3 `dwconv3x3_int8.hlsl` — depthwise, ALU

Existing `conv2d_int8` with `Groups = Cin`, rewritten on the padded layout (no bounds branches),
one thread per `(c, p)`, fused dequant/bias/requant epilogue like K1. Covers `proj_in.0`,
`to_q.1`, `to_kv.1`. Gate: `int8_dfm_dw3x3`.

### K4 `warp.hlsl` — motion warp of a hidden state

```
flow_l = resize_bilinear_align_false(flow_full → level size)          (values unchanged)
sx = x + flow_l.x[p], sy = y + flow_l.y[p]
out[c, p] = Σ over the 4 integer neighbours of (sx, sy), weights bilinear, tap = 0 if outside [0,W-1]×[0,H-1]
```

Manual 4‑tap gather from the FP16 hidden-state buffer (not a hardware sampler — buffers, and
exact vs `grid_sample`). Output goes straight into K2 for `to_kv.0` (single consumer), so K4 can
write INT8 directly with `to_kv.0.a_scale`. Gate: new `ctr_block_f1.rdnut` (§5).

### K5 CTR reductions at scale

- `l2norm2`: pass 1 — per `(c, group of 4096 px)` partial `Σx²` via wave reduction into a
  `[C × groups]` buffer; pass 2 — per channel sum, `inv = 1/max(sqrt(s), 1e-12)`; pass 3 — scale
  (fuse into `scores` loads instead of materialising `q1n`, `kn`: `scores` multiplies by
  `inv_q[i]·inv_k[j]` at the end).
- `scores`: `attn[hd,i,j] = inv_q·inv_k · Σ_p q1[hd,i,p]·k[hd,j,p]`, split‑K over pixel groups,
  partials `[head·Cq·Ck × groups]`, then a tiny reduce. All FP32.
- `softmax`: unchanged (rows of ≤ 54).
- `apply`: unchanged math, one thread per pixel computing all `head·Cq` outputs; fuse the
  `concat(att, q2, depth)` by writing into the channel ranges of one `(2·dim+1)`‑channel buffer
  that `proj_out`'s K2 quantizes.
- Gate: `ctr_block` (frame 0) and `ctr_block_f1` bundles at 64×64; then a 1920×1088 self-check
  against a CPU reference of `l2norm`/`scores` only (no golden needed).

### K6 Elementwise, resize, pixel-shuffle

`add`, `axpy`, `mul`, `gelu`, `concat` disappear into K1 epilogues where §1.5 allows; keep the
standalone kernels for the cases that remain (`x + res` before the head, the two `axpy` skip
merges — those can be the residual input of the next K1). `resize` (existing, `align_corners=False`)
serves `ups.*`, and the g/depth/flow downscales. `pixelshuffle` fuses into the head conv's
epilogue: `out[c, 2y+dy, 2x+dx] = conv[(c*2+dy)*2+dx, y, x]`.

### K7 `ingest.hlsl` / `egress.hlsl` — the Cauldron boundary

Ingest, one pass, writes the level‑0 inputs in the padded layout:

- colour: `desc->color` → exposure → reversible tonemap → encode per §1.2 → FP16 (3 ch).
- depth: `desc->depth` → linearise with `cameraNear/Far` if the loader used linear depth →
  normalise per §1.2. Per-frame min–max needs a reduction pass before ingest (K5 machinery).
- normal: Sobel on the normalised depth exactly as the loader did → `(n+1)/2`.
- brdf: `albedo` RT (extension input) → per §1.2.
- motion: `desc->motionVectors * motionVectorScale` → render-pixel offsets, current→previous.
- jitter: the model was trained on jittered LR frames; pass colour through unmodified and record
  `jitterOffset` for the egress resample (the ×2 output is jittered by `2·jitter`).

Egress: `PixelShuffle` output (`2H × 2W`, cropped) → inverse encode/tonemap → resample to
`upscaleSize` (bicubic, antialiased when downscaling; ratios > 2 are out of scope) → CAS as in
`VideoRenderModel.test` (`amount 0.7`) or Cauldron's RCAS → `desc->output`.

G-buffer extension: add `struct ffxDispatchDescRDGGBuffers { ffxApiHeader header; FfxApiResource normal, albedo; }`
chained via `header.pNext`; the fork's `fsrapirendermodule.cpp` supplies the GBuffer RTs in
`NON_PIXEL_SHADER_RESOURCE` state.

## 4. Phases

Phases A and B are correctness and run in the harness first; C is performance. Each step is
one commit with its validation output in the message, as the Phase 1 history does.

### Phase A — steady-state temporal network in the harness

A1. `dump_golden.py`: `--frames 2` real-input mode (a Sintel/vKITTI pair, or the 64×64 random
    inputs at minimum) emitting `ctr_block_f1.rdnut` (inputs `d1`, `pre_h`, `flow`, `depth`),
    `decode_layer_f1.rdnut`, and `full_model_f1.rdnut` with `pre_hs[0..2]`, `flow`, frame‑1
    output and checkpoints; `--res 1920x1088` for a shape/perf run without golden.
A2. `warp` op + K4, frame‑1 CTR bundle passes (FP32).
A3. Whole model frame 1 passes: `full_model_f1` (FP32, then INT8 graph).
A4. Hidden-state buffers + two graph variants + `reset` semantics in the engine; run frames
    0,1 back-to-back on the GPU without CPU round trips; frame‑1 output unchanged.
Exit: multi-frame recurrent output matches PyTorch (2.5e‑5 FP32; INT8 graph bit-exact).

### Phase B — engine library, padded layout, WMMA path, provider

B1. Move the engine to `runtime/src/rdnu_engine.{h,cpp}`; harness `main()` links it. Introduce
    the padded layout and arena allocation. All bundles still pass (FP32).
B2. K1 (1×1 first, then 3×3) + K2 + K3 with the fused epilogues; INT8 graph bit-exact in the
    FP32 build. Replace the plain int8 convs in the graph builder.
B3. K5 parallel reductions; 1920×1088 self-check.
B4. FP16 master storage behind the build switch; output PSNR vs golden ≥ 50 dB on the 64×64
    random inputs (set the real threshold from the first measurement, then freeze it).
B5. K7 ingest/egress + the FFX extension struct + `ffxDispatchDescRDG` recording the graph on
    the sample's command list. Delete the placeholder shaders and `rdg_upsample`. Visual output
    in the sample, temporal on, `reset` on camera cut.
B6. Input-convention closure: once the loader code is committed, make ingest match it and
    dump a golden from real engine captures (write the sample's ingest outputs to disk, run
    them through PyTorch, compare the sample's output).
Exit: correct, stable upscaling in Cauldron; sample output matches PyTorch on captured inputs.

### Phase C — performance

C1. WMMA throughput microbenchmark; set the real per-frame budget from §1.1 and the model
    decision.
C2. K1: 4 accumulators per wave, weight tiles kept in registers across pixel tiles, rows
    batched per group; measure VGPR count and occupancy with RGP.
C3. Fusion audit: every K2 that remains, every standalone elementwise pass — remove or justify.
C4. Space-to-depth for `downs.*` if they show up in the profile.
C5. Stop when the frame is within budget; then resolution changes, dynamic resolution, and the
    quality sweep.

## 5. Validation additions

- `full_model_f1.rdnut` (frame 1 with real hidden states) is the new "whole model" gate; frame 0
  stays as the reset-path gate.
- `--res WxH` non-square, non-multiple-of-4 sizes (e.g. 70×46) to exercise `check_image_size`
  padding and the padded layout's border handling.
- FP16 build gate: PSNR of the final ×2 RGB vs the FP32 golden, per bundle, threshold frozen
  after the first B4 measurement.
- Sample gate (B6): PSNR of sample output vs PyTorch on the same captured inputs.
- A CPU reference for `l2norm`/`scores` at full resolution (numpy, seconds) for K5.

## 6. Questions only you can answer

1. Performance target: 60 fps at 1080p→4K with Base, or RDNU‑S, or both (§1.1)?
2. Where is `RDG/basicsr/data/` (and the INT8 export/calibration script)? Both need to be
   committed; the input conventions cannot be finished without the loader (§1.2).
3. Colour: was `Image` fed as sRGB `[0,1]`, and was training data tonemapped? Decides the
   ingest/egress transfer functions.
4. Which fractional ratios must work? 2× (Performance) is native; 1.5× (Quality) needs the
   egress downscale; > 2× is out of scope.
