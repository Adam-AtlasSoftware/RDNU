# RDNU — WMMA Inference Engine Implementation Plan

Master plan for replacing the current bilinear-fallback placeholder with a **numerically-correct,
WMMA-accelerated INT8 inference** of the RDG UNet, integrated into the Cauldron/FSR sample.
Supersedes the earlier high-level sketch in [RDNU_Implementation_Plan.md](RDNU_Implementation_Plan.md).

> **Strategy (decided):** full hand-written WMMA HLSL, architected as a **reusable, metadata-driven
> inference engine** (durable across model changes / a future transformer) rather than bespoke
> per-block kernels. **Correctness first, WMMA optimization second.**

---

## 0. Where we are today

**Works / exists (keep):**
- DX12 host plumbing: device access, DXC compile path (`-T cs_6_6 -enable-16bit-types`), PSO creation,
  descriptor heap, the FSR-provider entry points (`ffxCreateContextDescRDG` / `Dispatch` / `Destroy`),
  ping-pong hidden-state allocation, and the build/run pipeline (VS Code + CMake).
- Weight assets: INT8 W8A8 `.bin` + metadata (`RDNU_WEIGHTS_W8A8[71]`) and FP16 `.bin` + metadata (131).
- Weights are **pre-padded to multiples of 16 in OHWI** (`export_dims`) — exactly the WMMA tile shape.

**Missing (the actual work):**
- The shaders are **concept sketches**, not the network. The dispatch fires 4 generic passes once each;
  the last one ([rdg_upsample.hlsl](../runtime/shaders/rdg_upsample.hlsl)) bilinearly upscales the input
  and overwrites everything — hence "just shows the downscaled input."
- No metadata-driven execution graph, no per-layer weight indexing, no real conv/attention/FFN/pixel-shuffle,
  no activation (re)quantization, no G-buffer inputs, no numerical validation.

---

## 1. Target engine architecture

A small, reusable inference runtime driven by the weight metadata — **not** one shader per model block.

- **Weight residency:** upload the whole INT8 `.bin` once into a DEFAULT-heap `ByteAddressBuffer`.
  Per-layer INT8 weights, FP16 per-output-channel scales, scalar FP16 activation scale, and FP16 bias
  are all addressed by the byte offsets already in `RDNU_WEIGHTS_W8A8`. WMMA `Load(ByteAddressBuffer,
  offset, stride)` reads weight tiles directly — no repacking at runtime.
- **Tensor (feature-map) manager:** a small pool of intermediate textures/buffers per pyramid level
  (full / ½ / ¼ res, channel counts 36/72/108 padded to 48/80/112…), double-buffered so each layer reads
  one and writes another. Explicit VRAM budget; reused across layers to bound memory.
- **Execution graph:** a C++-side ordered list of layer descriptors `{op type, in/out tensors, weight
  index, dims, stride, scales, activation}` built from the metadata + the fixed RDG topology (§2). The
  dispatcher loops the list, sets root constants (offsets/dims/scales), binds in/out, and dispatches the
  right PSO. One generic dispatch loop, data-driven — this is the reusable core that carries to future models.
- **Reusable primitive kernels (§3):** `wmma_gemm_i8`, `conv_implicit_gemm` (wraps the GEMM for 1×1/3×3),
  `depthwise_conv` (ALU), `shift_conv_1x7/7x1` (ALU), `cross_attention`, `activation+requantize`,
  `pixel_shuffle`, `resample`, elementwise add / skip-merge.

---

## 2. The model, layer by layer (the "nothing forgotten" spine)

Base variant (RDNU-L, 36 base channels). Inputs the network was trained on (per README): **LR color (3),
depth (1), surface normals + albedo/BRDF (6 → the HFB `conv_g` input), optical flow / motion vectors (2)**,
with Halton jitter. Topology, with per-stage channels (padded) and resolution:

```
Input assembly:  LR color(3) → first_conv → 36ch @ full-render-res           [3×3, quantized]
Encoder 0:       enc0.dfm (proj_in dw3×3 + 1×1, mlp_h 1×7, mlp_w 7×1, merge, proj_out) + enc0.ffn (3×3→1×1)
                 → store skip0 (36ch)                                         ↓ down0 (3×3 stride2) → 72ch @ ½
Encoder 1:       enc1.dfm + enc1.ffn → store skip1 (72ch)                     ↓ down1 (3×3 stride2) → 108ch @ ¼
Middle:          middle_enc.dfm + middle_enc.ffn                              (108ch @ ¼, bottleneck)
Middle decoder:  middle_dec.hfb(conv_g[6ch G-buffer], conv_x, proj_out) + middle_dec.ctr(to_q,to_kv,attn,proj_out)
                 + middle_dec.ffn                                             ↑ up0 (3×3) → 72ch @ ½
Decoder 0:       merge skip1×skip_alpha1 → dec0.hfb + dec0.ctr + dec0.ffn     ↑ up1 (3×3) → 36ch @ full
Decoder 1:       merge skip0×skip_alpha0 → dec1.hfb + dec1.ctr + dec1.ffn     (36ch @ full)
Final:           UpSample.0 (3×3 → 12ch) + pixel-shuffle ×2 → RGB @ 2×render-res
Post:            fractional resample to the requested upscale ratio + RCAS sharpening → output
```

**Op inventory (each becomes a validated primitive):**
| Op | Where | Impl |
|----|-------|------|
| 1×1 conv | proj_in.1, mlp_h/w, merge, proj_out, ffn.fn.2, hfb.proj_out, ctr.to_q/kv/proj_out | **WMMA GEMM** |
| 3×3 conv | first_conv, ffn.fn.0, hfb.conv_g/x, downs, ups, UpSample | **WMMA implicit-GEMM** (im2col in LDS) |
| depthwise 3×3 | dfm.proj_in.0, ctr.to_q.1/to_kv.1 | ALU (per-channel) |
| 1×7 / 7×1 shift-conv | dfm.mlp_h/w.shift_weight (non-quantized) | ALU (separable) |
| cross-attention + softmax | ctr blocks (Q from current, K/V from warped history) | ALU + WMMA for the projections |
| GELU + requantize→int8 | after every quantized layer | ALU epilogue |
| pixel-shuffle ×2 | final UpSample | ALU |
| bilinear/strided resample | up/down, fractional output | ALU |
| skip-merge (×alpha) | decoders | ALU |
| CTR temporal warp | ctr (motion-warp prev hidden state) | ALU (sampler) |

---

## 3. The WMMA compute primitive (the hard core, reusable)

INT8 16×16×16 wave-matrix on RDNA3 (Wave32):
- **A** = activations (int8, row-major tile of output pixels × K), **B** = weights (int8, col-major from
  OHWI), **accumulator** = int32. `A.Load` from LDS-staged activations, `B.Load(ByteAddressBuffer, offset,
  stride)` straight from the weight buffer, `acc.MultiplyAccumulate(A, B)`, loop K-tiles, `acc.Store`.
- **Conv as implicit GEMM:** M = tile of output pixels, N = output channels (padded 16), K =
  in_channels·kH·kW (padded 16). im2col gather staged through LDS to cut VRAM bandwidth.
- **Dequant epilogue (exact, matches PyTorch):** `f16 = (float)i32_acc * w_scale[n] * a_scale + bias[n]`.
- **Activation + requantize:** apply GELU, then quantize back to int8 using the *next* layer's `a_scale`
  (static per-tensor activation scales are baked into the metadata — W8**A8**).
- **Padding correctness:** the 16-multiple padded input channels must be zero so they contribute 0 to the
  accumulator; padded output channels are discarded. Verified in the harness.
- Wave32 register-layout + occupancy tuning (VGPR pressure, LDS budget) — the perf pass (§8, Phase 6).

---

## 4. Enabling AMD wave-matrix intrinsics

- Compile with DXC, SM 6.6, `-enable-16bit-types` (already in place) + include
  `AmdExtD3DShaderIntrinsicsMatrixOps.hlsl` (path wired in Phase 0 via an `-I` include dir).
- The AMD shader intrinsics use a driver mechanism that requires the **AMD extension UAV** bound at its
  reserved slot (and/or AGS initialization). Phase 0 wires this and adds a **RDNA3 capability check** at
  context creation; on non-RDNA3 hardware we fall back to the plain FP16 path (also lets the engine run —
  slowly — for validation on other GPUs).

---

## 5. Inputs & the G-buffer gap (Cauldron/FSR integration)

The model needs **color + depth + motion vectors + a 6-channel G-buffer (normals/albedo)**; the sample
currently passes only the first three. Work:
- Fetch Cauldron's existing G-buffer render targets (normal, albedo/diffuse — same source as
  `GBufferMotionVectorRT`) and bind them to the RDG dispatch (extend the dispatch inputs + root signature +
  SRVs). **Match the exact channels/encoding/order the model was trained on** (Sobel-derived normals,
  BRDF/albedo, MV magnitude scaling, Halton jitter, depth linearization) — a mismatch here silently wrecks
  quality even with a numerically-perfect network.
- Confirm the training preprocessing (in `RDG/` / `scripts/`) and reproduce it exactly on the GPU side.

---

## 6. Root signature & binding redesign

The current generic root signature is insufficient. New design (one root sig reused for all layer
dispatches): root constants (per-layer offsets/dims/scales/flags) + the weight `ByteAddressBuffer` + input
feature-map SRV table + output UAV table + G-buffer/input SRVs + the AMD-extension UAV + samplers. Feature
maps ping-pong between two descriptor ranges. Barriers between dependent dispatches.

---

## 7. Numerical validation harness (do this FIRST — it makes everything else mechanical)

- **Golden dumps:** a PyTorch script (extends `runtime/tools/`) that runs the trained model on one fixed
  input frame and dumps every layer's input/output tensor to disk.
- **GPU comparison:** a debug mode in the backend (or a standalone harness) that runs the engine up to layer
  *k*, reads back the tensor, and compares to the golden value (max-abs / relative error) within an INT8
  tolerance band. Drives layer-by-layer bring-up; catches the layer where divergence starts.
- Without this, debugging INT8 numerics on the GPU is guesswork; with it, it's a checklist.

---

## 8. Phased delivery (milestones & exit criteria)

| Phase | Deliverable | Exit criteria |
|------|-------------|---------------|
| **0. Foundations** | Metadata-driven engine skeleton; whole-`.bin` weight buffer; new root sig; WMMA intrinsics enabled + RDNA3 check; validation harness + golden dumps; **`first_conv` runs for real** replacing the bilinear fallback | `first_conv` GPU output matches PyTorch golden within tolerance |
| **1. Core primitives** | `wmma_gemm_i8`, implicit-GEMM conv, dequant+GELU+requantize, depthwise/shift/pixelshuffle (plain first) | 1×1 conv, 3×3 conv, one full FFN block match golden |
| **2. Encoder** | first_conv → enc0 → down0 → enc1 → down1 → middle_enc; skip capture | Encoder feature maps match golden at every stage |
| **3. Decoder (spatial)** | HFB (needs §5 G-buffers), CTR attention (temporal off), FFN, skip-merge, ups, final pixel-shuffle | **Full single-frame forward pass matches PyTorch**; visible SR output (temporal disabled) |
| **4. Temporal** | CTR history warp, ping-pong hidden states, `reset`/camera-cut clearing | Stable multi-frame output; matches PyTorch recurrent reference |
| **5. Real integration** | §5 G-buffer plumbing, jitter/MV/depth preprocessing, fractional resample + RCAS | Correct, stable upscaling selectable in Cauldron on real scenes |
| **6. WMMA optimization** | Swap GEMM-heavy layers to tuned wave-matrix kernels; LDS tiling; fuse safe epilogues; occupancy tuning | Meets per-frame perf budget on RX 7900 XTX |
| **7. Ship-hardening** | Resolution changes, quality/perf validation sweep, edge cases, cleanup | Ready to ship |

Phases 0–3 are the correctness backbone; 6 is where "full WMMA" pays off. Everything in 0–5 is the reusable
engine that carries to a future model/transformer; only the §2 graph and any model-specific fusions in 6
are model-bound.

---

## 9. Feasibility & risk

**Feasible?** Yes — no research unknowns. Every piece (WMMA GEMM, quantized inference, metadata-driven
dispatch, Cauldron G-buffer access) is established engineering. The cost is **breadth and numerical
discipline**, not novelty.

**Effort (collaborative):** Phase 0 is a solid focused chunk. Phases 1–4 (a correct full recurrent forward
pass) are the bulk — weeks of tight iteration. Phase 6 (WMMA perf) is weeks more. Realistically a multi-week
to multi-month project, front-loaded on correctness.

**Top risks & mitigations:**
1. **INT8 numerical divergence** → the §7 harness + layer-by-layer bring-up (single biggest de-risker).
2. **G-buffer/preprocessing mismatch with training** (§5) → reproduce training preprocessing exactly; validate
   against golden dumps that use the *same* inputs.
3. **WMMA intrinsic enablement quirks** (extension UAV, driver, RDNA3-only) → isolate in Phase 0 with the
   FP16 fallback so progress never blocks on it.
4. **Attention (CTR) cost/complexity** → bring up on ALU first, WMMA-ize the projections later.
5. **Fixed-2× network vs fractional FSR ratio** → final resample + RCAS stage (§2 Post) reconciles.

**Division of labor:** I write the engine, shaders, harness, and integration and iterate against your test
output. You are the hands on the RX 7900 XTX — build/run, capture validation + PIX/GPU-crash output, provide
the PyTorch golden dumps, and confirm visuals. The loop per layer is: I write layer + its validation → you
run → we compare to golden → fix → next. That loop is very doable.

---

## 10. Immediate next step (Phase 0, first slice)

1. Reproduce/confirm the model forward pass in PyTorch and add the golden-dump script (`runtime/tools/`).
2. Stand up the metadata-driven engine skeleton + whole-`.bin` weight buffer + new root signature.
3. Wire WMMA intrinsics + RDNA3 check (with FP16 fallback).
4. Implement + validate **`first_conv`** end-to-end, replacing the bilinear fallback with one real,
   golden-verified layer.

When `first_conv` matches PyTorch on your GPU, the pattern for the remaining 70 layers is mechanical.
