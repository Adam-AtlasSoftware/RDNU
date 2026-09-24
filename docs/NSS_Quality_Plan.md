# From NSS to FSR4-class quality on RDNA3, with an RDNA2 tier

Goal: match FSR 4.1 (INT8, RDNA3) in frame time and resolved detail with an open model, and keep
a tier that runs on RDNA2 (DP4a only). Facts below come from [Model_Candidates.md](Model_Candidates.md);
MAC counts for the proposed networks are computed here and labelled as such.

## 1. The bar

| | FSR 4.1 INT8 on RX 7900 XTX | Arm NSS-high (as released) |
|---|---|---|
| 4K output frame time | 3.07 ms (AMD figure) | 0.95–1.9 ms inferred at 20–40% of INT8 peak; unmeasured |
| Network cost | ~4.5K MAC / output px (6.7K dense-equivalent) | 2.8K MAC / output px at 2× |
| Where the net runs | input resolution, then ½, ¼ of it | **½ input resolution** (first conv is stride 2), then ¼, ⅛ |
| Reconstruction | per-**output**-pixel oriented 3×3 Gaussian (3 params) over jittered LR colour, applied at output res | one explicit 6×6 kernel per **4×4 input block** (36 weights, sigmoid), predicted at ¼ input res |
| History | Catmull-Rom reprojection, depth-dilated MVs, disocclusion, sigmoid blend, 4-ch recurrent state | Catmull-Rom (high only), disocclusion, luma-derivative flicker signal, theta/alpha blend, 4-ch state |
| Post | RCAS sharpening, auto-exposure | none |
| Training data | large multi-title corpus | ~50K frames of one scene (Bistro) |

So the compute headroom on RDNA3 is 1.5–3×, and NSS gives detail away in four places: the net
never sees full-resolution features, the filter field is 64× coarser than FSR4's (one kernel per
16 input pixels = per 64 output pixels at 2×, vs one per output pixel), there is no post-sharpen,
and the model has seen one scene.

## 2. Levers, ranked by expected effect on resolved detail

1. **Engine-side sampling (free, largest).** Negative texture LOD bias of `−log2(ratio) − ε`
   (ε ≈ 0.1–0.3) so textures are rendered at output-res detail; without it no upscaler can recover
   texture frequency. Jitter sequence long enough to cover each output pixel: 8·ratio² phases of
   Halton(2,3) (32 at 2×), matched between the capture plugin, training and runtime. Verify the Arm
   capture plugin applies both; if not, add them before capturing anything.
2. **Data (large).** Diverse captures: 100K+ frames across many scenes, thin geometry, foliage,
   particles, text/UI-like detail, fast motion, camera cuts. Arm states production needs
   "substantially more" than the 50K Bistro frames. GT: 8K render box-filtered 4×4 as in Arm's
   spec; try a Blackman-Harris/Gaussian reconstruction filter for GT instead of the box, because
   the box filter's own blur becomes the model's ceiling.
3. **Per-pixel parametric kernel instead of block-shared 6×6 (large, cheap).** Predict 3
   parameters per *input* pixel (orientation, along/across sigma; FSR4 and PSSR 2.0 both do this),
   upsample the parameter field to output res, apply as a 3×3 tap set over the jittered LR colour.
   Edges get an oriented filter instead of a filter averaged over a 4×4 block; the apply pass reads
   9 taps instead of 36. Kernel stays in input space, so reach scales with the ratio (needed for
   1.5×/3× tiers).
4. **Full-resolution features (medium, the main cost).** Remove the initial stride-2 or add a
   16-ch stem at input res so the kernel/blend head sees per-pixel evidence. This is where the
   RDNA3 headroom goes (§3).
5. **RCAS after the upscale (small, free).** FSR4 ships it; expose sharpness 0–1.
6. **Recurrent state 4 → 8 channels (small).** More sub-pixel history survives across frames;
   cost is a few hundred MAC/px in the stem and the reprojection of 4 more channels.
7. **Capacity at ≤ ½ input res (small–medium).** Widen 32/64 → 48/96 at the coarse levels;
   Sony's data says put capacity there, not at output res. Prefer ConvNeXt/FasterNet-style
   blocks (depthwise 3×3 + pointwise) over plain 3×3 for capacity per MAC.
8. **Loss (small).** Keep L1 + LPIPS + temporal-flicker; add a gradient/FFT term (RDG's FFT loss
   weight ≈ 0.05) to stop L1 from hedging on edges. No adversarial term: it buys crispness at the
   price of temporal shimmer, which is what a 0.7-weight flicker loss exists to prevent.
9. **Recurrent unroll 16 → 32 frames** during fp32 training so accumulation over long histories
   is learned, with the same 30% GT-history seeding.

Do 1, 2, 5 first: they change nothing in the network and set the baseline that architecture
changes are measured against.

## 3. Two tiers

### R2 tier (RDNA2 and any DP4a GPU): NSS-high as released, retrained

2.8K MAC/output px. Inferred 4K time on an RX 6800 XT (83 TOPS DP4a) 1.4–2.8 ms; 1440p 0.6–1.2 ms.
Changes from stock: per-pixel parametric kernel head (lever 3: replaces the 32→36 head with a
32→3 head plus a 1-ch blend, *cheaper* than stock), 8-ch state, RCAS, retrained on the new data.
Cost stays ≈ 2.7K MAC/output px.

### R3 tier (RDNA3 WMMA): "NSS-R3", FSR4-class layout, ~5.9K MAC/output px

Computed per input pixel (N = input pixel count, 2× scale, 3×3 unless stated):

| Stage | Layers | MAC / input px |
|---|---|---|
| Stem @N | 12+4 ch in → 16 | 1,728 |
| @N | 2 × ConvNeXt-16 (dw3×3 16: 144; pw 16→32: 512; pw 32→16: 512) | 2,336 |
| down @N/4 | 2×2 s2 16→32 | 512 |
| @N/4 | 2 × FasterNet-32 (3×3 on 16 of 32: 576; pw 32→64: 512; pw 64→32: 512) | 3,200 |
| down @N/16 | 2×2 s2 32→64 | 512 |
| @N/16 | 3 × FasterNet-64 (3×3 16→32: 288; pw 64→128: 512; pw 128→64: 512) | 3,936 |
| up @N/4 | 2×2 convT 64→32 + skip, 2 × FasterNet-32 | 2,048 + 3,200 |
| up @N | 2×2 convT 32→16 + skip, 2 × ConvNeXt-16 | 2,048 + 2,336 |
| head @N | 3×3 16 → 13 (3 kernel, 1 blend, 1 theta, 8 state) | 1,872 |
| **total** | | **23,728 → 5.9K MAC / output px** |

Inferred 4K time on a 7900 XTX at 20–40% of 123 TOPS: 2.0–4.0 ms; FSR4's own 3.07 ms sits at
~26%. At 1440p: 0.9–1.8 ms. On RDNA2 this tier is 1440p-only (1.3–2.7 ms on a 6800 XT).

Both tiers share the pre/post passes (reprojection, disocclusion, flicker signal, kernel apply,
blend, RCAS) and the training code; only the backbone config differs. Same input tensor, same
outputs, so a single INT8 HLSL runtime serves both.

Design provenance: the layout (input-res U-Net, oriented-Gaussian kernel in input space,
explicit blend, small recurrent state) is what Arm's paper, Sony's SIGGRAPH 2026 PSSR 2.0 talk and
AMD's public FSR4 description all converge on. Design from those public descriptions; do not read
or reuse the leaked FSR4 source.

## 4. Training changes (Model Gym, fp32 then QAT)

- Same recipe as stock (AdamW 1e-3 cosine, batch 8, 256-px crops, 15 + 2 QAT epochs) with the
  unroll at 32 and the FFT/gradient term added. Fits one 24 GB card; run the four cards as four
  experiments (R2 vs R3, kernel variants, loss variants, scale factors).
- Mixed scale factors in one model (1.5×, 2×, 2.5×, 3×) once 2× is solid; the input-space kernel
  makes this a data change, not a redesign.
- Self-distillation: train an fp32 R3 twice the width first, use it as a teacher for the INT8
  R3/R2 (feature + output loss). Typical gain is a few tenths of a dB at no runtime cost.
- QAT stays a 2-epoch tail; sigmoid/tanh heads keep outputs in [0,1] / [−1,1], so INT8 is safe.

## 5. Measure against FSR4, not against PSNR

- Capture identical input sequences (colour, depth, MVs, jitter, exposure) from the FidelityFX
  sample; run FSR 4.1 (driver/SDK DLL) and RDNU on them; keep FSR4 output as a **benchmark only**,
  never as a training target (SDK licence, and it would cap quality at FSR4).
- Metrics: FLIP, LPIPS, tPSNR (temporal), CGVQM; plus fixed crops for eyes (thin wires, foliage,
  text, specular highlights under motion, disocclusion edges). PSNR rewards blur; report it but
  don't optimise for it.
- Timing: RGP on a 6800 XT and a 7900 XTX at 1080p/1440p/4K, network and pre/post separately.

## 6. Order of work

1. Baseline: stock NSS-high in the HLSL runtime (the week-one experiment in Model_Candidates.md §9),
   RCAS added, measured against FSR4 on the same sequences.
2. Engine-side LOD bias + jitter length verified in capture and runtime; recapture; retrain stock.
   This alone should close a visible part of the texture-detail gap.
3. Kernel head swap (lever 3) on the stock backbone; retrain; compare edges.
4. R3 backbone; retrain; compare; then QAT and the INT8 WMMA path.
5. Data scale-up to 100K+ frames while 3–4 iterate; every later run trains on the larger set.
6. 8-ch state, unroll 32, distillation, mixed ratios.

Stop criterion: R3 within FLIP/LPIPS noise of FSR 4.1 on held-out sequences at ≤ 3 ms 4K on a
7900 XTX; R2 within reach of that at 1440p on a 6800 XT.
