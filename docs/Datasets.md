# Datasets and model sources

All sources are git submodules so their provenance and revision are pinned. Large ones are
marked `update = none` in `.gitmodules`: a plain `git submodule update --init --recursive`
skips them, and each is fetched only when asked for. Weights and datasets are Git LFS objects;
install `git-lfs` and pull selectively, never `git lfs pull` a terabyte repo by accident.

```bash
# code + small repos (model gym, capture plugin, NSS weights repo as LFS pointers, Arm SDK)
git submodule update --init --recursive

# NSS weights (15 MB)
git -C ml/models/nss lfs pull

# Arm sample dataset (35 GB; pull one split at a time)
git submodule update --init ml/data/arm-neural-graphics-dataset      # opt-in: update=none
git -C ml/data/arm-neural-graphics-dataset lfs pull --include "nss/test/*"
git -C ml/data/arm-neural-graphics-dataset lfs pull --include "nss/train/*,nss/val/*"

# NFRU weights (551 MB), GameIR (791 GB), M3VIR (1.27 TB): same pattern, only when needed
git submodule update --init ml/models/nfru ml/data/gameir ml/data/m3vir
```

`GIT_LFS_SKIP_SMUDGE=1` in front of any clone or submodule command keeps LFS objects as pointers.

## Models

| Path | Source | Licence | Notes |
|---|---|---|---|
| `ml/models/nss` | https://huggingface.co/Arm/neural-super-sampling | Arm AI Model Community License v1.0 (weights) | v1.0.1: `high` fp32 + INT8-QAT, `mid_low` INT8; GLSL pre/post shaders under `scenario/`; SIGGRAPH 2025 deck. Commercial, sublicensable, no hardware restriction; AUP pass-through, notices, mark modifications, no reverse engineering of the model. |
| `ml/models/nfru` | https://huggingface.co/Arm/neural-frame-rate-upscaling | same | Frame interpolation model; opt-in, for the frame-generation track. |
| `ml/model-gym` | https://github.com/arm/neural-graphics-model-gym | Apache-2.0 | Model definitions, pure-torch pre/post pipelines, train / fine-tune / QAT / export. Single-GPU trainer. |
| `ml/capture-unreal` | https://github.com/arm/neural-graphics-data-capture-for-unreal | Apache-2.0 | UE 5.5 plugin writing the model-gym dataset format (8K GT, Halton-jittered LR colour/depth/MVs, per-frame JSON). |
| `runtime/external/neural-graphics-sdk-for-game-engines` | https://github.com/arm/neural-graphics-sdk-for-game-engines | MIT | Arm's FFX-API NSS/NFRU runtime (Vulkan only), derived from FidelityFX SDK 1.1.3; the reference for the NSS component and shader passes. |

## Datasets

| Path / URL | Content | Size | Licence | Use |
|---|---|---|---|---|
| `ml/data/arm-neural-graphics-dataset` | Bistro (UE): NSS train/val/test as sequence safetensors: `colour_linear`, `ground_truth_linear`, `depth`, `depth_params`, `motion`, `motion_lr`, `exposure`, `jitter`, `render_size`, `zNear`, `zFar`; NFRU split | 35 GB | Arm AI Model Community | reproduction, fine-tune seed, format reference. Arm trained the released NSS on >50K frames; this sample is smaller. |
| `ml/data/gameir` | CARLA/UE, 19,200 2× LR/HR pairs, depth, segmentation | 791 GB | MIT | scene diversity for pre-training; no MVs, no jitter |
| `ml/data/m3vir` | UE5, 80 scenes × 6 views, 540p/1080p/1620p, depth, segmentation, camera | 1.27 TB | MIT | scene diversity; no MVs, no jitter |
| QRISP, https://www.qualcomm.com/developer/software/qualcomm-rasterized-images-dataset | 13 Unity scenes, 8,760 frames, 270p–1080p, colour/depth/MVs, Halton jitter, mip bias | 10 GB | research only, forbids commercial use of trained models | benchmarks and ablations only; never for shipped weights |
| RDG dataset, https://www.modelscope.cn/datasets/Sunny21/RDG_Dataset | 20 Cycles scenes, 480×270 / 1080p, normal/BRDF/depth/MV, no jitter | n/a | CC-BY-NC-4.0 (HF mirror tag) | RDG baseline comparison only |
| STSSNet dataset, https://www.modelscope.cn/datasets/ryanhe312/STSSNet-AAAI2024 | 4 UE scenes, ~6K train + ~1K test frames each, G-buffers, MVs, history | ~40 GB/scene | unstated | STSSNet reproduction, frame-extrapolation supervision |
| Noisebase SampleSet v1, https://balint.io/noisebase/datasets/sampleset_v1.html | 1,024 × 64 frames of 256² path-traced samples with G-buffers and MVs | 1.84 TB | Apache-2.0 / MIT | ray-reconstruction track |
| ORCA scenes, https://developer.nvidia.com/orca | Bistro, Emerald Square, Sun Temple, Zero-Day | n/a | CC BY 4.0 (confirm per asset) | render your own sequences |

## What the primary training set must look like

The model consumes exactly what the capture plugin writes (see
`ml/model-gym/docs/nss/nss_dataset_specification.md`): jittered linear-HDR LR colour, forward-Z
depth, non-jittered UV-space motion vectors at LR, jitter in pixels, FoV, near/far, view-projection,
exposure, camera-cut flag; ground truth is an 8K render box-filtered 4×4. Requirements for
resolved detail, in the captures and in the runtime alike: texture LOD bias of
`-log2(ratio) - ε` and a Halton(2,3) jitter sequence of `8·ratio²` phases. Target 100K+ frames
across diverse scenes before judging quality (see `NSS_Quality_Plan.md`).
