# Retraining NSS for RDNU

The runtime takes any model the model gym produces from the NSS-v1 architecture: `retrain.py`
fine-tunes, quantises, exports and installs it. Architecture changes (the kernel head and the R3
backbone in `docs/NSS_Quality_Plan.md`) are model-gym work first; this pipeline then carries them
the same way once `runtime/tools/nss_export.py` knows the new layers.

## Setup

A CUDA GPU with 24 GB (batch 8 at 256 px crops, unroll 16). Python 3.10 to 3.12.

```bash
python -m venv .venv && . .venv/bin/activate
pip install ml/model-gym                         # pins torch 2.11 with CUDA
git -C ml/models/nss lfs pull                    # the released weights, the starting point
```

## Data

The model learns exactly what is captured. Before capturing anything:

- **Texture LOD bias.** Games running RDNU as an FSR DLL sample textures at `-log2(ratio) - 1`
  (`-2` at 2×). Unreal's temporal-upscale path, which the capture plugin relies on, biases by
  `-log2(ratio)`; add the extra `-1` (Unreal: `r.MipMapLODBias -1`) so training data has the
  texture frequency the runtime will see.
- **Jitter.** The plugin tiles Halton(2,3) with periods 31 and 26 per axis; the runtime hands games
  FSR's 8·ratio² phases. Both cover the pixel; nothing to change.
- **Content.** Thin geometry, foliage, particles, text and UI, fast motion, camera cuts, dark and
  bright exposure. Arm trained the release on about 50K Bistro frames and calls that too little.

```bash
# Arm's Bistro splits (35 GB): the seed set and the format reference
git submodule update --init ml/data/arm-neural-graphics-dataset
git -C ml/data/arm-neural-graphics-dataset lfs pull --include "nss/*"

# your captures from the Unreal plugin (ml/capture-unreal): EXR + JSON -> safetensors -> 256 px crops
python ml/retrain/prepare_data.py convert  /captures/foliage_run ~/nss-data/foliage
python ml/retrain/prepare_data.py check    ~/nss-data/foliage/crops

# one training set: symlinks, whole captures on one side of the split, Arm's splits kept
python ml/retrain/prepare_data.py assemble ~/nss-data/set1 \
    ml/data/arm-neural-graphics-dataset/nss ~/nss-data/foliage/crops ~/nss-data/city/crops
```

`check` prints one line per sequence with the ranges the gym and the runtime rely on and flags
what is off. RDNU's own game captures (`RDNU_CAPTURE`) have no ground truth and are not training
data; they are for `runtime/tools/eval/compare.py`.

## Train, quantise, export

```bash
python ml/retrain/retrain.py runs/set1 --data ~/nss-data/set1             # fp32 15 epochs, QAT 2, evaluate, export
python ml/retrain/retrain.py runs/set1 --stages export --install          # into the runtime, verified by test_engine_core
cmake --build build/windows --config Release                              # embeds the new weights
```

What the run directory holds: `nss_config.json` (the gym configuration, edit and rerun a stage),
`fp32/` and `qat/` checkpoints (`best-validated-ckpt.pt` by validation loss), `output/` metrics
JSON, `tensorboard/`, `export/` (manifest, INT8 weights, goldens) and `run.json` (what each stage
produced). `--resume` continues an interrupted fp32 or QAT stage; `--from run/fp32/ckpt-14.pt`
starts from an earlier run instead of the released weights.

Knobs, and the lever in `docs/NSS_Quality_Plan.md` §2 each serves:

| Option | Default | Lever |
|---|---|---|
| `--epochs`, `--lr` | 15, 1e-3 | the released recipe; 3 to 5 epochs at 1e-4 for a light fine-tune on a small set |
| `--unroll` | 16 | 9: 32 teaches accumulation over long histories, at twice the memory |
| `--loss-temporal` | 0.7 | 8: flicker weight; lower sharpens and shimmers |
| `--scale` | 2.0 | mixed-ratio models need captures at that ratio (`UpscalingRatio` in the plugin) |
| `--qat-epochs`, `--qat-lr` | 2, 1e-4 | keep QAT a short tail after fp32 |

Judge a run on the test split's tPSNR and on `compare.py` crops of held-out game captures against
FSR 4.1, not on PSNR alone (§5 of the quality plan).

## What travels into the runtime

`nss_export.py` writes the INT8 weights, the requantisation tables and the manifest. The
manifest carries the learned input quantiser scale; `rdnu_shaderc` passes it to the pre-process
shader as `RDNU_INPUT_SCALE`, so a retrained model needs no shader edits. The sigmoid outputs use a
fixed quantiser (1/254), so the post-process pass is unchanged. `test_engine_core` then checks the
runtime's integer path against the exporter's golden for the new weights, bit for bit.

Smoke test without a GPU, one epoch on one sequence: `retrain.py runs/smoke --data ... --cpu
--batch 1 --unroll 8 --epochs 1 --qat-epochs 1`.
