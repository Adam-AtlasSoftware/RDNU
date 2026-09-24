# Radeon Decoupled Neural Upscaler (RDNU)

An open ML temporal upscaler for AMD RDNA2 and RDNA3 GPUs, in the class of FSR 4 and DLSS:
jittered low-resolution colour, depth and motion vectors in, anti-aliased high-resolution frames
out. The runtime is DirectX 12 compute with INT8 networks (DP4a on RDNA2, wave-matrix on RDNA3)
and exposes the FidelityFX API so it can drop into the FSR sample or be injected into games.

The network is [Arm Neural Super Sampling](https://huggingface.co/Arm/neural-super-sampling)
(148K params, ~2.8K MAC per output pixel at 2×), chosen after the review in
[docs/Model_Candidates.md](docs/Model_Candidates.md). The earlier RDG-based model (CNN with
G-buffer guidance, trained here on GameIR/M3VIR/vKITTI2/Sintel) is kept under `RDG/`, `scripts/`
and `runtime/models/` as the reference baseline.

## Status

- NSS backbone exported to the engine's bundle format with per-layer goldens (fp32 and INT8 QAT);
  the harness graph for it is written and awaits GPU validation.
- The RDG network runs end to end on the GPU in the harness (bit-exact INT8, WMMA 1×1 conv).
- Not yet: the NSS pre/post passes in HLSL, the DX12 backend for Arm's NSS component, sample
  integration. Order of work: [docs/NSS_Runtime_Plan.md](docs/NSS_Runtime_Plan.md).

## Layout

```
RDNU/
├── docs/                 # plans and reviews (start with NSS_Runtime_Plan.md)
├── ml/
│   ├── model-gym/        # Arm Neural Graphics Model Gym (train / fine-tune / QAT / export)   [submodule]
│   ├── capture-unreal/   # UE 5.5 dataset capture plugin                                     [submodule]
│   ├── models/nss, nfru  # released weights (Git LFS)                                        [submodules]
│   └── data/             # datasets, opt-in submodules (see docs/Datasets.md)
├── runtime/
│   ├── harness/          # standalone DX12 engine + kernels, validated layer by layer vs golden
│   ├── tools/            # nss_export.py (NSS bundles), dump_golden.py (RDG bundles)
│   ├── src/, shaders/    # FSR-sample backend and its placeholder shaders (to be replaced)
│   └── external/         # vcpkg, DirectX-Headers, FidelityFX SDK fork, Arm Neural Graphics SDK
├── RDG/, scripts/        # RDG baseline: training code, configs, utilities
└── CMakeLists.txt, CMakePresets.json
```

## Setup

```bash
git clone --recurse-submodules git@github.com:Adam-AtlasSoftware/RDNU.git   # FidelityFX fork needs SSH
cd RDNU
git -C ml/models/nss lfs pull                    # 15 MB of weights
```

Datasets are opt-in submodules; fetch instructions and licences: [docs/Datasets.md](docs/Datasets.md).
Windows build (VS Code + CMake, DXC): [docs/build-windows.md](docs/build-windows.md).

## Validating the network on the GPU

```bash
pip install torch numpy                          # CPU is fine
python runtime/tools/nss_export.py               # -> runtime/tools/golden/nss_backbone{,_int8}.rdnut
cmake --build build --target run_engine          # runs every bundle through the DX12 engine
```

`run_engine` prints max|err| per bundle and per checkpoint; fp32 must be within 1e-3, INT8
bit-exact against the integer-path reference.

## Documents

- [NSS_Runtime_Plan.md](docs/NSS_Runtime_Plan.md): architecture, passes, network contract, INT8 scheme, work list
- [NSS_Quality_Plan.md](docs/NSS_Quality_Plan.md): reaching FSR4-class detail; RDNA2 and RDNA3 tiers
- [Model_Candidates.md](docs/Model_Candidates.md): the model review that led here
- [Datasets.md](docs/Datasets.md): sources, licences, fetching
- [RDNU_Runtime_Plan.md](docs/RDNU_Runtime_Plan.md): engine design (tensor layout, WMMA conv), RDG parts superseded

## Licences

RDNU is GPL-3.0 ([LICENSE](LICENSE)). Arm NSS weights: Arm AI Model Community License v1.0
(commercial use permitted; acceptable-use pass-through, notices, marked modifications). Model gym
and capture plugin: Apache-2.0. Arm Neural Graphics SDK: MIT. RDG: Apache-2.0. Datasets: see
[docs/Datasets.md](docs/Datasets.md).

## Acknowledgments

Arm (Neural Super Sampling, Model Gym, SDK), the [RDG](https://github.com/sunny2109/RDG) authors,
and the creators of the GameIR, M3VIR, MPI-Sintel and Virtual KITTI 2 datasets.
