#!/bin/bash

# Navigate to the RDG directory so basic paths work
cd "$(dirname "$0")/../../RDG"

# CPU capability limits for Zen 2 compatibility
export ATEN_CPU_CAPABILITY=avx2
export MKL_DEBUG_CPU_TYPE=5
export OPENBLAS_CORETYPE=HASWELL

# Suppress memory allocator warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# Exclude the RTX 3060 (GPU 3) so we only use the three RTX 3090s
export CUDA_VISIBLE_DEVICES=0,1,2

# Bypass failing CPU cores 10 and 26
taskset -c 0-9,11-25,27-31 /home/adam/2tb-workspace/RDNU/environments/env310/bin/python -m torch.distributed.run --nproc_per_node=3 --master_port=29500 basicsr/train.py -opt options/train/train_RDG-s_x4.yml --launcher pytorch --auto_resume
