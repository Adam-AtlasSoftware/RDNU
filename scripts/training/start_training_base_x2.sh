#!/bin/bash
set -e

# Use GPUs 0, 2, 3, and 4 (the RTX 3090s) - GPU 1 (RTX 3060) remains free
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,2,3,4

# Optimize CPU instructions
export ONEDNN_MAX_CPU_ISA=AVX2
export MKL_ENABLE_INSTRUCTIONS=AVX2

# Keep distributed stability across mixed PCIe configurations
export NCCL_NVML_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Thread Management
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Allow torch.compile to use up to 8 threads to build GPU kernels much faster
export TORCHINDUCTOR_COMPILE_THREADS=8

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Setup Python Virtual Environment
echo "Setting up Python environment path..."
export PATH="/home/adam/2tb-workspace/RDNU/environments/env310/bin:$PATH"

# Setup RDG and BasicSR
echo "Setting up RDG..."
cd "$(dirname "$0")/../../RDG"

# Start training
echo "Starting training..."
# REMOVED numactl restriction so workers can scale across your EPYC cores naturally
/home/adam/2tb-workspace/RDNU/environments/env310/bin/python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29507 \
    basicsr/train.py \
    -opt options/train/train_RDG_Base_x2.yml \
    --launcher pytorch --auto_resume > ../training_outputs/training_output_base_x2.log 2>&1 &

echo "Training started in background! Check training_outputs/training_output_base_x2.log for progress."