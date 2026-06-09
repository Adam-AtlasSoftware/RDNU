#!/bin/bash
set -e

# Use GPUs 0, 1, and 3 (the RTX 3090s)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,3
export ONEDNN_MAX_CPU_ISA=AVX2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export NCCL_NVML_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Setup Python Virtual Environment
echo "Setting up Python environment path..."
export PATH="/home/adam/2tb-workspace/RDNU/environments/env310/bin:$PATH"

# Go to workspace root
cd "$(dirname "$0")/../../"

# Download GameIR dataset
echo "Downloading GameIR mini_dataset..."
/home/adam/2tb-workspace/RDNU/environments/venv/bin/hf download LLLebin/GameIR --repo-type dataset --include "mini_dataset/train/GameIR-SR/*" --local-dir data/GameIR
/home/adam/2tb-workspace/RDNU/environments/venv/bin/hf download LLLebin/GameIR --repo-type dataset --include "mini_dataset/test/GameIR-SR/*" --local-dir data/GameIR

# Extract dataset tar files
echo "Extracting datasets..."
cd data/GameIR/mini_dataset/train/GameIR-SR
for tarfile in GameIR-SR-*.tar; do tar -xf "$tarfile" || true; done
cd ../../../../../

cd data/GameIR/mini_dataset/test/GameIR-SR
for tarfile in GameIR-SR-*.tar; do tar -xf "$tarfile" || true; done
cd ../../../../../

# Setup RDG and BasicSR
echo "Setting up RDG..."
cd RDG
# Ensure we have the right pytorch version for cuda
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || pip install torch torchvision torchaudio
# pip install -r requirements.txt
# pip install -e . --use-pep517

# Start training
echo "Starting training..."
numactl --cpunodebind=0 --membind=0 /home/adam/2tb-workspace/RDNU/environments/env310/bin/python -m torch.distributed.run \
    --nproc_per_node=3 \
    --master_port=29505 \
    basicsr/train.py \
    -opt options/train/train_RDG-s_x4.yml \
    --launcher pytorch