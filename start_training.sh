#!/bin/bash
set -e

# Use GPUs 0, 1, 2, and 4 (the RTX 3090s)
export CUDA_VISIBLE_DEVICES=0,1,2,4

# Setup Python Virtual Environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Download GameIR dataset
echo "Downloading GameIR mini_dataset..."
hf download LLLebin/GameIR --repo-type dataset --include "mini_dataset/train/GameIR-SR/*" --local-dir GameIR
hf download LLLebin/GameIR --repo-type dataset --include "mini_dataset/test/GameIR-SR/*" --local-dir GameIR

# Extract dataset tar files
echo "Extracting datasets..."
cd GameIR/mini_dataset/train/GameIR-SR
for tarfile in GameIR-SR-*.tar; do tar -xf "$tarfile" || true; done
cd ../../../..

cd GameIR/mini_dataset/test/GameIR-SR
for tarfile in GameIR-SR-*.tar; do tar -xf "$tarfile" || true; done
cd ../../../..

# Setup RDG and BasicSR
echo "Setting up RDG..."
cd RDG
# Ensure we have the right pytorch version for cuda
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || pip install torch torchvision torchaudio
pip install -r requirements.txt
python setup.py develop

# Start training
echo "Starting training..."
python basicsr/train.py -opt options/train/train_RDG-s_x4.yml