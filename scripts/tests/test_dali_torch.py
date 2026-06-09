import sys
import os
sys.path.append('/home/adam/2tb-workspace/RDNU/RDG')
from basicsr.data.dali_dataloader import TorchDaliSource, M3VIR_TorchDataset
from basicsr.utils.options import yaml_load
import torch
import time

opt = yaml_load('/home/adam/2tb-workspace/RDNU/RDG/options/train/train_RDG-s_x4.yml')
dataset_opt = opt['datasets']['train']
dataset_opt['scale'] = 4
dataset_opt['batch_size_per_gpu'] = 8

dataset = M3VIR_TorchDataset(dataset_opt)
source = TorchDaliSource(dataset, batch_size=8, num_workers=16)

t0 = time.time()
for i in range(10):
    batch = next(source)
    print(f"Batch {i} time: {time.time() - t0:.2f}s")
    t0 = time.time()
