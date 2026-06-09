import sys
import os
sys.path.append('/home/adam/2tb-workspace/RDNU/RDG')
from basicsr.data.dali_dataloader import build_dali_dataloader
from basicsr.utils.options import yaml_load
import time

opt = yaml_load('/home/adam/2tb-workspace/RDNU/RDG/options/train/train_RDG-s_x4.yml')
dataset_opt = opt['datasets']['train']
dataset_opt['scale'] = 4
dataset_opt['batch_size_per_gpu'] = 16

loader = build_dali_dataloader(dataset_opt)

# Measure full loop speed with PyTorch included
t0 = time.time()
print("Warming up DALI pipe...")
dali_iter = iter(loader)
_ = next(dali_iter)
print("Warmup took", time.time() - t0)

for i in range(10):
    t0 = time.time()
    _ = next(dali_iter)
    print(f"Batch {i} iteration total time: {time.time() - t0:.2f}s")
