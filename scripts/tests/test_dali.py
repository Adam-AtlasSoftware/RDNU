import sys
import os
sys.path.append('/home/adam/2tb-workspace/RDNU/RDG')
from basicsr.data.dali_dataloader import build_dali_dataloader
from basicsr.utils.options import yaml_load
import torch
import time

def main():
    opt = yaml_load('/home/adam/2tb-workspace/RDNU/RDG/options/train/train_RDG-s_x4.yml')
    dataset_opt = opt['datasets']['train']
    dataset_opt['phase'] = 'train'
    dataset_opt['scale'] = opt['scale']
    dataset_opt['batch_size_per_gpu'] = 8
    
    loader = build_dali_dataloader(dataset_opt)

    print("Fetching batches...")
    for i in range(10):
        t0 = time.time()
        for batch in loader:
            print(f'Batch {i} fetched in {time.time()-t0:.2f}s')
            break

if __name__ == '__main__':
    main()
