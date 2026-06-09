import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import yaml
from basicsr.data import build_dataset, build_dataloader
from basicsr.utils.options import yaml_load
import multiprocessing as mp

if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    opt = yaml_load('options/train/train_RDG-s_x4.yml')
    dataset_opt = opt['datasets']['train']
    dataset_opt['phase'] = 'train'
    dataset_opt['scale'] = 4
    dataset_opt['num_worker_per_gpu'] = 4
    dataset_opt['batch_size_per_gpu'] = 8

    print("Building dataset...")
    dataset = build_dataset(dataset_opt)
    
    print("Building dataloader...")
    dataloader = build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=0)
    
    print(f"Dataloader length: {len(dataloader)}")

    import tqdm
    for i, data in enumerate(tqdm.tqdm(dataloader)):
        if i > 10:
            break
