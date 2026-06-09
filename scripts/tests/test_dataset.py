import yaml
from basicsr.data.render_dataset import RenderRecurrentDataset

with open('options/train/train_RDG-s_x4.yml', 'r') as f:
    opt = yaml.safe_load(f)

dataset_opt = opt['datasets']['train']
dataset_opt['scale'] = opt['scale']
dataset = RenderRecurrentDataset(dataset_opt)

print("Dataset size:", len(dataset))
import time
import torch
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from basicsr.data import build_dataloader, build_dataset
import yaml

def main():
    with open('options/train/train_RDG-s_x4.yml', 'r') as f:
        opt = yaml.safe_load(f)

    dataset_opt = opt['datasets']['train']
    dataset_opt['scale'] = opt['scale']
    dataset_opt['phase'] = 'train'
    # Use 0 workers so we run in the main thread to see exactly what crashes
    dataset_opt['num_worker_per_gpu'] = 0
    dataset_opt['persistent_workers'] = False
    
    dataset = build_dataset(dataset_opt)
    loader = build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False)

    print("Starting to iterate")
    for i, data in enumerate(loader):
        print(f"Loaded {i} batches")
        if i > 50:
            break
    print("Done!")

if __name__ == '__main__':
    main()
