import torch
from torch.utils.data import Dataset, DataLoader
import pyexr
import numpy as np
import time

depth_path = '/home/adam/2tb-workspace/RDNU/M3VIR-extracted/Track4/Residential-Areas/Scene04/MovingCameraStaticScene/Realistic_1920x1080_1024sample/Depth_images/Realistic_1920x1080_1024sample_Back_0000.exr'

class ExrDataset(Dataset):
    def __len__(self): return 96*5 # 5 batches
    def __getitem__(self, idx):
        return torch.from_numpy(pyexr.open(depth_path).get())

if __name__ == '__main__':
    dataset = ExrDataset()
    loader = DataLoader(dataset, batch_size=96, num_workers=16)
    
    # Warmup
    it = iter(loader)
    next(it)
    
    t0 = time.time()
    for batch in it:
        print("Batch time:", time.time() - t0)
        t0 = time.time()
