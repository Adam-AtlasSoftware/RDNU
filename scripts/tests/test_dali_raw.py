import sys
sys.path.append('/home/adam/2tb-workspace/RDNU/RDG')
from basicsr.data.dali_dataloader import build_dali_dataloader
from basicsr.utils.options import yaml_load
import torch

opt = yaml_load('/home/adam/2tb-workspace/RDNU/RDG/options/train/train_RDG-s_x4.yml')
dataset_opt = opt['datasets']['train']
dataset_opt['phase'] = 'train'
dataset_opt['scale'] = opt['scale']
dataset_opt['batch_size_per_gpu'] = 2 # lower for test

loader = build_dali_dataloader(dataset_opt)
dali_iter = loader.iterator
batch = next(dali_iter)
data = batch[0]
for k in data:
    print(k, data[k].shape)
