import sys
sys.path.append('/home/adam/2tb-workspace/RDNU/RDG')
from basicsr.data import build_dataset
from basicsr.utils.options import yaml_load
import torch

opt = yaml_load('/home/adam/2tb-workspace/RDNU/RDG/options/train/train_RDG-s_x4.yml')
dataset_opt = opt['datasets']['train']
dataset_opt['phase'] = 'train'
dataset_opt['scale'] = opt['scale']
dataset = build_dataset(dataset_opt)
print("Dataset size:", len(dataset))
sample = dataset[0]
print("Got sample!")
print("lq keys:", sample['lq'].keys() if isinstance(sample['lq'], dict) else "Not dict")
print("gt keys:", sample['gt'].keys() if isinstance(sample['gt'], dict) else "Not dict")
print("lq image shape:", sample['lq']['Image'].shape)
print("gt image shape:", sample['gt']['Image'].shape)
