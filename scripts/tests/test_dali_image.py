import sys
sys.path.append('/home/adam/2tb-workspace/RDNU/RDG')
from basicsr.data.dali_dataloader import build_dali_dataloader
from basicsr.utils.options import yaml_load
import torch
from PIL import Image
import numpy as np

opt = yaml_load('/home/adam/2tb-workspace/RDNU/RDG/options/train/train_RDG-s_x4.yml')
dataset_opt = opt['datasets']['train']
dataset_opt['phase'] = 'train'
dataset_opt['scale'] = opt['scale']
dataset_opt['batch_size_per_gpu'] = 2

loader = build_dali_dataloader(dataset_opt)
dali_iter = loader.iterator
batch = next(dali_iter)
data = batch[0]

img = data['lq_Image'][0, 0].permute(1, 2, 0).cpu().numpy() * 255.0
Image.fromarray(img.astype(np.uint8)).save('test_lq.png')

img2 = data['gt_Image'][0, 0].permute(1, 2, 0).cpu().numpy() * 255.0
Image.fromarray(img2.astype(np.uint8)).save('test_gt.png')
print("Saved")
