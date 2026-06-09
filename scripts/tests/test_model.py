import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['ONEDNN_MAX_CPU_ISA'] = 'AVX2'
os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
import yaml
from basicsr.models import build_model
from basicsr.utils.options import parse_options

opt = yaml.safe_load(open('options/train/train_RDG-s_x4.yml', 'r'))
opt['is_train'] = True
opt['dist'] = False
opt['num_gpu'] = 1

model = build_model(opt)
print("Model built successfully!")

# Test forward pass
data = {
    'lq_Image': torch.randn(2, 6, 3, 96, 96),
    'gt_Image': torch.randn(2, 6, 3, 384, 384),
    'lq_Depth': torch.randn(2, 6, 1, 96, 96),
    'gt_Depth': torch.randn(2, 6, 1, 384, 384),
    'lq_Motion': torch.randn(2, 6, 2, 96, 96),
    'gt_Motion': torch.randn(2, 6, 2, 384, 384)
}

model.feed_data(data)
model.optimize_parameters(1)
print("Optimization successful!")
