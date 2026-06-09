import torch
from basicsr.utils.options import parse_options
from basicsr.data import build_dataset, build_dataloader
import os

opt = {
    'name': 'train_RDG_s_x4_GameIR',
    'model_type': 'VideoRenderModel',
    'scale': 4,
    'num_gpu': 4,
    'dist': False,
    'datasets': {
        'train': {
            'name': 'Render',
            'type': 'RenderRecurrentDataset',
            'dataroot_gt': '/home/adam/workspace/RDNU/GameIR/mini_dataset/train/GameIR-SR_train.lmdb',
            'dataroot_lq': '/home/adam/workspace/RDNU/GameIR/mini_dataset/train/GameIR-SR_train.lmdb',
            'meta_info_file': 'basicsr/data/meta_info/TrainData_meta_info.txt',
            'io_backend': {'type': 'lmdb'},
            'num_frame': 6,
            'gt_size': 384,
            'use_lq_normal': False,
            'use_lq_brdf': False,
            'use_lq_motion:': False,
            'use_lq_depth': True,
            'num_worker_per_gpu': 4,
            'batch_size_per_gpu': 4,
            'pin_memory': True,
            'dataset_enlarge_ratio': 100,
            'prefetch_mode': None,
            'persistent_workers': False,
            'phase': 'train',
            'scale': 4
        }
    }
}

dataset_opt = opt['datasets']['train']
train_set = build_dataset(dataset_opt)
train_loader = build_dataloader(train_set, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=0)

for i, data in enumerate(train_loader):
    print(i, list(data.keys()))
    if i > 5:
        break
print("Dataloader test passed!")
