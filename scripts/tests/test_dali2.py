import os
import torch
import multiprocessing as mp
from basicsr.data.dali_dataloader import build_dali_dataloader

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
            'num_worker_per_gpu': 2,
            'batch_size_per_gpu': 4,
            'pin_memory': True,
            'dataset_enlarge_ratio': 100,
            'prefetch_mode': None,
            'persistent_workers': False,
            'phase': 'train',
            'scale': 4,
            'use_dali': True
        }
    }
}

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    dataset_opt = opt['datasets']['train']
    train_loader = build_dali_dataloader(dataset_opt, num_gpu=1, dist=False, seed=0)

    for i, data in enumerate(train_loader):
        print(i, list(data.keys()))
        if i > 2:
            break
    print("DALI test passed!")
