import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import time
from basicsr.data import build_dataset
from basicsr.utils.options import yaml_load

opt = yaml_load('options/train/train_RDG-s_x4.yml')
dataset_opt = opt['datasets']['train']
dataset_opt['phase'] = 'train'
dataset_opt['scale'] = 4

dataset = build_dataset(dataset_opt)

import cProfile
cProfile.run('dataset[0]', 'stats.prof')

import pstats
p = pstats.Stats('stats.prof')
p.strip_dirs().sort_stats('tottime').print_stats(20)
