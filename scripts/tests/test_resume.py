from basicsr.utils.options import parse_options
import yaml
import os
from os import path as osp
from basicsr.utils import scandir

with open('RDG/options/train/train_RDG-s_x2.yml', 'r') as f:
    opt = yaml.safe_load(f)

print("auto_resume:", opt.get('auto_resume'))
state_path = osp.join('RDG', 'experiments', opt['name'], 'training_states')
print("state_path exists?", osp.isdir(state_path))
states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
print("states found:", states)
