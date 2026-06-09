import sys
sys.path.append('/home/adam/2tb-workspace/RDNU/RDG')
from basicsr.data.dali_dataloader import build_dali_dataloader
from basicsr.utils.options import yaml_load

opt = yaml_load('/home/adam/2tb-workspace/RDNU/RDG/options/train/train_RDG-s_x4.yml')
dataset_opt = opt['datasets']['train']
dataset_opt['phase'] = 'train'
dataset_opt['scale'] = opt['scale']

loader = build_dali_dataloader(dataset_opt)
print(len(loader))
