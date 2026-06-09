import sys
import yaml
from basicsr.data.render_dataset import RenderRecurrentDataset

with open('options/train/train_RDG-s_x4.yml', 'r') as f:
    opt = yaml.safe_load(f)

dataset_opt = opt['datasets']['train']
dataset_opt['scale'] = opt['scale']
dataset_opt['phase'] = 'train'

dataset = RenderRecurrentDataset(dataset_opt)

print("Reading item 0")
data = dataset[0]
print("Done reading item 0")
print("Reading item 1")
data = dataset[1]
print("Done reading item 1")
