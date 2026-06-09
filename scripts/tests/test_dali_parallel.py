import sys
sys.path.append('/home/adam/2tb-workspace/RDNU/RDG')
from basicsr.data.dali_dataloader import build_dali_dataloader
from basicsr.utils.options import yaml_load
import torch
import time

opt = yaml_load('/home/adam/2tb-workspace/RDNU/RDG/options/train/train_RDG-s_x4.yml')
dataset_opt = opt['datasets']['train']
dataset_opt['phase'] = 'train'
dataset_opt['scale'] = 4
dataset_opt['batch_size_per_gpu'] = 24
dataset_opt['num_worker_per_gpu'] = 8

# We need to monkeypatch external_source to parallel=True
import nvidia.dali.fn as fn
original_ext = fn.external_source
def patched_ext(*args, **kwargs):
    kwargs['parallel'] = True
    return original_ext(*args, **kwargs)
fn.external_source = patched_ext

loader = build_dali_dataloader(dataset_opt)
t0 = time.time()
print("Pipeline built. Fetching first batch...")
for batch in loader:
    print('lq_Depth shape:', batch.get('lq_Depth', torch.tensor([])).shape)
    break
print("Done in", time.time() - t0)
