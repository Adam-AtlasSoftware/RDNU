import sys
sys.path.append('/home/adam/2tb-workspace/RDNU/RDG')
from basicsr.data.dali_dataloader import M3VIR_DALI_Source
from basicsr.utils.options import yaml_load
import time

opt = yaml_load('/home/adam/2tb-workspace/RDNU/RDG/options/train/train_RDG-s_x4.yml')
dataset_opt = opt['datasets']['train']
dataset_opt['scale'] = 4
source = M3VIR_DALI_Source(dataset_opt, batch_size=24)

t0 = time.time()
print("Starting next...")
batch = next(source)
print("Finished next in", time.time() - t0)
