import sys
sys.path.append('/home/adam/2tb-workspace/RDNU/RDG')
from basicsr.data.dali_dataloader import M3VIR_DALI_Source
from basicsr.utils.options import yaml_load
import time

class DummySampleInfo:
    def __init__(self, idx):
        self.idx_in_epoch = idx

opt = yaml_load('/home/adam/2tb-workspace/RDNU/RDG/options/train/train_RDG-s_x4.yml')
dataset_opt = opt['datasets']['train']
dataset_opt['scale'] = 4
source = M3VIR_DALI_Source(dataset_opt, batch_size=4)

times = []
for i in range(10):
    t0 = time.perf_counter()
    res = source(DummySampleInfo(i))
    times.append(time.perf_counter() - t0)

print(f"Average time to load ONE sample (6 frames): {sum(times)/len(times):.4f}s")
