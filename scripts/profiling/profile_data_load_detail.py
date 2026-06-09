import sys
import os
sys.path.append('/home/adam/2tb-workspace/RDNU/RDG')
from basicsr.data.dali_dataloader import M3VIR_DALI_Source
from basicsr.utils.options import yaml_load
import time
from basicsr.utils.render_data_util import load_image, load_exr

opt = yaml_load('/home/adam/2tb-workspace/RDNU/RDG/options/train/train_RDG-s_x4.yml')
dataset_opt = opt['datasets']['train']
dataset_opt['scale'] = 4
source = M3VIR_DALI_Source(dataset_opt, batch_size=4)

rel_path, prefix, num_frames, min_idx, max_idx = source.keys[0]

img_path = os.path.join(source.gt_root, rel_path, f"{prefix}_{min_idx:04d}.png")
depth_rel_path = rel_path.replace('RGB_images', 'Depth_images')
depth_path = os.path.join(source.gt_root, depth_rel_path, f"{prefix}_{min_idx:04d}.exr")

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import cv2
cv2.setNumThreads(0)

# Profile PNG
t0 = time.perf_counter()
for _ in range(10): load_image(img_path, 'rgb')
print(f"PNG avg load: {(time.perf_counter() - t0)/10:.4f}s")

# Profile EXR
t0 = time.perf_counter()
for _ in range(10): load_exr(depth_path, 'Depth')
print(f"EXR avg load: {(time.perf_counter() - t0)/10:.4f}s")
