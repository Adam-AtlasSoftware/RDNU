import sys
import os
sys.path.append('/home/adam/2tb-workspace/RDNU/RDG')
from basicsr.utils.options import yaml_load
import torch
import time
import numpy as np
import random
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def

opt = yaml_load('/home/adam/2tb-workspace/RDNU/RDG/options/train/train_RDG-s_x4.yml')
dataset_opt = opt['datasets']['train']

class OptSource:
    def __init__(self, opt):
        self.num_frame = opt['num_frame']
        self.gt_root = opt['dataroot_gt']
        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) >= 5:
                    self.keys.append((parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4])))
                    
    def __call__(self, sample_info):
        import cv2
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        idx = sample_info.idx_in_epoch % len(self.keys)
        rel_path, prefix, num_frames, min_idx, max_idx = self.keys[idx]
        
        start_frame_idx = random.randint(min_idx, max_idx - self.num_frame + 1)
        end_frame_idx = start_frame_idx + self.num_frame
        
        encoded_images = []
        depths = []
        
        for i in range(start_frame_idx, end_frame_idx):
            frame_idx_str = f"{i:04d}"
            
            img_path = os.path.join(self.gt_root, rel_path, f"{prefix}_{frame_idx_str}.png")
            depth_rel_path = rel_path.replace('RGB_images', 'Depth_images')
            depth_path = os.path.join(self.gt_root, depth_rel_path, f"{prefix}_{frame_idx_str}.exr")
            
            with open(img_path, 'rb') as f:
                encoded_images.append(np.frombuffer(f.read(), dtype=np.uint8))
                
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth.ndim == 2:
                depth = depth[:, :, None]
            depths.append(depth)
            
        depths_out = np.stack(depths, axis=0).astype(np.float32)
        return tuple(encoded_images) + (depths_out,)

@pipeline_def(batch_size=2, num_threads=2, device_id=0)
def pipe(num_frame):
    source = OptSource(dataset_opt)
    inputs = fn.external_source(source=source, num_outputs=num_frame+1, batch=False, parallel=False)
    encoded = [inputs[i] for i in range(num_frame)]
    gt_depths = inputs[num_frame]
    
    decoded = [fn.decoders.image(f, device='mixed') for f in encoded]
    gt_imgs_gpu = fn.stack(*decoded)
    
    return gt_imgs_gpu, gt_depths

p = pipe(dataset_opt['num_frame'])
p.build()
t0 = time.time()
for _ in range(5):
    t1 = time.time()
    out = p.run()
    print("Batch fetched in", time.time() - t1)
print("Total 5 batches in", time.time() - t0)
