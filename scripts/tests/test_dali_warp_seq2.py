import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
import numpy as np

class DummySource:
    def __iter__(self): return self
    def __next__(self):
        frame0 = np.ones((4, 4, 1), dtype=np.float32) * 10
        frame1 = np.ones((4, 4, 1), dtype=np.float32) * 20
        seq = np.stack([frame0, frame1], axis=0) # [2, 4, 4, 1]
        return [seq]

@pipeline_def(batch_size=1, num_threads=1, device_id=0)
def pipe():
    video = fn.external_source(source=DummySource(), batch=True).gpu()
    
    offset = fn.random.uniform(range=(-0.5, 0.5), shape=[2])
    # Duplicate offset for each frame (2 frames)
    # wait, if offset is [2], we can use fn.stack(offset, offset) to get [2, 2]? No, stack creates a new dimension.
    offset_seq = fn.stack(offset, offset)
    
    mtx = fn.transforms.translation(offset=offset_seq)
    
    warped = fn.warp_affine(video, matrix=mtx)
    return warped

try:
    p = pipe()
    p.build()
    out = p.run()
    print("Success")
except Exception as e:
    print(e)
