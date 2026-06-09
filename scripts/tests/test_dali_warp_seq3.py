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
    mtx = fn.transforms.translation(offset=offset)
    
    # We need 2 matrices for 2 frames.
    mtx_seq = fn.stack(mtx, mtx)
    
    warped = fn.warp_affine(video, matrix=mtx_seq)
    return warped

try:
    p = pipe()
    p.build()
    out = p.run()
    print("Success, shape:", out[0].as_tensor().shape())
except Exception as e:
    print(e)
