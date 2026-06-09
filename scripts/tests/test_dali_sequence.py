import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import numpy as np

class SeqSource:
    def __init__(self):
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i > 0: raise StopIteration
        self.i += 1
        
        batch = []
        for _ in range(2): # batch_size
            seq = []
            for _ in range(3): # num_frames
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                seq.append(img)
            batch.append(np.stack(seq, axis=0)) # [frames, H, W, 3]
        return batch # return batch directly if num_outputs=1 and batch=True? Wait.
        # Actually, if num_outputs>1, return tuple of batches. If num_outputs=1, return batch.

@pipeline_def(batch_size=2, num_threads=2, device_id=0)
def pipe():
    video = fn.external_source(source=SeqSource(), batch=True)
    video = video.gpu()
    of = fn.optical_flow(video, device="gpu")
    of = fn.resize(of, resize_x=100, resize_y=100)
    return video, of

p = pipe()
p.build()
out = p.run()
print(out[0].as_tensor().shape())
print(out[1].as_tensor().shape())
