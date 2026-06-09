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
    video = fn.external_source(source=DummySource(), batch=True)
    video = video.gpu()
    reshaped = fn.reshape(video, shape=[4, 4, 2])
    back = fn.reshape(reshaped, shape=[2, 4, 4, 1])
    return video, reshaped, back

p = pipe()
p.build()
out = p.run()
o1 = np.array(out[0].as_cpu().as_tensor())
o2 = np.array(out[1].as_cpu().as_tensor())
o3 = np.array(out[2].as_cpu().as_tensor())

print("Original frame 0, 1:")
print(o1[0, :, 0, 0, 0])
print("Reshaped first pixel channels:")
print(o2[0, 0, 0, :])
print("Back frame 0, 1:")
print(o3[0, :, 0, 0, 0])
