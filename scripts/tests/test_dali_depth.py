import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
import nvidia.dali.types as types

@pipeline_def(batch_size=2, num_threads=2, device_id=0)
def pipe():
    video = fn.random.uniform(range=[0, 255], shape=[6, 384, 384, 1]).gpu()
    resized = fn.resize(video, resize_x=96, resize_y=96, interp_type=types.INTERP_NN)
    return resized

p = pipe()
p.build()
out = p.run()
print("Resize shape:", out[0].as_tensor().shape())
