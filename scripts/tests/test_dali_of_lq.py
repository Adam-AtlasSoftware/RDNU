import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
import nvidia.dali.types as types
import numpy as np

@pipeline_def(batch_size=2, num_threads=2, device_id=0)
def pipe():
    video = fn.random.uniform(range=[0, 255], shape=[6, 96, 96, 3])
    video = fn.cast(video, dtype=types.UINT8).gpu()
    
    of = fn.optical_flow(video, device="gpu")
    of_resized = fn.resize(of, resize_x=96, resize_y=96, interp_type=types.INTERP_LINEAR)
    of_resized = of_resized * 4.0 
    
    return of_resized

p = pipe()
p.build()
out = p.run()
print(out[0].as_tensor().shape())
