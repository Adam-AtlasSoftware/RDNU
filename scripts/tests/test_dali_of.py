import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
import nvidia.dali.types as types
import numpy as np

@pipeline_def
def of_pipeline():
    video = fn.random.uniform(range=[0, 255], shape=[3, 100, 100, 3])
    video = fn.cast(video, dtype=types.UINT8)
    video = video.gpu()
    of = fn.optical_flow(video, device="gpu")
    # Resize OF to original size
    of = fn.resize(of, resize_x=100, resize_y=100, interp_type=types.INTERP_LINEAR)
    return of

try:
    pipe = of_pipeline(batch_size=2, num_threads=2, device_id=0)
    pipe.build()
    out = pipe.run()
    print(out[0].as_tensor().shape())
except Exception as e:
    print("Error:", e)
