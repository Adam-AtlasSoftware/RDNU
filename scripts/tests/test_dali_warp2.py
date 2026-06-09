import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def

@pipeline_def(batch_size=2, num_threads=2, device_id=0)
def pipe():
    video = fn.random.uniform(range=[0, 255], shape=[3, 100, 100, 3]).gpu()
    mtx = fn.transforms.translation(offset=[10, 10])
    
    # reshape to [H, W, frames*3]
    v_reshaped = fn.reshape(video, shape=[100, 100, 9])
    warped = fn.warp_affine(v_reshaped, matrix=mtx)
    warped = fn.reshape(warped, shape=[3, 100, 100, 3])
    return warped

p = pipe()
p.build()
try:
    p.run()
    print("Success")
except Exception as e:
    print(e)
