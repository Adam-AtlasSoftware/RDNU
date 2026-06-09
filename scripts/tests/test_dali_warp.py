import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def

@pipeline_def(batch_size=2, num_threads=2, device_id=0)
def pipe():
    video = fn.random.uniform(range=[0, 255], shape=[3, 100, 100, 3]).gpu()
    mtx = fn.transforms.translation(offset=[10, 10])
    
    # We can use fn.sequence_rearrange or fn.squeeze? 
    # Let's try fn.warp_affine on video
    try:
        return fn.warp_affine(video, matrix=mtx)
    except Exception as e:
        print("Error natively")
    
    return video
p = pipe()
p.build()
