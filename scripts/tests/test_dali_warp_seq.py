import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def

@pipeline_def(batch_size=2, num_threads=2, device_id=0)
def pipe():
    video = fn.random.uniform(range=[0, 255], shape=[6, 100, 100, 3]).gpu()
    mtx = fn.transforms.translation(offset=[10, 10])
    
    # Try just returning mtx and see its shape.
    # What if we use sequence_rearrange on mtx?
    # No, wait. If we reshape video to [100, 100, 18], that worked! 
    # But wait, did it scramble? Let's check with an actual image.
    pass

