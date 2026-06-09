import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def

@pipeline_def(batch_size=2, num_threads=2, device_id=0)
def pipe():
    jx = fn.random.uniform(range=(-0.5, 0.5))
    jy = fn.random.uniform(range=(-0.5, 0.5))
    offset = fn.stack(jx, jy)
    return fn.transforms.translation(offset=offset)

p = pipe()
p.build()
out = p.run()
print(out[0].as_tensor().shape())
