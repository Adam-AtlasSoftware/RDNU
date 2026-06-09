import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
import numpy as np

class DummySource:
    def __iter__(self): return self
    def __next__(self):
        img_path = '/home/adam/2tb-workspace/RDNU/M3VIR-extracted/Track4/Residential-Areas/Scene04/MovingCameraStaticScene/Realistic_1920x1080_1024sample/RGB_images/Realistic_1920x1080_1024sample_Back_0000.png'
        with open(img_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        # return 3 outputs: 2 frames and 1 depth dummy
        return data, data, np.zeros((2, 10, 10, 1), dtype=np.float32)

@pipeline_def(batch_size=1, num_threads=1, device_id=0)
def pipe():
    inputs = fn.external_source(source=DummySource(), num_outputs=3, batch=False)
    f0 = fn.decoders.image(inputs[0], device='mixed')
    f1 = fn.decoders.image(inputs[1], device='mixed')
    seq = fn.stack(f0, f1)
    return seq, inputs[2]

p = pipe()
p.build()
out = p.run()
print(out[0].as_tensor().shape())
print(out[1].as_tensor().shape())
