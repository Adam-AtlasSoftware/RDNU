import pyexr
import numpy as np
import time
import concurrent.futures

depth_path = '/home/adam/2tb-workspace/RDNU/M3VIR-extracted/Track4/Residential-Areas/Scene04/MovingCameraStaticScene/Realistic_1920x1080_1024sample/Depth_images/Realistic_1920x1080_1024sample_Back_0000.exr'

def load(path):
    return pyexr.open(path).get()

if __name__ == '__main__':
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(load, [depth_path] * 96))
    print('pyexr load 96 depths time (Thread Pool):', time.time() - t0)
