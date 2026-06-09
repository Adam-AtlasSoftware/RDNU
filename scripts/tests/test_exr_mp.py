import multiprocessing as mp
import lmdb
import numpy as np
import cv2
import os

def worker_fn(idx):
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    env = lmdb.open('/home/adam/workspace/RDNU/GameIR/mini_dataset/train/GameIR-SR_train.lmdb', readonly=True, lock=False, readahead=False)
    with env.begin(write=False) as txn:
        # get an exr
        val = txn.get(b'00/00/00/0001.depth.png')
        if val is None:
            return
        img_np = np.frombuffer(val, np.uint8)
        try:
            bgr = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print("decode error", e)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    with mp.Pool(16) as p:
        p.map(worker_fn, range(500))
    print("MP EXR LMDB OK")
