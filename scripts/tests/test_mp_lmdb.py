import multiprocessing as mp
import lmdb
import numpy as np
import cv2

def worker_fn(idx):
    env = lmdb.open('/home/adam/workspace/RDNU/GameIR/mini_dataset/train/GameIR-SR_train.lmdb', readonly=True, lock=False, readahead=False)
    with env.begin(write=False) as txn:
        val = txn.get(b'00/00/00/0001.png')
        if val is not None:
            img_np = np.frombuffer(val, np.uint8)
            bgr = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    with mp.Pool(4) as p:
        p.map(worker_fn, range(100))
    print("MP LMDB OK")
