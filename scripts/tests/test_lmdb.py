import lmdb
import cv2
import numpy as np
import os

os.environ['OMP_NUM_THREADS'] = '1'
cv2.setNumThreads(0)

env = lmdb.open('/home/adam/workspace/RDNU/GameIR/mini_dataset/train/GameIR-SR_train.lmdb', readonly=True, lock=False)
with env.begin(write=False) as txn:
    key = b'static_town08/19/1440p/00000000.rgb.png'
    val = txn.get(key)
    if val is None:
        print("Key not found!")
    else:
        print("Val size:", len(val))
        img_np = np.frombuffer(val, np.uint8)
        bgr = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
        print("Decoded shape:", bgr.shape)
