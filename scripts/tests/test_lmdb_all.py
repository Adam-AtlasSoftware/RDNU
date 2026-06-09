import lmdb
import cv2
import numpy as np
import os
import tqdm

os.environ['OMP_NUM_THREADS'] = '1'
cv2.setNumThreads(0)

env = lmdb.open('/home/adam/workspace/RDNU/GameIR/mini_dataset/train/GameIR-SR_train.lmdb', readonly=True, lock=False)
with env.begin(write=False) as txn:
    cursor = txn.cursor()
    for key, value in tqdm.tqdm(cursor, total=txn.stat()['entries']):
        if key.endswith(b'.png'):
            img_np = np.frombuffer(value, np.uint8)
            bgr = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
            if bgr is None:
                print(f"Failed to decode: {key}")
            else:
                try:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"Failed cvtColor on {key} with shape {bgr.shape}: {e}")
