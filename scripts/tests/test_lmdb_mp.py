import lmdb
import multiprocessing as mp
import cv2
import numpy as np

def worker(worker_id):
    env = lmdb.open('/home/adam/workspace/RDNU/GameIR/mini_dataset/train/GameIR-SR_train.lmdb', readonly=True, lock=False)
    with env.begin(write=False) as txn:
        key = b'static_town08/19/1440p/00000000.rgb.png'
        val = txn.get(key)
        if val is None:
            print(f"[{worker_id}] Key not found!")
        else:
            print(f"[{worker_id}] Val size:", len(val))
            import io
            from PIL import Image
            try:
                img = Image.open(io.BytesIO(val)).convert('RGBA')
                rgb = np.array(img)
                print(f"[{worker_id}] Decoded shape:", rgb.shape)
            except Exception as e:
                print(f"[{worker_id}] Error decoding: {e}")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    processes = []
    for i in range(4):
        p = mp.Process(target=worker, args=(i,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
