import cv2
import multiprocessing as mp
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
import scipy.stats

def worker():
    print("Worker starting")
    print("Worker finished")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    p = mp.Process(target=worker)
    p.start()
    p.join()
