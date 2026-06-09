import subprocess
import os

env = os.environ.copy()
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
env["CUDA_VISIBLE_DEVICES"] = "0"

cmd = [
    "/home/adam/workspace/RDNU/env310/bin/python", 
    "basicsr/train.py",
    "-opt", "options/train/train_RDG-s_x4.yml"
]

proc = subprocess.Popen(cmd, env=env)
proc.wait()
