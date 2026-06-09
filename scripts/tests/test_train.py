import subprocess
import os

env = os.environ.copy()
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
env["CUDA_VISIBLE_DEVICES"] = "0,1,2,4"
env["NCCL_P2P_DISABLE"] = "1"
env["NCCL_IB_DISABLE"] = "1"

cmd = [
    "/home/adam/workspace/RDNU/env310/bin/python", "-m", "torch.distributed.run",
    "--nproc_per_node=4",
    "--master_port=29505",
    "basicsr/train.py",
    "-opt", "options/train/train_RDG-s_x4.yml",
    "--launcher", "pytorch"
]

proc = subprocess.Popen(cmd, env=env)
proc.wait()
