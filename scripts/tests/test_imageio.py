import imageio.v3 as iio
import lmdb
import numpy as np

env = lmdb.open('/home/adam/workspace/RDNU/GameIR/mini_dataset/train/GameIR-SR_train.lmdb', readonly=True, lock=False)
with env.begin(write=False) as txn:
    content = txn.get(b'static_town08/19/1440p/00000000.exr')
    if content:
        img = iio.imread(content, extension=".exr")
        print("imageio shape:", img.shape)
        
    content2 = txn.get(b'static_town08/19/1440p/00000000.rgb.png')
    if content2:
        img2 = iio.imread(content2, extension=".png")
        print("imageio shape 2:", img2.shape)
