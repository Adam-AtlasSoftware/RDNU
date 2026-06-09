import lmdb

env = lmdb.open('/home/adam/workspace/RDNU/GameIR/mini_dataset/train/GameIR-SR_train.lmdb', readonly=True, lock=False)
with env.begin(write=False) as txn:
    cursor = txn.cursor()
    for i, (key, value) in enumerate(cursor):
        print(key)
        if i > 10:
            break
