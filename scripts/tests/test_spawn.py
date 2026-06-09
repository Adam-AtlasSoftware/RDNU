import torch.multiprocessing as mp

def worker():
    print("Worker starting")
    import scipy.stats
    print("Worker imported scipy.stats successfully")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    p = mp.Process(target=worker)
    p.start()
    p.join()
