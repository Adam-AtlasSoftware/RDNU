import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    tensor = torch.zeros(1).cuda()
    dist.all_reduce(tensor)
    print("Success on rank", dist.get_rank())

if __name__ == "__main__":
    main()
