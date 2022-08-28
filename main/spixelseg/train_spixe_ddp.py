import argparse
import sys
import os

import torch
import torch.distributed as dist

os.chdir(sys.path[0])
sys.path.append("..")
import _init_paths
from utils_argument import ddp_spixel_argparse
from utils_train import *
from train_spixel import train_model, dataset_info_from_argument


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = ddp_spixel_argparse(parser)
    args = parser.parse_args()

    # set up torch.distributed
    torch.set_num_threads(1)
    set_local_rank_environ(args.local_rank)
    torch.backends.cudnn.benchmark = True  # Set to True by default for speeding up
    init_dist()
    rank, world_size = get_dist_info()
    #print("Running DDP train on rank {} in {}. local rank {}".format(rank, world_size, args.local_rank))

    set_random_seed(0)
    gpu_num = world_size
    train_model(args, gpu_num, rank, is_ddp=True)
    dist.destroy_process_group()
