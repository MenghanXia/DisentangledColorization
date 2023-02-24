import os, glob, sys, pdb
from joblib import delayed, Parallel
from itertools import chain
import logging
import numpy as np
import random
import os.path as osp
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

import _init_paths
import dataset_lab


def build_optimizer(optim_name, lr, wd, params):
    if optim_name == 'adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
    elif optim_name == 'sgd':
        optimizer = optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise NotImplementedError
    return optimizer


def build_LR_scheduler(optimizer, scheduler_name, lr_decay_ratio, max_epochs, start_epoch=0):
    #print("-LR scheduler:%s"%scheduler_name)
    if scheduler_name == 'linear':
        decay_ratio = lr_decay_ratio
        decay_epochs = max_epochs
        polynomial_decay = lambda epoch: 1 + (decay_ratio - 1) * ((epoch+start_epoch)/decay_epochs)\
            if (epoch+start_epoch) < decay_epochs else decay_ratio
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=polynomial_decay)
    elif scheduler_name == 'cosine':
        last_epoch = -1 if start_epoch == 0 else start_epoch
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, last_epoch=last_epoch)
    elif scheduler_name == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, threshold=0.01, patience=5)
    else:
        raise NotImplementedError
    return lr_scheduler


def build_dataloader(dataset_info, mode, logger, gpu_num=1, rank=0, is_ddp=False):
    #data_transform = get_data_transform(mode, dataset_info)
    def get_filelist(data_dir):
        return glob.glob(os.path.join(data_dir, '*.*'))
    
    def get_subdirlist(data_dir):
        #[os.path.join(data_dir,name) for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,name))]
        return filter(os.path.isdir, [os.path.join(data_dir,dirname) for dirname in os.listdir(data_dir)])

    def load_filelist(file_path, root_dir):
        file_list = []
        with open(file_path, "r") as fin:
            for line in fin:
                splits = line.strip().split()
                fname = splits[0]
                label = int(splits[1])
                file_list.append(os.path.join(root_dir, fname))
        return file_list

    def random_picker(file_list, ratio=0.5):
        sub_list = random.sample(file_list, round(len(file_list)*ratio))
        return sub_list

    file_list = []
    if dataset_info['dataset'] == 'disco':
        #data_dir = os.path.join(dataset_info['data_dir'], 'target')
        data_dir = dataset_info['data_dir']
        assert os.path.exists(data_dir), "@dir:'%s' NOT exist ..."%data_dir
        file_list = get_filelist(data_dir)
        file_list.sort()
        dataset = dataset_lab.LabDataset(filelist=file_list)
    elif dataset_info['dataset'] == 'imagenet':
        data_dir = os.path.join(dataset_info['data_dir'])
        assert os.path.exists(data_dir), "@dir:'%s' NOT exist ..."%data_dir
        '''
        subdir_list = get_subdirlist(data_dir)
        filelist_set = Parallel(n_jobs=4)(delayed(get_filelist)(subdir) for subdir in subdir_list)
        file_list = list(chain(*filelist_set))
        '''
        ## speedup: load the filename list of dataset split
        filepath = os.path.join('/apdcephfs/share_1290939/0_public_datasets/imageNet_2012/', '%s_list.txt'%mode)
        file_list = load_filelist(filepath, data_dir)
        if mode == 'val':
            file_list = random_picker(file_list, ratio=0.1)
        file_list.sort()
        dataset = dataset_lab.LabDataset(filelist=file_list, resize=dataset_info['input_size'])
    elif dataset_info['dataset'] == 'coco':
        data_dir = dataset_info['data_dir']+'2017'
        assert os.path.exists(data_dir), "@dir:'%s' NOT exist ..."%data_dir
        file_list = get_filelist(data_dir)
        file_list.sort()
        dataset = dataset_lab.LabDataset(filelist=file_list, resize=dataset_info['input_size'])
    else:
        raise NotImplementedError
    if rank == 0:
        logger.info(">> loaded %d images from %s [%s]." % (len(file_list), dataset_info['dataset'], mode))
    data_loader = get_dataloader(dataset, mode, dataset_info['batch_size'], dataset_info['num_workers'], gpu_num, rank, is_ddp)
    return data_loader


def get_dataloader(dataset, mode, batch_size, num_workers, gpu_num=1, rank=0, is_ddp=False):
    need_shuffle = True if mode == 'train' else False
    drop_last = False if mode == 'test' else True
    if is_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, \
            num_replicas=gpu_num, rank=rank, shuffle=need_shuffle)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 sampler=train_sampler,
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 drop_last=drop_last)
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=need_shuffle,
                                                 num_workers=num_workers,
                                                 drop_last=drop_last)
    return dataloader


def adjust_learning_rate(lr_scheduler, scheduler_name, optimizer, epoch):
    '''decay the learning rate based on schedule'''
    if lr_scheduler:
        lr_scheduler.step()
        epoch_lr = lr_scheduler.get_lr()[0]
    else:
        epoch_lr = optimizer.state_dict()['param_groups'][0]['lr']
    return epoch_lr


def load_checkpoint(checkpt_path, model_without_dp, optimizer=None, is_resume=False):
    start_epoch = 0
    best_acc = 0.0
    data_dict = torch.load(checkpt_path, map_location=torch.device('cpu'))
    '''
    model_dict = model_without_dp.state_dict()
    load_dict = data_dict['state_dict']
    for param_tensor in load_dict:
        print(param_tensor,'\t', load_dict[param_tensor].size())
    pdb.set_trace()
    '''
    model_without_dp.load_state_dict(data_dict['state_dict'])
    if is_resume:
        optimizer.load_state_dict(data_dict['optimizer'])
        start_epoch = data_dict['epoch'] + 1
        best_acc = data_dict['best_loss']
    return start_epoch, best_acc


def save_checkpoint(checkpt_path, data_dict, is_best=False, epoch_points=[]):
    epoch = data_dict['epoch']
    if is_best:
        save_path = os.path.join(checkpt_path, 'model_best.pth.tar')
    else:
        #save_path = os.path.join(ckpt_path, 'model_epoch%d.pth.tar' % epoch)
        save_path = os.path.join(checkpt_path, 'model_last.pth.tar')
        if data_dict['epoch'] in epoch_points:
            save_path = os.path.join(checkpt_path, 'model_epoch%d.pth.tar'%data_dict['epoch'])
    torch.save(data_dict, save_path)


def set_path(save_dir, exp_name, gpu_no):
    output_path = os.path.join(save_dir, exp_name)
    if gpu_no == 0 and not os.path.exists(output_path):
        os.makedirs(output_path)
    ckpt_path = os.path.join(output_path, 'checkpts')
    log_path = os.path.join(output_path, 'logdir')
    img_path = os.path.join(output_path, 'image')
    if gpu_no == 0 and not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if  gpu_no == 0 and not os.path.exists(log_path):
        os.makedirs(log_path)
    if  gpu_no == 0 and not os.path.exists(img_path):
        os.makedirs(img_path)
    return ckpt_path, log_path, img_path, output_path


def set_logger(logfile):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


## DDP APIs ----------------------------------------------------------
def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def set_local_rank_environ(local_rank):
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(local_rank)


def _init_dist_pytorch(backend, **kwargs):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    #local_rank = int(os.environ('LOCAL_RANK'))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, **kwargs)

def init_dist(launcher='pytorch', backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        # _init_dist_mpi(backend, **kwargs)
        raise ValueError(f'Invalid launcher type: {launcher}')
    elif launcher == 'slurm':
        # _init_dist_slurm(backend, **kwargs)
        raise ValueError(f'Invalid launcher type: {launcher}')
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sum_reduce_tensor(tensor):
    if not dist.is_available() or \
        not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def sum_reduce_scalar(value):
    spawn_tenor = torch.Tensor([value]).cuda()
    spawn_tenor = sum_reduce_tensor(spawn_tenor)
    return spawn_tenor.item()

def mean_reduce_tensor(tensor):
    if not dist.is_available() or \
        not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    world_size = dist.get_world_size()
    rt /= world_size
    return rt

def mean_reduce_scalar(value):
    spawn_tenor = torch.Tensor([value]).cuda()
    spawn_tenor = mean_reduce_tensor(spawn_tenor)
    return spawn_tenor.item()