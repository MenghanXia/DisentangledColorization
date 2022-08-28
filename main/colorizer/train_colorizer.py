import os, glob, sys, logging
import argparse, pdb, datetime, time
import numpy as np
import random, pickle
from tqdm import tqdm 
from PIL import Image
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F

os.chdir(sys.path[0])
sys.path.append("..")
import _init_paths
from utils_argument import pcolor_argparser
from utils_train import *
import model, loss, basic
import util


def train_model(args, gpu_num, gpu_no, is_ddp):
    datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    is_multi_gpus = gpu_num > 1
    ## WORKSPACE >>>>>>>>>>>>>>>
    checkpt_dir, log_dir, img_dir, work_dir = set_path(args.save_dir, args.exp_name, gpu_no)
    ## plot tools
    train_plotter, val_plotter = None, None
    logger = None
    if gpu_no == 0:
        writer_val = SummaryWriter(logdir=os.path.join(log_dir, 'val'))
        writer_train = SummaryWriter(logdir=os.path.join(log_dir, 'train'))
        val_plotter = util.PlotterThread(writer_val)
        train_plotter = util.PlotterThread(writer_train)
        logger = set_logger(logfile=os.path.join(work_dir, 'log_%s.txt'%datetime))
        logger.info('@%s (effective batch size = %d)' % (datetime, gpu_num*args.batch_size))
        logger.info('***** %s [GPU_Num:%d | DDP:%s | Resume:%s] *****'%(args.exp_name, gpu_num, is_ddp, args.resume))
    
    ## DATASET >>>>>>>>>>>>>>>
    data_dir = args.data_dir
    if not is_ddp and is_multi_gpus:
        args.batch_size = args.batch_size*gpu_num 
    dataset_info = dataset_info_from_argument(args)
    dataset_info['data_dir'] = os.path.join(data_dir, 'train')
    train_loader = build_dataloader(dataset_info, mode='train', logger=logger, gpu_num=gpu_num, rank=gpu_no, is_ddp=is_ddp)
    dataset_info['data_dir'] = os.path.join(data_dir, 'val')
    val_loader = build_dataloader(dataset_info, mode='val', logger=logger, gpu_num=gpu_num, rank=gpu_no, is_ddp=is_ddp)
    if gpu_no == 0:
        logger.info(">> dataset (%d iters) was created" % len(train_loader))

    ## MODEL >>>>>>>>>>>>>>>
    hcolor_model = eval('model.'+args.model)(inChannel=1, outChannel=313, sp_size=args.psize, use_dense_pos=args.dense_pos,\
                                             n_clusters=args.n_clusters, random_hint=args.random_hint, spix_pos=args.spix_pos, \
                                             learning_pos=args.learning_pos, hint2regress=args.hint2regress, enhanced=args.enhanced, rank=gpu_no)
    ckpt_name = 'spix8ab-imagenet_last.pth.tar' if args.psize == 8 else 'spix16ab-imagenet_last.pth.tar'
    pretrain_weight = os.path.join(args.ckpt_dir, ckpt_name)
    hcolor_model.load_and_froze_weight(pretrain_weight)
    if gpu_no == 0:
        print('@model params:%.3f (M)'%(basic.getParamsAmount(hcolor_model)/1e6))
    ## distributed model
    if is_ddp:
        hcolor_model = hcolor_model.cuda(gpu_no) 
        hcolor_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(hcolor_model)
        hcolor_model = torch.nn.parallel.DistributedDataParallel(hcolor_model, device_ids=[gpu_no], find_unused_parameters=True)
        model_without_dp = hcolor_model.module
    else:
        if is_multi_gpus:
            hcolor_model = torch.nn.DataParallel(hcolor_model).cuda()
            model_without_dp = hcolor_model.module
        else:
            hcolor_model = hcolor_model.cuda(gpu_no)
            model_without_dp = hcolor_model

    ## OPTIMIZER >>>>>>>>>>>>>>>
    #params = model_without_dp.get_trainable_params()
    optimizer = build_optimizer(args.optim, args.lr, args.wd, model_without_dp.parameters())
    ## resume
    start_epoch, best_loss = 0, 999
    resume_pth = glob.glob(os.path.join(checkpt_dir, 'model_last.pth.tar'))
    if resume_pth and args.resume:
        start_epoch, best_loss = load_checkpoint(resume_pth[0], model_without_dp, optimizer, True)
        if gpu_no == 0:
            logger.info('>> resume at checkpoint epoch %d'%start_epoch)
    ## learning rate scheduler
    lr_scheduler = build_LR_scheduler(optimizer, args.scheduler, args.decay_ratio, args.epochs, start_epoch)
    ## ce_loss
    ce_loss = loss.AnchorColorProbLoss(hint2regress=args.hint2regress, enhanced=args.enhanced, \
                                            with_grad=args.in_gradient, mpdist=is_ddp, gpu_no=gpu_no)

    meta_dict = pack_meta_data(args, img_dir, gpu_no)
    ## LOOP >>>>>>>>>>>>>>>
    torch.backends.cudnn.benchmark = True
    for epoch in range(start_epoch, args.epochs):
        ## training
        train_loss = train_one_epoch(epoch, train_loader, hcolor_model, ce_loss, optimizer, meta_dict, train_plotter, \
                                     logger, gpu_no, is_ddp)
        lr = adjust_learning_rate(lr_scheduler, args.scheduler, optimizer, epoch)
        ## validating
        if epoch % args.eval_freq == 0 or epoch == args.epochs-1:
            val_loss = validate(epoch, val_loader, hcolor_model, ce_loss, meta_dict, val_plotter, logger, gpu_no, is_ddp)            
        
        if gpu_no != 0:
            continue
        ## saving
        logger.info('>> saving checkpoint ...')
        save_dict = {
            'epoch': epoch,
            'state_dict': model_without_dp.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpt_dir, save_dict)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(checkpt_dir, save_dict, True)
    if gpu_no == 0:
        writer_train.close()
        writer_val.close()
        logger.info('@Training from ep %d to ep %d finished' % (start_epoch, args.epochs))
 

def train_one_epoch(epoch, data_loader, model, criterion, optimizer, meta_dict, train_plotter, logger, gpu_no, is_ddp):
    try:
        model.module.set_train()
    except:
        model.set_train()
    n_epochs, sp_size = meta_dict['n_epochs'], meta_dict['sp_size']
    color_class = meta_dict['color_class']

    loss_terms = {'totalLoss':0, 'palLoss':0, 'refLoss':0, 'recLoss':0}
    start_time = time.time()
    st = time.time()
    for batch_idx, sample_batch in enumerate(data_loader):
        input_grays, input_colors = sample_batch['gray'], sample_batch['color']
        input_grays = input_grays.cuda(non_blocking=True)
        input_colors = input_colors.cuda(non_blocking=True)

        et = time.time()
        ## forward
        pal_logit, ref_logit, pred_AB, affinity_map, spix_colors, hint_mask = model(input_grays, input_colors)
        spix_gt_labels = torch.max(color_class.encode_ab2ind(spix_colors), dim=1, keepdim=True)[1]
        class_weights = color_class.get_classweights(spix_gt_labels)
        data_dict = {'target_label':spix_gt_labels, 'pal_prob':pal_logit, 'ref_prob':ref_logit, 'pred_color':pred_AB, \
                     'class_weight':class_weights, 'input_gray':input_grays, 'input_color':input_colors, 'spix_color':spix_colors}
        loss_dict = criterion(data_dict, epoch)
        totalLoss_idx = loss_dict['totalLoss']

        optimizer.zero_grad()
        totalLoss_idx.backward()
        optimizer.step()

        ## average loss
        with torch.no_grad():
            for key in loss_terms:
                loss_terms[key] += loss_dict[key]

        ## print iteration
        if gpu_no == 0 and (batch_idx+1) % 100 == 0: 
            logger.info(">> [%d/%d] iter:%d loss:%4.4f [io/proc:%4.3f%%]" % \
                    (epoch+1, n_epochs, batch_idx+1, totalLoss_idx.item(), 100*(et-st)/(time.time()-et)))
        st = time.time()

    ## epoch summary: plot and print
    if is_ddp:
        for key in loss_terms:
            loss_terms[key] = mean_reduce_tensor(loss_terms[key])
    if gpu_no == 0:
        for key in loss_terms:
            loss_terms[key] /= (batch_idx+1)
            train_plotter.add_data('train/%s'%key, loss_terms[key].item(), epoch)
        logger.info("--- epoch: %d/%d | train: %4.4f | duration: %4.2f ---" % \
                (epoch+1, n_epochs, loss_terms['totalLoss'].item(), (time.time() - start_time)))
        
    return loss_terms['totalLoss'].item()


def validate(epoch, data_loader, model, criterion, meta_dict, val_plotter, logger, gpu_no, is_ddp):
    model.eval()
    n_epochs = meta_dict['n_epochs']
    color_class = meta_dict['color_class']
    sp_size = meta_dict['sp_size']
    gpu_num = torch.cuda.device_count() if is_ddp else 1

    def split_spixels(assign_map):
        N,C,H,W = assign_map.shape
        spixel_id_map = meta_dict['spixel_ids'].expand(N,-1,-1,-1)
        assig_max,_ = torch.max(assign_map, dim=1, keepdim=True)
        assignment_ = torch.where(assign_map == assig_max, torch.ones(assign_map.shape).cuda(),torch.zeros(assign_map.shape).cuda())
        ## winner take all
        new_spixl_map_ = spixel_id_map * assignment_
        new_spixl_map = torch.sum(new_spixl_map_,dim=1,keepdim=True).type(torch.int)
        return new_spixl_map

    start_time = time.time()
    with torch.no_grad():
        total_loss = 0
        for batch_idx, sample_batch in enumerate(data_loader):
            input_grays, input_colors, input_BGRs = sample_batch['gray'], sample_batch['color'], sample_batch['BGR']
            input_grays = input_grays.cuda(non_blocking=True)
            input_colors = input_colors.cuda(non_blocking=True)
            input_BGRs = input_BGRs.cuda(non_blocking=True)

            ## forward
            pal_logit, ref_logit, pred_AB, affinity_map, spix_colors, hint_mask = model(input_grays, input_colors)
            spix_gt_labels = torch.max(color_class.encode_ab2ind(spix_colors), dim=1, keepdim=True)[1]
            class_weights = color_class.get_classweights(spix_gt_labels)
            data_dict = {'target_label':spix_gt_labels, 'pal_prob':pal_logit, 'ref_prob':ref_logit, 'pred_color':pred_AB, \
                         'class_weight':class_weights, 'input_gray':input_grays, 'input_color':input_colors, 'spix_color':spix_colors}
            loss_dict = criterion(data_dict, epoch)
            totalLoss_idx = loss_dict['totalLoss']
            total_loss += totalLoss_idx
            
            ## save intermediate images
            pred_colors = color_class.decode_ind2ab(pal_logit, T=0)
            pred_colors = basic.upfeat(pred_colors, affinity_map, sp_size, sp_size)
            pred_labs = torch.cat((input_grays,pred_colors), dim=1)
            lab_imgs = basic.tensor2array(pred_labs)
            util.save_normLabs_from_batch(lab_imgs, meta_dict['img_dir'], None, batch_idx*gpu_num+gpu_no)

            #hint_maps = F.interpolate(gate_maps, scale_factor=16, mode='nearest')
            analysis_anchor_pos = False
            if analysis_anchor_pos:
                N,C,HW = hint_mask.shape
                hint_mask = hint_mask.view(N,C,16,16)
                show_maps = torch.where(hint_mask > 0.5, -1.0*torch.ones_like(hint_mask), cluster_mask)
                show_maps = basic.upfeat(show_maps, affinity_map, sp_size, sp_size)
                show_imgs = basic.tensor2array(show_maps)
                util.save_images_from_batch(show_imgs, meta_dict['img_dir'], None, batch_idx*gpu_num+gpu_no, suffix='anchor')
                hint_mask = torch.sum(hint_mask, dim=1, keepdim=True).view(N,1,16,16)
            
            hint_maps = basic.upfeat(hint_mask, affinity_map, sp_size, sp_size)
            pred_colors = ref_logit if meta_dict['hint2regress'] else color_class.decode_ind2ab(ref_logit, T=0)
            pred_colors = basic.upfeat(pred_colors, affinity_map, sp_size, sp_size)
            marked_labs = basic.mark_color_hints(input_grays, input_colors, hint_maps, base_ABs=pred_colors)
            hint_imgs = basic.tensor2array(marked_labs)
            util.save_normLabs_from_batch(hint_imgs, meta_dict['img_dir'], None, batch_idx*gpu_num+gpu_no, suffix='hint')

            if meta_dict['enhanced']:
                recon_labs = torch.cat((input_grays,pred_AB), dim=1)
                recon_lab_imgs = basic.tensor2array(recon_labs)
                util.save_normLabs_from_batch(recon_lab_imgs, meta_dict['img_dir'], None, batch_idx*gpu_num+gpu_no, suffix='enhanced')

    #pdb.set_trace()
    ## epoch summary: plot and print
    if is_ddp:
        total_loss = mean_reduce_tensor(total_loss)
    if gpu_no == 0:
        total_loss = total_loss / (batch_idx+1)
        val_plotter.add_data('val/totalLoss', total_loss.item(), epoch)
        logger.info("--- epoch: %d/%d | val  : %4.4f | duration: %4.2f ---" % \
                (epoch+1, n_epochs, total_loss.item(), (time.time() - start_time)))
        
    return total_loss.item()
 

def dataset_info_from_argument(args):
    dataset_info = dict()
    dataset_info['dataset'], dataset_info['data_dir'] = args.dataset, args.data_dir
    dataset_info['input_size'], dataset_info['crop_size'] = args.input_dim, args.image_dim
    dataset_info['batch_size'], dataset_info['num_workers'] = args.batch_size, args.workers
    return dataset_info


def pack_meta_data(args, img_dir, gpu_no):
    meta_dict = dict()
    meta_dict['img_dir'], meta_dict['n_epochs'], meta_dict['sp_size'] = img_dir, args.epochs, args.psize
    ## color class mapper
    coeff = 1.0 - args.colorfulness
    meta_dict['color_class'] = basic.ColorLabel(lambda_=coeff, device=torch.device("cuda:%d"%gpu_no))
    meta_dict['hint2regress'] = args.hint2regress
    meta_dict['enhanced'] = args.enhanced
    ## content-agnostic tensors: (1,C,H,W)
    spixel_ids, coord_feat = basic.init_spixel_grid(256, 256, spixel_size=args.psize)
    meta_dict['spixel_ids'] = spixel_ids.unsqueeze(0).cuda()
    meta_dict['coord_feat'] = coord_feat.unsqueeze(0).cuda()
    return meta_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pcolor_argparser(parser)
    args = parser.parse_args()
    set_random_seed(args.seed)
    gpu_num = torch.cuda.device_count()
    rank = 0
    train_model(args, gpu_num, rank, is_ddp=False)
