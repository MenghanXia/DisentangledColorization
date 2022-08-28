import os, glob, sys, logging
import argparse, datetime, time
import random, pickle
from tqdm import tqdm 
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("..")
import _init_paths
from utils_train import load_checkpoint
import model, basic
import util


def fetch_data(img_path, need_padding=False):
    bgr_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if need_padding:
        scale = 8
        H, W = gray_img.shape[:2]
        if H % scale != 0 or W % scale != 0:
            bgr_img = np.pad(bgr_img, [0, scale - W % scale, 0, scale - H % scale], mode='reflect')

    bgr_img = np.array(bgr_img / 255., np.float32)
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    lab_img = torch.from_numpy(lab_img.transpose((2, 0, 1)))
    bgr_img = torch.from_numpy(bgr_img.transpose((2, 0, 1)))
    gray_img = (lab_img[0:1,:,:]-50.) / 50.
    color_map = lab_img[1:3,:,:] / 110.
    bgr_img = bgr_img*2. - 1.
    return gray_img.unsqueeze(0), color_map.unsqueeze(0), bgr_img.unsqueeze(0)


def test_model(data_dir, model_name, sp_size, checkpt_path, name):
    print('@Inference: [%s] (spixel-size=%d)'%(model_name, sp_size))
    root_dir = os.path.abspath(os.path.join(checkpt_path, '..', '..'))
    save_dir = os.path.join(root_dir, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('-loading dir:%s'%data_dir)
    print('-saving dir:%s'%save_dir)
    
    ## MODEL >>>>>>>>>>>>>>>
    spix_model = eval('model.'+model_name)(inChannel=1, outChannel=9, batchNorm=True)
    is_multi_gpus = torch.cuda.device_count() > 1
    if is_multi_gpus:
        spix_model = torch.nn.DataParallel(spix_model).cuda()
        model_without_dp = spix_model.module
    else:
        spix_model = spix_model.cuda()
        model_without_dp = spix_model
    ## load weight
    assert os.path.exists(checkpt_path)
    load_checkpoint(checkpt_path, model_without_dp)
    print('-weight loaded successfully.')

    ## build spixe grid
    spixel_ids, coord_feat = basic.init_spixel_grid(256, 256, spixel_size=sp_size)
    spixel_ids = spixel_ids.unsqueeze(0).cuda()

    ## PROCESS >>>>>>>>>>>>>>>
    def split_spixels(assign_map, spixel_ids):
        N,C,H,W = assign_map.shape
        spixel_id_map = spixel_ids.expand(N,-1,-1,-1)
        assig_max,_ = torch.max(assign_map, dim=1, keepdim=True)
        assignment_ = torch.where(assign_map == assig_max, torch.ones(assign_map.shape).cuda(), torch.zeros(assign_map.shape).cuda())
        ## winner take all
        new_spixl_map_ = spixel_id_map * assignment_
        new_spixl_map = torch.sum(new_spixl_map_,dim=1,keepdim=True).type(torch.int)
        return new_spixl_map

    img_list = util.get_filelist(data_dir)
    nn = 0
    start_time = time.time()
    for img_pth in img_list:
        _, file_name = os.path.split(img_pth)
        print('-processing %s ...' % file_name)

        ## load data
        input_gray, input_color, input_bgr = fetch_data(img_pth, False)
        input_gray = input_gray.cuda(non_blocking=True)
        input_color = input_color.cuda(non_blocking=True)
        input_bgr = input_bgr.cuda(non_blocking=True)

        ## inference
        pred_probs = spix_model(input_gray)
        pooled_bgr, spixel_votes = basic.poolfeat(input_bgr, pred_probs, sp_size, sp_size, True)
        #print('=========', torch.max(spixel_votes), torch.min(spixel_votes))
        #print('--------', spixel_votes.shape)
        empty_maps = torch.where(spixel_votes < 0.1, torch.ones(spixel_votes.shape).cuda(), torch.zeros(spixel_votes.shape).cuda())
        empty_maps = F.interpolate(empty_maps, scale_factor=16, mode='nearest')
        
        ## saving result
        pred_spixel_map = split_spixels(pred_probs, spixel_ids)
        spixel_maps = basic.tensor2array(pred_spixel_map)
        base_imgs = basic.tensor2array(input_bgr)
        #fake_gray = torch.zeros_like(input_gray)
        #show_rgb = basic.lab2rgb(torch.cat([fake_gray,input_color], dim=1))
        #base_imgs = basic.tensor2array(show_rgb)
        util.save_markedSP_from_batch(base_imgs, spixel_maps, save_dir, [file_name], -1)
        ## visualize spixel voting magnitute
        #vote_imgs = (basic.tensor2array(empty_maps) - 0.5) * 2.0
        #util.save_images_from_batch(vote_imgs, save_dir, [file_name], -1)
        ## reconstruction
        spix_color = basic.poolfeat(input_color, pred_probs, sp_size, sp_size)
        recon_color = basic.upfeat(spix_color, pred_probs, sp_size, sp_size)
        pred_lab = torch.cat((input_gray,recon_color), dim=1)
        lab_imgs = basic.tensor2array(pred_lab)
        util.save_normLabs_from_batch(lab_imgs, save_dir, [file_name], -1, suffix='recon')
        gray_imgs = basic.tensor2array(input_gray)
        util.save_images_from_batch(gray_imgs, save_dir, [file_name], -1, suffix='g')

        nn += 1
    print("-processed %d imgs. consumed %f sec" % (nn, (time.time() - start_time)))


if __name__ == '__main__':
    print("FLAG: %s" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='result', help='save dir name')
    parser.add_argument('--data', type=str, default='../../../0DataZoo/Dataset_C/VOC2012/Val/target', help='path of images')
    parser.add_argument('--psize', default='16', type=int, help='super-pixel size')
    parser.add_argument('--model', type=str, default='SpixelSeg', help='which model to use')
    parser.add_argument('--checkpt', type=str, default='../../Saved/spixG2C_16/checkpts/model_last.pth.tar', help='path of weight')
    args = parser.parse_args()
    test_model(args.data, args.model, args.psize, args.checkpt, args.name)