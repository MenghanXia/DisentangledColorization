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

os.chdir(sys.path[0])
sys.path.append("..")
import _init_paths
from utils_train import load_checkpoint
import model, basic
import util
import clusterkit
from skimage import segmentation, color


def fetch_data(img_path, org_size=True):
    bgr_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    H, W = rgb_img.shape[:2]
    if org_size:
        scale = 16
        if H % scale != 0 or W % scale != 0:
            ## H,W,C
            rgb_img = np.pad(rgb_img, ((0, scale - H % scale), (0, scale - W % scale), (0, 0)), mode='edge')
    else:
        rgb_img = cv2.resize(rgb_img, (256,256), interpolation=cv2.INTER_LINEAR)

    rgb_img = np.array(rgb_img / 255., np.float32)
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    lab_img = torch.from_numpy(lab_img.transpose((2, 0, 1)))
    rgb_img = torch.from_numpy(rgb_img.transpose((2, 0, 1)))
    gray_img = (lab_img[0:1,:,:]-50.) / 50.
    ab_chans = lab_img[1:3,:,:] / 110.
    rgb_img = rgb_img*2. - 1.
    return gray_img.unsqueeze(0), ab_chans.unsqueeze(0), rgb_img.unsqueeze(0), (H,W)


def load_filelist(file_path, root_dir):
    file_list = []
    with open(file_path, "r") as fin:
        for line in fin:
            splits = line.strip().split()
            fname = splits[0]
            label = int(splits[1])
            file_list.append(os.path.join(root_dir, fname))
    return file_list


def test_model(data_dir, model_name, sp_size, checkpt_path, name, seq_model_args=None):
    print('@Inference: [%s] (spixel-size=%d)'%(model_name, sp_size))
    np.random.seed(seq_model_args.seed)
    torch.manual_seed(seq_model_args.seed)
    torch.cuda.manual_seed(seq_model_args.seed)
    root_dir = os.path.abspath(os.path.join(checkpt_path, '..', '..'))
    save_dir = os.path.join(root_dir, name+'-anchor%d'%args.n_clusters)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    img_list = util.get_filelist(data_dir)
    print('-data dir (%d images):%s'%(len(img_list),data_dir))
    print('-saving dir:%s'%save_dir)
    
    ## MODEL >>>>>>>>>>>>>>>
    color_model = eval('model.'+args.model)(inChannel=1, outChannel=313, sp_size=args.psize, d_model=args.d_model,\
                                            use_dense_pos=args.dense_pos, spix_pos=args.spix_pos, learning_pos=args.learning_pos,\
                                            n_clusters=args.n_clusters, random_hint=args.random_hint, \
                                            hint2regress=args.hint2regress, enhanced=True)
        
    is_multi_gpus = torch.cuda.device_count() > 1
    if is_multi_gpus:
        color_model = torch.nn.DataParallel(color_model).cuda()
        model_without_dp = color_model.module
    else:
        color_model = color_model.cuda()
        model_without_dp = color_model
    ## load weight
    assert os.path.exists(checkpt_path)
    load_checkpoint(checkpt_path, model_without_dp)
    print('-weight loaded successfully.')

    ## PROCESS >>>>>>>>>>>>>>>
    color_model.eval()
    color_class = basic.ColorLabel()

    start_time = time.time()
    for nn, img_pth in enumerate(img_list):
        _, file_name = os.path.split(img_pth)
        print('-processing %s ...' % file_name)
        ## save as png format
        filename, ext = os.path.splitext(file_name)
        file_name = "%s.png"%filename
        ## load data
        input_grays, input_colors, input_RGBs, (H,W) = fetch_data(img_pth, args.no_resize)
        input_grays = input_grays.cuda(non_blocking=True)
        input_colors = input_colors.cuda(non_blocking=True)
        input_RGBs = input_RGBs.cuda(non_blocking=True)
        
        ## inference
        is_test_mode = True
        sampled_T = 2 if args.diverse else 0
        pal_logit, ref_logit, enhanced_ab, affinity_map, spix_colors, hint_mask = color_model(input_grays, \
                                                            input_colors, is_test_mode, sampled_T)
        pred_probs = pal_logit
        if args.hint2regress:
            guided_colors = ref_logit
        else:
            guided_colors = color_class.decode_ind2ab(ref_logit, T=0)
        guided_colors = basic.upfeat(guided_colors, affinity_map, sp_size, sp_size)
        if args.diverse:
            for no in range(3):
                pred_labs = torch.cat((input_grays,enhanced_ab[no:no+1,:,:,:]), dim=1)
                lab_imgs = basic.tensor2array(pred_labs)
                lab_imgs = batch_depadding(lab_imgs, H, W, args)
                util.save_normLabs_from_batch(lab_imgs, save_dir, [file_name], -1, suffix='c%d'%no)
        else:
            pred_labs = torch.cat((input_grays,enhanced_ab), dim=1)
            lab_imgs = basic.tensor2array(pred_labs)
            lab_imgs = batch_depadding(lab_imgs, H, W, args)
            util.save_normLabs_from_batch(lab_imgs, save_dir, [file_name], -1)#, suffix='enhanced')

            ## visualize anchor locations
            anchor_masks = basic.upfeat(hint_mask, affinity_map, sp_size, sp_size)
            marked_labs = basic.mark_color_hints(input_grays, enhanced_ab, anchor_masks, base_ABs=enhanced_ab)
            hint_imgs = basic.tensor2array(marked_labs)
            hint_imgs = batch_depadding(hint_imgs, H, W, args)
            #util.save_normLabs_from_batch(hint_imgs, save_dir, [file_name], -1, suffix='anchors')

    print("-processed %d imgs. consumed %f sec" % (nn+1, (time.time() - start_time)))


def batch_depadding(lab_img, H, W, args):
    return lab_img[:,:H,:W,:] if args.no_resize else lab_img 


if __name__ == '__main__':
    print("FLAG: %s" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='result', help='save dir name')
    parser.add_argument('--seed', default='130', type=int, help='random seed')
    parser.add_argument('--psize', default='16', type=int, help='super-pixel size')
    parser.add_argument('--data', type=str, default='../../../0DataZoo/Dataset_C/VOC2012/Val/target', help='path of images')
    parser.add_argument('--model', type=str, default='AnchorColorProb', help='which model to use')
    parser.add_argument('--checkpt', type=str, default='../../Saved/colorProb/checkpts/model_last.pth.tar', help='path of weight')
    ## exclusive options
    parser.add_argument('--n_enc', default=3, type=int, help='number of encoder layers')
    parser.add_argument('--n_dec', default=6, type=int, help='number of decoder layers')
    parser.add_argument('--d_model', default=64, type=int, help='feature dimension of transformer')
    parser.add_argument('--dense_pos', action='store_true', default=False, help='use pos encoding at each SA block')
    parser.add_argument('--spix_pos', action='store_true', default=False, help='use pos of spixel centroid')
    parser.add_argument('--learning_pos', action='store_true', default=False, help='learnable pos embedding')
    parser.add_argument('--hint2regress', action='store_true', default=False, help='predict ab values from hint')
    parser.add_argument('--n_clusters', default=8, type=int, help='number of color clusters')
    parser.add_argument('--random_hint', action='store_true', default=False, help='sample anchors randomly')
    parser.add_argument('--no_resize', action='store_true', default=False, help='input the original resolution')
    parser.add_argument('--diverse', action='store_true', default=False, help='use pixel-level enhancement or not')

    args = parser.parse_args()
    args.dense_pos = True
    test_model(args.data, args.model, args.psize, args.checkpt, args.name, args)