from __future__ import division
import os, glob, shutil, math, random, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import basic
from utils import util

eps = 0.0000001

class SPixelLoss:
    def __init__(self, psize=8, mpdist=False, gpu_no=0):
        self.mpdist = mpdist
        self.gpu_no = gpu_no
        self.sp_size = psize
    
    def __call__(self, data, epoch_no):
        kernel_size = self.sp_size
        #pos_weight = 0.003
        prob = data['pred_prob']
        labxy_feat = data['target_feat']
        N,C,H,W = labxy_feat.shape
        pooled_labxy = basic.poolfeat(labxy_feat, prob, kernel_size, kernel_size)
        reconstr_feat = basic.upfeat(pooled_labxy, prob, kernel_size, kernel_size)
        loss_map = reconstr_feat[:,:,:,:] - labxy_feat[:,:,:,:]
        featLoss_idx = torch.norm(loss_map[:,:-2,:,:], p=2, dim=1).mean()
        posLoss_idx = torch.norm(loss_map[:,-2:,:,:], p=2, dim=1).mean() / kernel_size
        totalLoss_idx = 10*featLoss_idx + 0.003*posLoss_idx
        return {'totalLoss':totalLoss_idx, 'featLoss':featLoss_idx, 'posLoss':posLoss_idx}


class AnchorColorProbLoss:
    def __init__(self, hint2regress=False, enhanced=False, with_grad=False, mpdist=False, gpu_no=0):
        self.mpdist = mpdist
        self.gpu_no = gpu_no
        self.hint2regress = hint2regress
        self.enhanced = enhanced
        self.with_grad = with_grad
        self.rebalance_gradient = basic.RebalanceLoss.apply
        self.entropy_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if self.enhanced:
            self.VGGLoss = VGG19Loss(gpu_no=gpu_no, is_ddp=mpdist)
    
    def _perceptual_loss(self, input_grays, input_colors, pred_colors):
        input_RGBs = basic.lab2rgb(torch.cat([input_grays,input_colors], dim=1))
        pred_RGBs = basic.lab2rgb(torch.cat([input_grays,pred_colors], dim=1))
        ## the output of "lab2rgb" just matches the input of "VGGLoss": [0,1]
        return self.VGGLoss(input_RGBs, pred_RGBs)
    
    def _laplace_gradient(self, pred_AB, target_AB):
        N,C,H,W = pred_AB.shape
        kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], device=pred_AB.get_device()).float()
        kernel = kernel.view(1, 1, *kernel.size()).repeat(C,1,1,1)
        grad_pred = F.conv2d(pred_AB, kernel, groups=C)
        grad_trg = F.conv2d(target_AB, kernel, groups=C)
        return l1_loss(grad_trg, grad_pred)
        
    def __call__(self, data, epoch_no):
        N,C,H,W = data['target_label'].shape
        pal_probs = self.rebalance_gradient(data['pal_prob'], data['class_weight'])
        #ref_probs = data['ref_prob']
        pal_probs = pal_probs.permute(0,2,3,1).contiguous().view(N*H*W, -1)
        gt_labels = data['target_label'].permute(0,2,3,1).contiguous().view(N*H*W, -1)
        '''        
        igored_mask = data['empty_entries'].permute(0,2,3,1).contiguous().view(N*H*W, -1)
        gt_labels[igored_mask] = -1
        gt_labels = gt_probs.squeeze()
        '''
        palLoss_idx = self.entropy_loss(pal_probs, gt_labels.squeeze(dim=1))
        if self.hint2regress: 
            ref_probs = data['ref_prob']
            refLoss_idx = 50 * l2_loss(data['spix_color'], ref_probs)
        else:
            ref_probs = self.rebalance_gradient(data['ref_prob'], data['class_weight'])
            ref_probs = ref_probs.permute(0,2,3,1).contiguous().view(N*H*W, -1)
            refLoss_idx = self.entropy_loss(ref_probs, gt_labels.squeeze(dim=1))
        reconLoss_idx = torch.zeros_like(palLoss_idx)
        if self.enhanced:
            scalar = 1.0 if self.hint2regress else 5.0
            reconLoss_idx = scalar * self._perceptual_loss(data['input_gray'], data['pred_color'], data['input_color'])
            if self.with_grad:
                gradient_loss = self._laplace_gradient(data['pred_color'], data['input_color'])
                reconLoss_idx += gradient_loss
        totalLoss_idx = palLoss_idx + refLoss_idx + reconLoss_idx
        #print("loss terms:", palLoss_idx.item(), refLoss_idx.item(), reconLoss_idx.item())
        return {'totalLoss':totalLoss_idx, 'palLoss':palLoss_idx, 'refLoss':refLoss_idx, 'recLoss':reconLoss_idx}


def compute_affinity_pos_loss(prob_in, labxy_feat, pos_weight=0.003, kernel_size=16):
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()
    N,C,H,W = labxy_feat.shape
    pooled_labxy = basic.poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = basic.upfeat(pooled_labxy, prob, kernel_size, kernel_size)
    loss_map = reconstr_feat[:,:,:,:] - labxy_feat[:,:,:,:]
    loss_feat = torch.norm(loss_map[:,:-2,:,:], p=2, dim=1).mean()
    loss_pos = torch.norm(loss_map[:,-2:,:,:], p=2, dim=1).mean() * m / S
    loss_affinity = loss_feat + loss_pos
    return loss_affinity


def l2_loss(y_input, y_target, weight_map=None):
    if weight_map is None:
        return F.mse_loss(y_input, y_target)
    else:
        diff_map = torch.mean(torch.abs(y_input-y_target), dim=1, keepdim=True)
        batch_dev = torch.sum(diff_map*diff_map*weight_map, dim=(1,2,3)) / (eps+torch.sum(weight_map, dim=(1,2,3)))
        return batch_dev.mean()


def l1_loss(y_input, y_target, weight_map=None):
    if weight_map is None:
        return F.l1_loss(y_input, y_target)
    else:
        diff_map = torch.mean(torch.abs(y_input-y_target), dim=1, keepdim=True)
        batch_dev = torch.sum(diff_map*weight_map, dim=(1,2,3)) / (eps+torch.sum(weight_map, dim=(1,2,3)))
        return batch_dev.mean()


def masked_l1_loss(y_input, y_target, outlier_mask):
    one = torch.tensor([1.0]).cuda(y_input.get_device())
    weight_map = torch.where(outlier_mask, one * 0.0, one * 1.0)
    return l1_loss(y_input, y_target, weight_map)


def huber_loss(y_input, y_target, delta=0.01):
    mask = torch.zeros_like(y_input)
    mann = torch.abs(y_input - y_target)
    eucl = 0.5 * (mann**2)
    mask[...] = mann < delta
    loss = eucl * mask / delta + (mann - 0.5 * delta) * (1 - mask)
    return torch.mean(loss)


## Perceptual loss that uses a pretrained VGG network
class VGG19Loss(nn.Module):
    def __init__(self, feat_type='liu', gpu_no=0, is_ddp=False, requires_grad=False):
        super(VGG19Loss, self).__init__()
        #os.environ['TORCH_HOME'] = '/apdcephfs/share_1290939/richardxia/Saved/Checkpoints/VGG19'
        ## data requirement: (N,C,H,W) in RGB format, [0,1] range, and resolution >= 224x224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.feat_type = feat_type

        vgg_model = torchvision.models.vgg19(pretrained=True)
        ## AssertionError: DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient
        '''
        if is_ddp:
            vgg_model = vgg_model.cuda(gpu_no)
            vgg_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(vgg_model)
            vgg_model = torch.nn.parallel.DistributedDataParallel(vgg_model, device_ids=[gpu_no], find_unused_parameters=True)
        else:
            vgg_model = vgg_model.cuda(gpu_no)
        '''
        vgg_model = vgg_model.cuda(gpu_no)
        if self.feat_type == 'liu':
            ## conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
            self.slice1 = nn.Sequential(*list(vgg_model.features)[:2]).eval()
            self.slice2 = nn.Sequential(*list(vgg_model.features)[2:7]).eval()
            self.slice3 = nn.Sequential(*list(vgg_model.features)[7:12]).eval()
            self.slice4 = nn.Sequential(*list(vgg_model.features)[12:21]).eval()
            self.slice5 = nn.Sequential(*list(vgg_model.features)[21:30]).eval()
            self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        elif self.feat_type == 'lei':
            ## conv1_2, conv2_2, conv3_2, conv4_2, conv5_2
            self.slice1 = nn.Sequential(*list(vgg_model.features)[:4]).eval()
            self.slice2 = nn.Sequential(*list(vgg_model.features)[4:9]).eval()
            self.slice3 = nn.Sequential(*list(vgg_model.features)[9:14]).eval()
            self.slice4 = nn.Sequential(*list(vgg_model.features)[14:23]).eval()
            self.slice5 = nn.Sequential(*list(vgg_model.features)[23:32]).eval()
            self.weights = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10.0/1.5]
        else:
            ## maxpool after conv4_4
            self.featureExactor = nn.Sequential(*list(vgg_model.features)[:28]).eval()
        '''
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), pretrained_features[x])
        '''
        self.criterion = nn.L1Loss()

        ## fixed parameters
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.eval()
        print('[*] VGG19Loss init!')

    def normalize(self, tensor):
        tensor = tensor.clone()
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor.device)
        tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return tensor

    def forward(self, x, y):
        ## x denotes the groundtruth; y denoets the prediction
        norm_x, norm_y = self.normalize(x), self.normalize(y)
        ## feature extract
        if self.feat_type == 'liu' or self.feat_type == 'lei':
            x_relu1, y_relu1 = self.slice1(norm_x), self.slice1(norm_y)
            x_relu2, y_relu2 = self.slice2(x_relu1), self.slice2(y_relu1)
            x_relu3, y_relu3 = self.slice3(x_relu2), self.slice3(y_relu2)
            x_relu4, y_relu4 = self.slice4(x_relu3), self.slice4(y_relu3)
            x_relu5, y_relu5 = self.slice5(x_relu4), self.slice5(y_relu4)
            x_vgg = [x_relu1, x_relu2, x_relu3, x_relu4, x_relu5]
            y_vgg = [y_relu1, y_relu2, y_relu3, y_relu4, y_relu5]
            loss = 0    
            for i in range(len(x_vgg)):
                loss += self.weights[i] * self.criterion(x_vgg[i].detach(), y_vgg[i])
        else:
            x_vgg, y_vgg = self.featureExactor(norm_x), self.featureExactor(norm_y)
            loss = self.criterion(x_vgg.detach(), y_vgg)
        return loss
