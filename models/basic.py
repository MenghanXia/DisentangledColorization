import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torch.autograd import Function
from utils import util, cielab
import cv2, math, random

def tensor2array(tensors):
    arrays = tensors.detach().to("cpu").numpy()
    return np.transpose(arrays, (0, 2, 3, 1))


def rgb2gray(color_batch):
    #! gray = 0.299*R+0.587*G+0.114*B
    gray_batch = color_batch[:, 0, ...] * 0.299 + color_batch[:, 1, ...] * 0.587 + color_batch[:, 2, ...] * 0.114
    gray_batch = gray_batch.unsqueeze_(1)
    return gray_batch


def getParamsAmount(model):
    params = list(model.parameters())
    count = 0
    for var in params:
        l = 1
        for j in var.size():
            l *= j
        count += l
    return count


def checkAverageGradient(model):
    meanGrad, cnt = 0.0, 0
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            meanGrad += torch.mean(torch.abs(parms.grad))
            cnt += 1
    return meanGrad.item() / cnt


def get_random_mask(N, H, W, minNum, maxNum):
    binary_maps = np.zeros((N, H*W), np.float32)
    for i in range(N):
        locs = random.sample(range(0, H*W), random.randint(minNum,maxNum))
        binary_maps[i, locs] = 1
    return binary_maps.reshape(N,1,H,W)


def io_user_control(hint_mask, spix_colors, output=True):
    cache_dir = '/apdcephfs/private_richardxia'
    if output:
        print('--- data saving')
        mask_imgs = tensor2array(hint_mask) * 2.0 - 1.0
        util.save_images_from_batch(mask_imgs, cache_dir, ['mask.png'], -1)
        fake_gray = torch.zeros_like(spix_colors[:,[0],:,:])
        spix_labs = torch.cat((fake_gray,spix_colors), dim=1)
        spix_imgs = tensor2array(spix_labs)
        util.save_normLabs_from_batch(spix_imgs, cache_dir, ['color.png'], -1)
        return hint_mask, spix_colors
    else:
        print('--- data loading')
        mask_img = cv2.imread(cache_dir+'/mask.png', cv2.IMREAD_GRAYSCALE)
        mask_img = np.expand_dims(mask_img, axis=2) / 255.
        hint_mask = torch.from_numpy(mask_img.transpose((2, 0, 1)))
        hint_mask = hint_mask.unsqueeze(0).cuda()
        bgr_img = cv2.imread(cache_dir+'/color.png', cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_img = np.array(rgb_img / 255., np.float32)
        lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        lab_img = torch.from_numpy(lab_img.transpose((2, 0, 1)))
        ab_chans = lab_img[1:3,:,:] / 110.
        spix_colors = ab_chans.unsqueeze(0).cuda()
        return hint_mask.float(), spix_colors.float()


class Quantize(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x.round()
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        inputX = ctx.saved_tensors
        return grad_output


def mark_color_hints(input_grays, target_ABs, gate_maps, kernel_size=3, base_ABs=None):
    ## to highlight the seeds with 1-pixel margin
    binary_map = torch.where(gate_maps>0.7, torch.ones_like(gate_maps), torch.zeros_like(gate_maps))
    center_mask = dilate_seeds(binary_map, kernel_size=kernel_size)
    margin_mask = dilate_seeds(binary_map, kernel_size=kernel_size+2) - center_mask
    ## drop colors
    dilated_seeds = dilate_seeds(gate_maps, kernel_size=kernel_size+2)
    marked_grays = torch.where(margin_mask > 1e-5, torch.ones_like(gate_maps), input_grays)
    if base_ABs is None:
        marked_ABs = torch.where(center_mask < 1e-5, torch.zeros_like(gate_maps), target_ABs)
    else:
        marked_ABs = torch.where(margin_mask > 1e-5, torch.zeros_like(gate_maps), base_ABs)
        marked_ABs = torch.where(center_mask > 1e-5, target_ABs, marked_ABs)
    return torch.cat((marked_grays,marked_ABs), dim=1)

def dilate_seeds(gate_maps, kernel_size=3):
    N,C,H,W = gate_maps.shape
    input_unf = F.unfold(gate_maps, kernel_size, padding=kernel_size//2)
    #! Notice: differentiable? just like max pooling?
    dilated_seeds, _ = torch.max(input_unf, dim=1, keepdim=True)
    output = F.fold(dilated_seeds, output_size=(H,W), kernel_size=1)
    #print('-------', input_unf.shape)
    return output


class RebalanceLoss(Function):
    @staticmethod
    def forward(ctx, data_input, weights):
        ctx.save_for_backward(weights)
        return data_input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        weights, = ctx.saved_tensors
        # reweigh gradient pixelwise so that rare colors get a chance to
        # contribute
        grad_input = grad_output * weights
        # second return value is None since we are not interested in the
        # gradient with respect to the weights
        return grad_input, None


class GetClassWeights:
    def __init__(self, cielab, lambda_=0.5, device='cuda'):
        prior = torch.from_numpy(cielab.gamut.prior).cuda()
        uniform = torch.zeros_like(prior)
        uniform[prior > 0] = 1 / (prior > 0).sum().type_as(uniform)
        self.weights = 1 / ((1 - lambda_) * prior + lambda_ * uniform)
        self.weights /= torch.sum(prior * self.weights)

    def __call__(self, ab_actual):
        return self.weights[ab_actual.argmax(dim=1, keepdim=True)]


class ColorLabel:
    def __init__(self, lambda_=0.5, device='cuda'):
        self.cielab = cielab.CIELAB()
        self.q_to_ab = torch.from_numpy(self.cielab.q_to_ab).to(device)
        prior = torch.from_numpy(self.cielab.gamut.prior).to(device)
        uniform = torch.zeros_like(prior)
        uniform[prior>0] = 1 / (prior>0).sum().type_as(uniform)
        self.weights = 1 / ((1-lambda_) * prior + lambda_ * uniform)
        self.weights /= torch.sum(prior * self.weights)

    def visualize_label(self, step=3):
        height, width = 200, 313*step
        label_lab = np.ones((height,width,3), np.float32)
        for x in range(313):
            ab = self.cielab.q_to_ab[x,:]
            label_lab[:,step*x:step*(x+1),1:] = ab / 110.
        label_lab[:,:,0] = np.zeros((height,width), np.float32)
        return label_lab

    @staticmethod
    def _gauss_eval(x, mu, sigma):
        norm = 1 / (2 * math.pi * sigma)
        return norm * torch.exp(-torch.sum((x - mu)**2, dim=0) / (2 * sigma**2))
    
    def get_classweights(self, batch_gt_indx):
        #return self.weights[batch_gt_q.argmax(dim=1, keepdim=True)]
        return self.weights[batch_gt_indx]

    def encode_ab2ind(self, batch_ab, neighbours=5, sigma=5.0):
        batch_ab = batch_ab * 110.
        n, _, h, w = batch_ab.shape
        m = n * h * w
        # find nearest neighbours
        ab_ = batch_ab.permute(1, 0, 2, 3).reshape(2, -1) # (2, n*h*w) 
        cdist = torch.cdist(self.q_to_ab, ab_.t())
        nns = cdist.argsort(dim=0)[:neighbours, :]
        # gaussian weighting
        nn_gauss = batch_ab.new_zeros(neighbours, m)
        for i in range(neighbours):
            nn_gauss[i, :] = self._gauss_eval(self.q_to_ab[nns[i, :], :].t(), ab_, sigma)
        nn_gauss /= nn_gauss.sum(dim=0, keepdim=True)
        # expand
        bins = self.cielab.gamut.EXPECTED_SIZE
        q = batch_ab.new_zeros(bins, m)
        q[nns, torch.arange(m).repeat(neighbours, 1)] = nn_gauss        
        return q.reshape(bins, n, h, w).permute(1, 0, 2, 3)

    def decode_ind2ab(self, batch_q, T=0.38):
        _, _, h, w = batch_q.shape
        batch_q = F.softmax(batch_q, dim=1)
        if T%1 == 0:
            # take the T-st probable index
            sorted_probs, batch_indexs = torch.sort(batch_q, dim=1, descending=True)
            #print('checking [index]', batch_indexs[:,0:5,5,5])
            #print('checking [probs]', sorted_probs[:,0:5,5,5])
            batch_indexs = batch_indexs[:,T:T+1,:,:]
            #batch_indexs = torch.where(sorted_probs[:,T:T+1,:,:] > 0.25, batch_indexs[:,T:T+1,:,:], batch_indexs[:,0:1,:,:])
            ab = torch.stack([
                self.q_to_ab.index_select(0, q_i.flatten()).reshape(h,w,2).permute(2,0,1)
                for q_i in batch_indexs])
        else:
            batch_q = torch.exp(batch_q / T)
            batch_q /= batch_q.sum(dim=1, keepdim=True)
            a = torch.tensordot(batch_q, self.q_to_ab[:,0], dims=((1,), (0,)))
            a = a.unsqueeze(dim=1)
            b = torch.tensordot(batch_q, self.q_to_ab[:,1], dims=((1,), (0,)))
            b = b.unsqueeze(dim=1)
            ab = torch.cat((a, b), dim=1)
        ab = ab / 110.
        return ab.type(batch_q.dtype)


def init_spixel_grid(img_height, img_width, spixel_size=16):
    # get spixel id for the final assignment
    n_spixl_h = int(np.floor(img_height/spixel_size))
    n_spixl_w = int(np.floor(img_width/spixel_size))
    spixel_height = int(img_height / (1. * n_spixl_h))
    spixel_width = int(img_width / (1. * n_spixl_w))
    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    
    def shift9pos(input, h_shift_unit=1, w_shift_unit=1):
        # input should be padding as (c, 1+ height+1, 1+width+1)
        input_pd = np.pad(input, ((h_shift_unit, h_shift_unit), (w_shift_unit, w_shift_unit)), mode='edge')
        input_pd = np.expand_dims(input_pd, axis=0)
        # assign to ...
        top     = input_pd[:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
        bottom  = input_pd[:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
        left    = input_pd[:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
        right   = input_pd[:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]
        center = input_pd[:,h_shift_unit:-h_shift_unit,w_shift_unit:-w_shift_unit]
        bottom_right    = input_pd[:, 2 * h_shift_unit:,   2 * w_shift_unit:]
        bottom_left     = input_pd[:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
        top_right       = input_pd[:, :-2 * h_shift_unit,  2 * w_shift_unit:]
        top_left        = input_pd[:, :-2 * h_shift_unit,  :-2 * w_shift_unit]
        shift_tensor = np.concatenate([     top_left,    top,      top_right,
                                            left,        center,      right,
                                            bottom_left, bottom,    bottom_right], axis=0)
        return shift_tensor

    spix_idx_tensor_ = shift9pos(spix_values)
    spix_idx_tensor = np.repeat(
        np.repeat(spix_idx_tensor_, spixel_height, axis=1), spixel_width, axis=2)
    spixel_id_tensor = torch.from_numpy(spix_idx_tensor).type(torch.float)

    #! pixel coord feature maps
    all_h_coords = np.arange(0, img_height, 1)
    all_w_coords = np.arange(0, img_width, 1)
    curr_pxl_coord = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing='ij'))
    coord_feat_tensor = np.concatenate([curr_pxl_coord[1:2, :, :], curr_pxl_coord[:1, :, :]])
    coord_feat_tensor = torch.from_numpy(coord_feat_tensor).type(torch.float)

    return spixel_id_tensor, coord_feat_tensor


def split_spixels(assign_map, spixel_ids):
    N,C,H,W = assign_map.shape
    spixel_id_map = spixel_ids.expand(N,-1,-1,-1)
    assig_max,_ = torch.max(assign_map, dim=1, keepdim=True)
    assignment_ = torch.where(assign_map == assig_max, torch.ones(assign_map.shape).cuda(),torch.zeros(assign_map.shape).cuda())
    ## winner take all
    new_spixl_map_ = spixel_id_map * assignment_
    new_spixl_map = torch.sum(new_spixl_map_,dim=1,keepdim=True).type(torch.int)
    return new_spixl_map


def poolfeat(input, prob, sp_h=2, sp_w=2, need_entry_prob=False):
    def feat_prob_sum(feat_sum, prob_sum, shift_feat):
        feat_sum += shift_feat[:, :-1, :, :]
        prob_sum += shift_feat[:, -1:, :, :]
        return feat_sum, prob_sum

    b, _, h, w = input.shape
    h_shift_unit = 1
    w_shift_unit = 1
    p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)
    feat_ = torch.cat([input, torch.ones([b, 1, h, w]).cuda()], dim=1)  # b* (n+1) *h*w
    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 0, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w)) # b * (n+1) * h* w
    send_to_top_left =  F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:, 2 * w_shift_unit:]
    feat_sum = send_to_top_left[:, :-1, :, :].clone()
    prob_sum = send_to_top_left[:, -1:, :, :].clone()

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 1, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 2, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top_right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 3, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 4, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    center = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, center)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 5, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    right = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 6, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, :-2 * h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 7, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 8, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_right)
    pooled_feat = feat_sum / (prob_sum + 1e-8)
    if need_entry_prob:
        return pooled_feat, prob_sum
    return pooled_feat


def get_spixel_size(affinity_map, sp_h=2, sp_w=2, elem_thres=25):
    N,C,H,W = affinity_map.shape
    assign_max,_ = torch.max(affinity_map, dim=1, keepdim=True)
    assign_map = torch.where(affinity_map==assign_max, torch.ones(affinity_map.shape).cuda(),torch.zeros(affinity_map.shape).cuda())
    ## one_map = (N,1,H,W)
    _, elem_num_maps = poolfeat(torch.ones(assign_max.shape).cuda(), assign_map, sp_h, sp_w, True)
    #all_one_map = torch.ones(elem_num_maps.shape).cuda()
    #empty_mask = torch.where(elem_num_maps < elem_thres/256, all_one_map, 1-all_one_map)
    return elem_num_maps
    

def upfeat(input, prob, up_h=2, up_w=2):
    # input b*n*H*W  downsampled
    # prob b*9*h*w
    b, c, h, w = input.shape

    h_shift = 1
    w_shift = 1

    p2d = (w_shift, w_shift, h_shift, h_shift)
    feat_pd = F.pad(input, p2d, mode='constant', value=0)

    gt_frm_top_left = F.interpolate(feat_pd[:, :, :-2 * h_shift, :-2 * w_shift], size=(h * up_h, w * up_w),mode='nearest')
    feat_sum = gt_frm_top_left * prob.narrow(1,0,1)

    top = F.interpolate(feat_pd[:, :, :-2 * h_shift, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top * prob.narrow(1, 1, 1)

    top_right = F.interpolate(feat_pd[:, :, :-2 * h_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top_right * prob.narrow(1,2,1)

    left = F.interpolate(feat_pd[:, :, h_shift:-w_shift, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += left * prob.narrow(1, 3, 1)

    center = F.interpolate(input, (h * up_h, w * up_w), mode='nearest')
    feat_sum += center * prob.narrow(1, 4, 1)

    right = F.interpolate(feat_pd[:, :, h_shift:-w_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += right * prob.narrow(1, 5, 1)

    bottom_left = F.interpolate(feat_pd[:, :, 2 * h_shift:, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_left * prob.narrow(1, 6, 1)

    bottom = F.interpolate(feat_pd[:, :, 2 * h_shift:, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom * prob.narrow(1, 7, 1)

    bottom_right =  F.interpolate(feat_pd[:, :, 2 * h_shift:, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_right * prob.narrow(1, 8, 1)

    return feat_sum


def suck_and_spread(self, base_maps, seg_layers):
    N,S,H,W = seg_layers.shape
    base_maps = base_maps.unsqueeze(1)
    seg_layers = seg_layers.unsqueeze(2)
    ## (N,S,C,1,1) = (N,1,C,H,W) * (N,S,1,H,W)
    mean_val_layers = (base_maps * seg_layers).sum(dim=(3,4), keepdim=True) / (1e-5 + seg_layers.sum(dim=(3,4), keepdim=True))
    ## normalized to be sum one
    weight_layers = seg_layers / (1e-5 + torch.sum(seg_layers, dim=1, keepdim=True))
    ## (N,S,C,H,W) = (N,S,C,1,1) * (N,S,1,H,W)
    recon_maps = mean_val_layers * weight_layers
    return recon_maps.sum(dim=1)


#! copy from Richard Zhang [SIGGRAPH2017]
# RGB grid points maps to Lab range: L[0,100], a[-86.183,98,233], b[-107.857,94.478]
#------------------------------------------------------------------------------
def rgb2xyz(rgb):  # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        #  [0.212671, 0.715160, 0.072169],
        #  [0.019334, 0.119193, 0.950227]])
    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()
    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)
    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)
    return out

def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])
    r = 3.24048134*xyz[:,0,:,:]-1.53715152*xyz[:,1,:,:]-0.49853633*xyz[:,2,:,:]
    g = -0.96925495*xyz[:,0,:,:]+1.87599*xyz[:,1,:,:]+.04155593*xyz[:,2,:,:]
    b = .05564664*xyz[:,0,:,:]-.20404134*xyz[:,1,:,:]+1.05731107*xyz[:,2,:,:]
    rgb = torch.cat((r[:,None,:,:],g[:,None,:,:],b[:,None,:,:]),dim=1)
    #！ sometimes reaches a small negative number, which causes NaNs
    rgb = torch.max(rgb,torch.zeros_like(rgb))
    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()
    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)
    return rgb

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()
    xyz_scale = xyz/sc
    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()
    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)
    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)
    return out

def lab2xyz(lab):
    y_int = (lab[:,0,:,:]+16.)/116.
    x_int = (lab[:,1,:,:]/500.) + y_int
    z_int = y_int - (lab[:,2,:,:]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)
    out = torch.cat((x_int[:,None,:,:],y_int[:,None,:,:],z_int[:,None,:,:]),dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()
    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    sc = sc.to(out.device)
    out = out*sc
    return out

def rgb2lab(rgb, l_mean=50, l_norm=50, ab_norm=110):
    #! input rgb: [0,1]
    #! output lab: [-1,1]
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:,[0],:,:]-l_mean) / l_norm
    ab_rs = lab[:,1:,:,:] / ab_norm
    out = torch.cat((l_rs,ab_rs),dim=1)
    return out
    
def lab2rgb(lab_rs, l_mean=50, l_norm=50, ab_norm=110):
    #! input lab: [-1,1]
    #! output rgb: [0,1]
    l_ = lab_rs[:,[0],:,:] * l_norm + l_mean
    ab = lab_rs[:,1:,:,:] * ab_norm
    lab = torch.cat((l_,ab), dim=1)
    out = xyz2rgb(lab2xyz(lab))
    return out


if __name__ == '__main__':
    minL, minA, minB = 999., 999., 999.
    maxL, maxA, maxB = 0., 0., 0.
    for r in range(256):
        print('h',r)
        for g in range(256):
            for b in range(256):
                rgb = np.array([r,g,b], np.float32).reshape(1,1,-1) / 255.0
                #lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
                rgb = torch.from_numpy(rgb.transpose((2, 0, 1)))
                rgb = rgb.reshape(1,3,1,1)
                lab = rgb2lab(rgb)
                lab[:,[0],:,:] = lab[:,[0],:,:] * 50 + 50
                lab[:,1:,:,:] = lab[:,1:,:,:] * 110
                lab = lab.squeeze()
                lab_float = lab.numpy()
                #print('zhang vs. cv2:', lab_float, lab_img.squeeze())
                minL = min(lab_float[0], minL)
                minA = min(lab_float[1], minA)
                minB = min(lab_float[2], minB)
                maxL = max(lab_float[0], maxL)
                maxA = max(lab_float[1], maxA)
                maxB = max(lab_float[2], maxB)
    print('L:', minL, maxL)
    print('A:', minA, maxA)
    print('B:', minB, maxB)