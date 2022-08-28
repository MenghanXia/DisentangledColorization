import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import basic, clusterkit
import pdb

class AnchorAnalysis:
    def __init__(self, mode, colorLabeler):
        ## anchor generating mode: 1.random; 2.clustering
        self.mode = mode
        self.colorLabeler = colorLabeler

    def _detect_correlation(self, data_tensors, color_probs, hint_masks, thres=0.1):
        N,C,H,W = data_tensors.shape
        ## (N,C,HW)
        data_vecs = data_tensors.flatten(2)
        prob_vecs = color_probs.flatten(2)
        mask_vecs = hint_masks.flatten(2)
        #anchor_data = torch.masked_select(data_vecs, mask_vecs.bool()).view(N,C,-1)
        #anchor_prob = torch.masked_select(prob_vecs, mask_vecs.bool()).view(N,313,-1)
        #_,_,K = anchor_data.shape
        anchor_mask = torch.matmul(mask_vecs.permute(0,2,1), mask_vecs)
        cosine_sim = True
        ## non-similarity matrix
        if cosine_sim:
            norm_data = F.normalize(data_vecs, p=2, dim=1)
            ## (N,HW,HW) = (N,HW,C) X (N,C,HW)
            corr_matrix = torch.matmul(norm_data.permute(0,2,1), norm_data)
            ## remapping: [-1.0,1.0] to [0.0,1.0], and convert into dis-similarity
            dist_matrix = 1.0 - 0.5*(corr_matrix + 1.0)
        else:
            ## (N,HW,HW) = (N,HW,C) X (N,C,HW)
            XtX = torch.matmul(data_vecs.permute(0,2,1), data_vecs)
            diag_vec = torch.diagonal(XtX, dim1=-2, dim2=-1)
            A = diag_vec.unsqueeze(1).repeat(1,H*W,1)
            At = diag_vec.unsqueeze(2).repeat(1,1,H*W)
            dist_matrix = A - 2*XtX + At
        #dist_matrix = dist_matrix + 1e7*torch.eye(K).to(data_tensors.device).repeat(N,1,1)
        ## for debug use
        K = 8
        anchor_adj_matrix = torch.masked_select(dist_matrix, anchor_mask.bool()).view(N,K,K)
        ## dectect connected nodes
        adj_matrix = torch.where((dist_matrix < thres) & (anchor_mask > 0), torch.ones_like(dist_matrix), torch.zeros_like(dist_matrix))
        adj_matrix = torch.matmul(adj_matrix, adj_matrix)
        adj_matrix = adj_matrix / (1e-7+adj_matrix)
        ## merge nodes
        ## (N,K,C) = (N,K,K) X (N,K,C)
        anchor_prob = torch.matmul(adj_matrix, prob_vecs.permute(0,2,1)) / torch.sum(adj_matrix, dim=2, keepdim=True)
        updated_prob_vecs = anchor_prob.permute(0,2,1) * mask_vecs + (1-mask_vecs) * prob_vecs
        color_probs = updated_prob_vecs.view(N,313,H,W)
        return color_probs, anchor_adj_matrix

    def _sample_anchor_colors(self, pred_prob, hint_mask, T=0):
        N,C,H,W = pred_prob.shape
        topk = 10
        assert T < topk
        sorted_probs, batch_indexs = torch.sort(pred_prob, dim=1, descending=True)
        ## (N,topk,H,W,1)
        topk_probs = torch.softmax(sorted_probs[:,:topk,:,:], dim=1).unsqueeze(4)
        topk_indexs = batch_indexs[:,:topk,:,:]
        topk_ABs = torch.stack([self.colorLabeler.q_to_ab.index_select(0, q_i.flatten()).reshape(topk,H,W,2)
                    for q_i in topk_indexs])
        ## (N,topk,H,W,2)
        topk_ABs = topk_ABs / 110.0
        ## choose the most distinctive 3 colors for each anchor
        if T == 0:
            sampled_ABs = topk_ABs[:,0,:,:,:]
        elif T == 1:
            sampled_AB0 = topk_ABs[:,[0],:,:,:]
            internal_diff = torch.norm(topk_ABs-sampled_AB0, p=2, dim=4, keepdim=True)
            _, batch_indexs = torch.sort(internal_diff, dim=1, descending=True)
            ## (N,1,H,W,2)
            selected_index = batch_indexs[:,[0],:,:,:].expand([-1,-1,-1,-1,2])
            sampled_ABs = torch.gather(topk_ABs, 1, selected_index)
            sampled_ABs = sampled_ABs.squeeze(1)
        else:
            sampled_AB0 = topk_ABs[:,[0],:,:,:]
            internal_diff = torch.norm(topk_ABs-sampled_AB0, p=2, dim=4, keepdim=True)
            _, batch_indexs = torch.sort(internal_diff, dim=1, descending=True)
            selected_index = batch_indexs[:,[0],:,:,:].expand([-1,-1,-1,-1,2])
            sampled_AB1 = torch.gather(topk_ABs, 1, selected_index)
            internal_diff2 = torch.norm(topk_ABs-sampled_AB1, p=2, dim=4, keepdim=True)
            _, batch_indexs = torch.sort(internal_diff+internal_diff2, dim=1, descending=True)
            ## (N,1,H,W,2)
            selected_index = batch_indexs[:,[T-2],:,:,:].expand([-1,-1,-1,-1,2])
            sampled_ABs = torch.gather(topk_ABs, 1, selected_index)
            sampled_ABs = sampled_ABs.squeeze(1)

        return sampled_ABs.permute(0,3,1,2)

    def __call__(self, data_tensors, n_anchors, spixel_sizes, use_sklearn_kmeans=False):
        N,C,H,W = data_tensors.shape
        if self.mode == 'clustering':
            ## clusters map: (N,K,H,W)
            cluster_mask = clusterkit.batch_kmeans_pytorch(data_tensors, n_anchors, 'euclidean', use_sklearn_kmeans)
            #noises = torch.rand(N,1,H,W).to(cluster_mask.device)
            perturb_factors = spixel_sizes
            cluster_prob = cluster_mask + perturb_factors * 0.01
            hint_mask_layers = F.one_hot(torch.argmax(cluster_prob.flatten(2), dim=-1), num_classes=H*W).float()
            hint_mask = torch.sum(hint_mask_layers, dim=1, keepdim=True).view(N,1,H,W)
        else:
            #print('----------hello, random!')
            cluster_mask = torch.zeros(N,n_anchors,H,W).to(data_tensors.device)
            binary_mask = basic.get_random_mask(N, H, W, minNum=n_anchors, maxNum=n_anchors)
            hint_mask = torch.from_numpy(binary_mask).to(data_tensors.device)
        return hint_mask, cluster_mask