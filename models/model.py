import torch
import torch.nn as nn
import torch.nn.functional as F
from network import HourGlass2, SpixelNet, ColorProbNet
from transformer2d import EncoderLayer, DecoderLayer, TransformerEncoder, TransformerDecoder
from position_encoding import build_position_encoding
import basic, clusterkit, anchor_gen
from collections import OrderedDict
from utils import util, cielab


class SpixelSeg(nn.Module):
    def __init__(self, inChannel=1, outChannel=9, batchNorm=True):
        super(SpixelSeg, self).__init__()
        self.net = SpixelNet(inChannel=inChannel, outChannel=outChannel, batchNorm=batchNorm)

    def get_trainable_params(self, lr=1.0):
        #print('=> [optimizer] finetune backbone with smaller lr')
        params = []
        for name, param in self.named_parameters():
            if 'xxx' in name:
                params.append({'params': param, 'lr': lr})
            else:
                params.append({'params': param})
        return params

    def forward(self, input_grays):
        pred_probs = self.net(input_grays)
        return pred_probs


class AnchorColorProb(nn.Module):
    def __init__(self, inChannel=1, outChannel=313, sp_size=16, d_model=64, use_dense_pos=True, spix_pos=False, learning_pos=False, \
                n_clusters=8, random_hint=False, hint2regress=False, enhanced=False, use_mask=False, rank=0):
        super(AnchorColorProb, self).__init__()
        self.sp_size = sp_size
        self.spix_pos = spix_pos
        self.use_token_mask = use_mask
        self.hint2regress = hint2regress
        self.segnet = SpixelSeg(inChannel=1, outChannel=9, batchNorm=True)
        self.repnet = ColorProbNet(inChannel=inChannel, outChannel=64)
        self.enhanced = enhanced
        if self.enhanced:
            self.enhanceNet = HourGlass2(inChannel=64+1, outChannel=2, resNum=3, normLayer=nn.BatchNorm2d)

        ## transformer architecture
        self.n_vocab = 313
        self.hint_num = n_clusters
        d_model, dim_feedforward, nhead = d_model, 4*d_model, 8
        dropout, activation = 0.1, "relu"
        n_enc_layers, n_dec_layers = 6, 6
        enc_layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, use_dense_pos)
        self.wildpath = TransformerEncoder(enc_layer, n_enc_layers, use_dense_pos)
        self.hintpath = TransformerEncoder(enc_layer, n_enc_layers, use_dense_pos)
        if self.spix_pos:
            n_pos_x, n_pos_y = 256, 256
        else:
            n_pos_x, n_pos_y = 256//sp_size, 16//sp_size
        self.pos_enc = build_position_encoding(d_model//2, n_pos_x, n_pos_y, is_learned=False)

        self.mid_word_prj = nn.Linear(d_model, self.n_vocab, bias=False)
        if self.hint2regress:
            self.trg_word_emb = nn.Linear(d_model+2+1, d_model, bias=False)
            self.trg_word_prj = nn.Linear(d_model, 2, bias=False)
        else:
            self.trg_word_emb = nn.Linear(d_model+self.n_vocab+1, d_model, bias=False)
            self.trg_word_prj = nn.Linear(d_model, self.n_vocab, bias=False)
        self.colorLabeler = basic.ColorLabel(device=torch.device("cuda:%d"%rank))
        anchor_mode = 'random' if random_hint else 'clustering'
        self.anchorGen = anchor_gen.AnchorAnalysis(mode=anchor_mode, colorLabeler=self.colorLabeler)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def load_and_froze_weight(self, checkpt_path):
        data_dict = torch.load(checkpt_path, map_location=torch.device('cpu'))
        '''
        for param_tensor in data_dict['state_dict']:
            print(param_tensor,'\t',data_dict['state_dict'][param_tensor].size())
        '''
        self.segnet.load_state_dict(data_dict['state_dict'])
        for name, param in self.segnet.named_parameters():
            param.requires_grad = False
        self.segnet.eval()

    def set_train(self):
        ## running mode only affect certain modules, e.g. Dropout, BN, etc.
        self.repnet.train()
        self.wildpath.train()
        self.hintpath.train()
        if self.enhanced:
            self.enhanceNet.train()
    
    def get_entry_mask(self, mask_tensor):
        if mask_tensor is None:
            return None
        ## flatten (N,1,H,W) to (N,HW)
        return mask_tensor.flatten(1)

    def forward(self, input_grays, input_colors, test_mode=False, sampled_T=0):
        affinity_map = self.segnet(input_grays)
        pred_feats = self.repnet(input_grays)
        if self.spix_pos:
            full_pos_feats = self.pos_enc(pred_feats)
            proxy_feats = torch.cat([pred_feats, input_colors, full_pos_feats], dim=1)
            pooled_proxy_feats, conf_sum = basic.poolfeat(proxy_feats, affinity_map, self.sp_size, self.sp_size, True)
            feat_tokens = pooled_proxy_feats[:,:64,:,:]
            spix_colors = pooled_proxy_feats[:,64:66,:,:]
            pos_feats = pooled_proxy_feats[:,66:,:,:]
        else:
            proxy_feats = torch.cat([pred_feats, input_colors], dim=1)
            pooled_proxy_feats, conf_sum = basic.poolfeat(proxy_feats, affinity_map, self.sp_size, self.sp_size, True)
            feat_tokens = pooled_proxy_feats[:,:64,:,:]
            spix_colors = pooled_proxy_feats[:,64:,:,:]
            pos_feats = self.pos_enc(feat_tokens)

        token_labels = torch.max(self.colorLabeler.encode_ab2ind(spix_colors), dim=1, keepdim=True)[1]
        spixel_sizes = basic.get_spixel_size(affinity_map, self.sp_size, self.sp_size)
        all_one_map = torch.ones(spixel_sizes.shape).cuda()
        empty_entries = torch.where(spixel_sizes < 25/(self.sp_size**2), all_one_map, 1-all_one_map)
        src_pad_mask = self.get_entry_mask(empty_entries) if self.use_token_mask else None
        trg_pad_mask = src_pad_mask
        
        ## parallel prob
        N,C,H,W = feat_tokens.shape
        ## (N,C,H,W) -> (HW,N,C)
        src_pos_seq = pos_feats.flatten(2).permute(2, 0, 1)
        src_seq = feat_tokens.flatten(2).permute(2, 0, 1)
        ## color prob branch
        enc_out, _ = self.wildpath(src_seq, src_pos_seq, src_pad_mask)
        pal_logit = self.mid_word_prj(enc_out)
        pal_logit = pal_logit.permute(1, 2, 0).view(N,self.n_vocab,H,W)

        ## seed prob branch
        ## mask(N,1,H,W): sample anchors at clustering layers
        if test_mode:
            color_feat = enc_out.permute(1, 2, 0).view(N,C,H,W)
            hint_mask, cluster_mask = self.anchorGen(color_feat, self.hint_num, spixel_sizes, use_sklearn_kmeans=False)
            pred_prob = torch.softmax(pal_logit, dim=1)
            color_feat2 = src_seq.permute(1, 2, 0).view(N,C,H,W)
            #pred_prob, adj_matrix = self.anchorGen._detect_correlation(color_feat, pred_prob, hint_mask, thres=0.1)
            if sampled_T < 0:
                ## GT anchor colors
                sampled_spix_colors = spix_colors
            elif sampled_T > 0:                
                top1_spix_colors = self.anchorGen._sample_anchor_colors(pred_prob, hint_mask, T=0)
                top2_spix_colors = self.anchorGen._sample_anchor_colors(pred_prob, hint_mask, T=1)
                top3_spix_colors = self.anchorGen._sample_anchor_colors(pred_prob, hint_mask, T=2)
                ## duplicate meta tensors
                sampled_spix_colors = torch.cat((top1_spix_colors,top2_spix_colors,top3_spix_colors), dim=0)
                N = 3*N
                input_grays = input_grays.expand(N,-1,-1,-1)
                hint_mask = hint_mask.expand(N,-1,-1,-1)
                affinity_map = affinity_map.expand(N,-1,-1,-1)
                src_seq = src_seq.expand(-1, N,-1)
                src_pos_seq = src_pos_seq.expand(-1, N,-1)
            else:
                sampled_spix_colors = self.anchorGen._sample_anchor_colors(pred_prob, hint_mask, T=sampled_T)
            ## debug: controllable
            if False:
                hint_mask, sampled_spix_colors = basic.io_user_control(hint_mask, spix_colors, output=False)

            sampled_token_labels = torch.max(self.colorLabeler.encode_ab2ind(sampled_spix_colors), dim=1, keepdim=True)[1]
            ## for anchor visualization use
            spix_colors = sampled_spix_colors
        else:
            with torch.no_grad():
                hint_mask, cluster_mask = self.anchorGen(spix_colors, self.hint_num, spixel_sizes)
        
        ## hint based prediction
        ## (N,C,H,W) -> (HW,N,C)
        mask_seq = hint_mask.flatten(2).permute(2, 0, 1)
        if self.hint2regress:
            spix_colors_ = sampled_spix_colors if test_mode else spix_color
            gt_seq = spix_colors_.flatten(2).permute(2, 0, 1)
            hint_seq = self.trg_word_emb(torch.cat([src_seq, mask_seq * gt_seq, mask_seq], dim=2))
            dec_out, _ = self.hintpath(hint_seq, src_pos_seq, src_pad_mask)
        else:
            token_labels_ = sampled_token_labels if test_mode else token_labels
            label_map = F.one_hot(token_labels_, num_classes=313).squeeze(1).float()
            label_seq = label_map.permute(0, 3, 1, 2).flatten(2).permute(2, 0, 1)
            hint_seq = self.trg_word_emb(torch.cat([src_seq, mask_seq * label_seq, mask_seq], dim=2))
            dec_out, _ = self.hintpath(hint_seq, src_pos_seq, src_pad_mask)
        ref_logit = self.trg_word_prj(dec_out)
        Ct = 2 if self.hint2regress else self.n_vocab
        ref_logit = ref_logit.permute(1, 2, 0).view(N,Ct,H,W)
        
        ## pixelwise enhancement
        pred_colors = None
        if self.enhanced:
            proc_feats = dec_out.permute(1, 2, 0).view(N,64,H,W)
            full_feats = basic.upfeat(proc_feats, affinity_map, self.sp_size, self.sp_size)
            pred_colors = self.enhanceNet(torch.cat((input_grays,full_feats), dim=1))
            pred_colors = torch.tanh(pred_colors)

        return pal_logit, ref_logit, pred_colors, affinity_map, spix_colors, hint_mask