import torch
import torch.nn.functional as F
from torch import nn
import copy, math
import basic
from position_encoding import build_position_encoding


class TransformerEncoder(nn.Module):

    def __init__(self, enc_layer, num_layers, use_dense_pos=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(enc_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.use_dense_pos = use_dense_pos

    def forward(self, src, pos, padding_mask=None):
        if self.use_dense_pos:
            ## pos encoding at each MH-Attention block (q,k)
            output, pos_enc = src, pos
            for layer in self.layers:
                output, att_map = layer(output, pos_enc, padding_mask)
        else:
            ## pos encoding at input only (q,k,v)
            output, pos_enc = src + pos, None
            for layer in self.layers:
                output, att_map = layer(output, pos_enc, padding_mask)
        return output, att_map


class EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                use_dense_pos=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, padding_mask):
        q = k = self.with_pos_embed(src, pos)
        src2, attn = self.self_attn(q, k, value=src, key_padding_mask=padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class TransformerDecoder(nn.Module):

    def __init__(self, dec_layer, num_layers, use_dense_pos=False, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(dec_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.use_dense_pos = use_dense_pos
        self.return_intermediate = return_intermediate

    def forward(self, tgt, tgt_pos, memory, memory_pos, 
                tgt_padding_mask, src_padding_mask, tgt_attn_mask=None):
        intermediate = []
        if self.use_dense_pos:
            ## pos encoding at each MH-Attention block (q,k)
            output = tgt
            tgt_pos_enc, memory_pos_enc = tgt_pos, memory_pos
            for layer in self.layers:
                output, att_map = layer(output, tgt_pos_enc, memory, memory_pos_enc, 
                                tgt_padding_mask, src_padding_mask, tgt_attn_mask)
                if self.return_intermediate:
                    intermediate.append(output)
        else:
            ## pos encoding at input only (q,k,v)
            output = tgt + tgt_pos
            tgt_pos_enc, memory_pos_enc = None, None
            for layer in self.layers:
                output, att_map = layer(output, tgt_pos_enc, memory, memory_pos_enc, 
                                tgt_padding_mask, src_padding_mask, tgt_attn_mask)
                if self.return_intermediate:
                    intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output, att_map


class DecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 use_dense_pos=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.corr_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, tgt_pos, memory, memory_pos, 
                tgt_padding_mask, memory_padding_mask, tgt_attn_mask):
        q = k = self.with_pos_embed(tgt, tgt_pos)
        tgt2, attn = self.self_attn(q, k, value=tgt, key_padding_mask=tgt_padding_mask,
                                    attn_mask=tgt_attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.corr_attn(query=self.with_pos_embed(tgt, tgt_pos),
                                    key=self.with_pos_embed(memory, memory_pos),
                                    value=memory, key_padding_mask=memory_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



#-----------------------------------------------------------------------------------
'''
copy from the implementatoin of "attention-is-all-you-need-pytorch-master" by Yu-Hsiang Huang
'''

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn