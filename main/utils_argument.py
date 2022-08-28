import os, glob, sys
import argparse, time


def spixel_argparser(parser):
    parser.add_argument('--exp_name', default='spixelG2C', type=str, help='experiment name')
    parser.add_argument('--model', default='SpixelSeg', type=str, help='which model to use')
    parser.add_argument('--psize', default='16', type=int, help='super-pixel size')
    parser.add_argument('--feat', default='ab', type=str, help='supervision feature: {g, ab, rgb}')
    parser.add_argument('--resume', action='store_true', default=False, help='resume from existing checkpoint')

    parser.add_argument('--optim', default='adam', type=str, help='adam, sgd')
    parser.add_argument('--scheduler', default='linear', type=str, help='LR scheduler')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--wd', default=0, type=float, help='weight decay')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--workers', default=16, type=int, help='number of worker to use')
    parser.add_argument('--eval_freq', default=1, type=int)

    parser.add_argument('--dataset', default='voc', type=str)
    parser.add_argument('--input_dim', default=256, type=int)
    parser.add_argument('--image_dim', default=224, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    
    parser.add_argument('--data_dir', default='../../../0DataZoo/Dataset_C/VOC2012/', type=str, help='dataset directory')
    parser.add_argument('--ckpt_dir', default='../../Saved/', type=str, help='pretrained weight')
    parser.add_argument('--save_dir', default='../../Saved/', type=str, help='output directory')
    return parser


def ddp_spixel_argparse(parser):
    parser = spixel_argparser(parser)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    return parser


def pcolor_argparser(parser):
    parser.add_argument('--exp_name', default='colorProb', type=str, help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--model', default='ColorProb', type=str, help='which model to use')
    parser.add_argument('--psize', default='16', type=int, help='super-pixel size')
    parser.add_argument('--resume', action='store_true', default=False, help='resume from existing checkpoint')

    parser.add_argument('--n_enc', default=3, type=int, help='number of encoder layers')
    parser.add_argument('--n_dec', default=6, type=int, help='number of decoder layers')
    parser.add_argument('--d_model', default=64, type=int, help='feature dimension of transformer')
    parser.add_argument('--d_mlp', default=256, type=int, help='feature dimension of feedforward')
    parser.add_argument('--dense_pos', action='store_true', default=False, help='use pos encoding at each SA block')
    parser.add_argument('--spix_pos', action='store_true', default=False, help='use pos of spixel centroid')
    parser.add_argument('--learning_pos', action='store_true', default=False, help='learnable pos embedding')
    parser.add_argument('--hint2regress', action='store_true', default=False, help='predict ab values from hint')
    parser.add_argument('--n_clusters', default=8, type=int, help='number of color clusters')
    parser.add_argument('--random_hint', action='store_true', default=False, help='sample anchors randomly')
    parser.add_argument('--enhanced', action='store_true', default=False, help='use pixel-level enhancement or not')
    parser.add_argument('--vgg_type', default='liu', type=str, help='which vgg features to use: {lei, liu, mine}')
    parser.add_argument('--in_gradient', action='store_true', default=False, help='supervision in gradient domain')
    parser.add_argument('--colorfulness', default=0.5, type=float, help='color class rebalance in training')

    parser.add_argument('--optim', default='adam', type=str, help='adam, sgd')
    parser.add_argument('--scheduler', default='linear', type=str, help='LR scheduler')
    parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parser.add_argument('--decay_ratio', default=1e-2, type=float, help='only applicable to linear scheduler')
    parser.add_argument('--wd', default=0, type=float, help='weight decay')
    parser.add_argument('--epochs', default=60, type=int, help='number of total epochs to run')
    parser.add_argument('--workers', default=16, type=int, help='number of worker to use')
    parser.add_argument('--eval_freq', default=1, type=int)

    parser.add_argument('--dataset', default='imagenet', type=str)
    parser.add_argument('--input_dim', default=256, type=int)
    parser.add_argument('--image_dim', default=224, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='batch size per GPU')
    
    parser.add_argument('--data_dir', default='../../../0DataZoo/Dataset_C/ILSVRC2012/', type=str, help='dataset directory')
    parser.add_argument('--ckpt_dir', default='../../Saved/', type=str, help='pretrained weight')
    parser.add_argument('--save_dir', default='../../Saved/', type=str, help='output directory')
    return parser


def ddp_pcolor_argparse(parser):
    parser = pcolor_argparser(parser)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    return parser