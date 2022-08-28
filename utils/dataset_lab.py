from __future__ import print_function, division
import torch, os, glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2


class LabDataset(Dataset):

    def __init__(self, rootdir=None, filelist=None, resize=None):

        if filelist:
            self.file_list = filelist
        else:
            assert os.path.exists(rootdir), "@dir:'%s' NOT exist ..."%rootdir
            self.file_list = glob.glob(os.path.join(rootdir, '*.*'))
            self.file_list.sort()
        self.resize = resize

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        bgr_img = cv2.imread(self.file_list[idx], cv2.IMREAD_COLOR)
        if self.resize:
            bgr_img = cv2.resize(bgr_img, (self.resize,self.resize), interpolation=cv2.INTER_CUBIC)
        bgr_img = np.array(bgr_img / 255., np.float32)
        lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        #print('--------L:', np.min(lab_img[:,:,0]), np.max(lab_img[:,:,0]))
        #print('--------ab:', np.min(lab_img[:,:,1:3]), np.max(lab_img[:,:,1:3]))
        lab_img = torch.from_numpy(lab_img.transpose((2, 0, 1)))
        bgr_img = torch.from_numpy(bgr_img.transpose((2, 0, 1)))
        gray_img = (lab_img[0:1,:,:]-50.) / 50.
        color_map = lab_img[1:3,:,:] / 110.
        bgr_img = bgr_img*2. - 1.
        return {'gray': gray_img, 'color': color_map, 'BGR': bgr_img}