from __future__ import division
from __future__ import print_function
import os, glob, shutil, math, json
from queue import Queue
from threading import Thread
from skimage.segmentation import mark_boundaries
import numpy as np
from PIL import Image
import cv2, torch

def get_gauss_kernel(size, sigma):
    '''Function to mimic the 'fspecial' gaussian MATLAB function'''
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def batchGray2Colormap(gray_batch):
    colormap = plt.get_cmap('viridis')
    heatmap_batch = []
    for i in range(gray_batch.shape[0]):
        # quantize [-1,1] to {0,1}
        gray_map = gray_batch[i, :, :, 0]
        heatmap = (colormap(gray_map) * 2**16).astype(np.uint16)[:,:,:3]
        heatmap_batch.append(heatmap/127.5-1.0)
    return np.array(heatmap_batch)


class PlotterThread():
    '''log tensorboard data in a background thread to save time'''
    def __init__(self, writer):
        self.writer = writer
        self.task_queue = Queue(maxsize=0)
        worker = Thread(target=self.do_work, args=(self.task_queue,))
        worker.setDaemon(True)
        worker.start()

    def do_work(self, q):
        while True:
            content = q.get()
            if content[-1] == 'image':
                self.writer.add_image(*content[:-1])
            elif content[-1] == 'scalar':
                self.writer.add_scalar(*content[:-1])
            else:
                raise ValueError
            q.task_done()

    def add_data(self, name, value, step, data_type='scalar'):
        self.task_queue.put([name, value, step, data_type])

    def __len__(self):
        return self.task_queue.qsize()


def save_images_from_batch(img_batch, save_dir, filename_list, batch_no=-1, suffix=None):
    N,H,W,C = img_batch.shape
    if C == 3:
        #! rgb color image
        for i in range(N):
            # [-1,1] >>> [0,255]
            image = Image.fromarray((127.5*(img_batch[i,:,:,:]+1.)).astype(np.uint8))
            save_name = filename_list[i] if batch_no==-1 else '%05d.png' % (batch_no*N+i)
            save_name = save_name.replace('.png', '-%s.png'%suffix) if suffix else save_name
            image.save(os.path.join(save_dir, save_name), 'PNG')
    elif C == 1:
        #! single-channel gray image
        for i in range(N):
            # [-1,1] >>> [0,255]
            image = Image.fromarray((127.5*(img_batch[i,:,:,0]+1.)).astype(np.uint8))
            save_name = filename_list[i] if batch_no==-1 else '%05d.png' % (batch_no*img_batch.shape[0]+i)
            save_name = save_name.replace('.png', '-%s.png'%suffix) if suffix else save_name
            image.save(os.path.join(save_dir, save_name), 'PNG')
    else:
        #! multi-channel: save each channel as a single image
        for i in range(N):
            # [-1,1] >>> [0,255]
            for j in range(C):
                image = Image.fromarray((127.5*(img_batch[i,:,:,j]+1.)).astype(np.uint8))
                if batch_no == -1:
                    _, file_name = os.path.split(filename_list[i])
                    name_only, _ = os.path.os.path.splitext(file_name)
                    save_name = name_only + '_c%d.png' % j
                else:
                    save_name = '%05d_c%d.png' % (batch_no*N+i, j)
                save_name = save_name.replace('.png', '-%s.png'%suffix) if suffix else save_name
                image.save(os.path.join(save_dir, save_name), 'PNG')
    return None


def save_normLabs_from_batch(img_batch, save_dir, filename_list, batch_no=-1, suffix=None):
    N,H,W,C = img_batch.shape
    if C != 3:
        print('@Warning:the Lab images are NOT in 3 channels!')
        return None
    # denormalization: L: (L+1.0)*50.0 | a: a*110.0| b: b*110.0
    img_batch[:,:,:,0] = img_batch[:,:,:,0] * 50.0 + 50.0
    img_batch[:,:,:,1:3] = img_batch[:,:,:,1:3] * 110.0
    #! convert into RGB color image
    for i in range(N):
        rgb_img = cv2.cvtColor(img_batch[i,:,:,:], cv2.COLOR_LAB2RGB)
        image = Image.fromarray((rgb_img*255.0).astype(np.uint8))
        save_name = filename_list[i] if batch_no==-1 else '%05d.png' % (batch_no*N+i)
        save_name = save_name.replace('.png', '-%s.png'%suffix) if suffix else save_name
        image.save(os.path.join(save_dir, save_name), 'PNG')
    return None


def save_markedSP_from_batch(img_batch, spix_batch, save_dir, filename_list, batch_no=-1, suffix=None):
    N,H,W,C = img_batch.shape
    #! img_batch: BGR nd-array (range:0~1)
    #! map_batch: single-channel spixel map
    #print('----------', img_batch.shape, spix_batch.shape)
    for i in range(N):
        norm_image = img_batch[i,:,:,:]*0.5+0.5
        spixel_bd_image = mark_boundaries(norm_image, spix_batch[i,:,:,0].astype(int), color=(1,1,1))
        #spixel_bd_image = cv2.cvtColor(spixel_bd_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray((spixel_bd_image*255.0).astype(np.uint8))
        save_name = filename_list[i] if batch_no==-1 else '%05d.png' % (batch_no*N+i)
        save_name = save_name.replace('.png', '-%s.png'%suffix) if suffix else save_name
        image.save(os.path.join(save_dir, save_name), 'PNG')
    return None


def get_filelist(data_dir):
    file_list = glob.glob(os.path.join(data_dir, '*.*'))
    file_list.sort()
    return file_list
    

def collect_filenames(data_dir):
    file_list = get_filelist(data_dir)
    name_list = []
    for file_path in file_list:
        _, file_name = os.path.split(file_path)
        name_list.append(file_name)
    name_list.sort()
    return name_list


def exists_or_mkdir(path, need_remove=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif need_remove:
        shutil.rmtree(path)
        os.makedirs(path)
    return None


def save_list(save_path, data_list, append_mode=False):
    n = len(data_list)
    if append_mode:
        with open(save_path, 'a') as f:
            f.writelines([str(data_list[i]) + '\n' for i in range(n-1,n)])
    else:
        with open(save_path, 'w') as f:
            f.writelines([str(data_list[i]) + '\n' for i in range(n)])
    return None
    
    
def save_dict(save_path, dict):
    json.dumps(dict, open(save_path,"w"))
    return None


if __name__ == '__main__':
    data_dir = '../PolyNet/PolyNet/cache/'
    #visualizeLossCurves(data_dir)
    clbar = GamutIndex()
    ab, ab_gamut_mask = clbar._get_gamut_mask()
    ab2q = clbar._get_ab_to_q(ab_gamut_mask)
    q2ab = clbar._get_q_to_ab(ab, ab_gamut_mask)
    maps = ab_gamut_mask*255.0
    image = Image.fromarray(maps.astype(np.uint8))
    image.save('gamut.png', 'PNG')
    print(ab2q.shape)
    print(q2ab.shape)
    print('label range:', np.min(ab2q), np.max(ab2q))