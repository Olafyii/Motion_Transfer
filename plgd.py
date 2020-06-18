'''
    Debug playground
'''
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import scipy.io as sio
import os
import torch.nn as nn
from PIL import Image

if __name__ == '__main__':
    # from dataset import mtdataset
    # from param import get_general_params
    # params = get_general_params()
    # a = mtdataset(params, full_y=False)
    # out = a[0]
    # mask_gt = out[-1]
    # src_img = out[0]*0.5+0.5
    # mask_gt = nn.Threshold(0.01, 0)(mask_gt)
    # mask_gt = (torch.cat((mask_gt, mask_gt, mask_gt), dim=0).permute(1, 2, 0)*255).numpy().astype(np.uint8)
    # src_img = (src_img*255).permute(1, 2, 0).numpy().astype(np.uint8)
    # print(mask_gt.dtype)
    # Image.fromarray(mask_gt).save('1758.jpg')
    # Image.fromarray(src_img).save('1758_si.jpg')

    # conv = nn.Conv2d(3, 3, kernel_size=7,stride=2,padding=int((7-1)/2),bias=True)
    # a = torch.zeros((1, 3, 256, 256))
    # print(conv(a).size())

    # print(torch.load('/versa/kangliwei/motion_transfer/0426-vgg/epoch_350.pth').keys())

    # a = [1,1,1,1]
    # a = torch.tensor(a)
    # print(a)
    
    # a = nn.Conv2d(3, 1, 3)
    # op = torch.optim.Adam(a.parameters(), lr=1e-4)
    # print(type(op) == type('D'))

    print(torch.load('/versa/kangliwei/motion_transfer/0429-256-gan/d_epoch_40.pth').keys())