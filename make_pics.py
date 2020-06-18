'''
    整个网络的demo.
'''
from net import MModel
# from dataset import mtdataset
from dataset_face import mtdataset
from param import get_general_params
from vgg_loss import VGGPerceptualLoss, L1MaskLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import time
from resnet_unet import UNetWithResnet50Encoder

def crop_face(gen, size, tgt_face_box):
    gen_face = torch.zeros(size).cuda()
    for i in range(gen.size(0)):
        # print(tgt_face_box[i][1], tgt_face_box[i][3], tgt_face_box[i][0], tgt_face_box[i][2])
        face = gen[i, :, tgt_face_box[i][1]:tgt_face_box[i][3], tgt_face_box[i][0]:tgt_face_box[i][2]]
        # print(face.size())
        face = F.interpolate(face.unsqueeze(0), size=256)
        gen_face[i] = face
    return gen_face

def paste_face(img, box, face):
    for i in range(img.size(0)):
        img[i][:, box[i][1]:box[i][3], box[i][0]:box[i][2]] = F.interpolate(face[i].unsqueeze(0), size=(box[i][3]-box[i][1], box[i][2]-box[i][0]))
    return

if __name__ == '__main__':
    params = get_general_params()
    params['IMG_HEIGHT'] = 256
    params['IMG_WIDTH'] = 256
    params['posemap_downsample'] = 2
    bs = 1
    shuffle = False
    epoch = 0
    ds = mtdataset(params, mini=False, full_y=False, mode='train')
    dl = DataLoader(ds, bs, shuffle)
    model = MModel(params, use_cuda=True).cuda()
    writer = SummaryWriter('runs/makepic')
    # model.load_state_dict(torch.load('/versa/kangliwei/motion_transfer/0429-256-gan/g_epoch_720.pth'))
    model.load_state_dict(torch.load('/versa/kangliwei/motion_transfer/0605-gan/g_epoch_610.pth'))
    model.eval()

    Face_G = UNetWithResnet50Encoder(in_ch=4, n_classes=3).cuda()
    # Face_G.load_state_dict(torch.load('/versa/kangliwei/motion_transfer/0605-face-dlib/faceg_epoch_100.pth'))
    # Face_G.load_state_dict(torch.load('/versa/kangliwei/motion_transfer/0605-face-dlib/faceg_epoch_1050.pth'))
    Face_G.load_state_dict(torch.load('/versa/kangliwei/motion_transfer/0603-face-dlib-2/faceg_epoch_2400.pth'))
    Face_G.eval()

    for i, (src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, src_mask_gt, tgt_face, tgt_face_box, src_face_box, tgt_face_heatmap) in enumerate(dl):
        print('epoch:', epoch, 'iter:', i)
        src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, tgt_face, tgt_face_heatmap = src_img.cuda(), y.cuda(), src_pose.cuda(), tgt_pose.cuda(), src_mask_prior.cuda(), x_trans.cuda(), tgt_face.cuda(), tgt_face_heatmap.cuda()
        # print('src_pose.size(), tgt_pose.size()', src_pose.size(), tgt_pose.size())
        with torch.no_grad():
            out = model(src_img, src_pose, tgt_pose, src_mask_prior, x_trans)
            gen = out[0].clone()
            gen_face = crop_face(gen, tgt_face.size(), tgt_face_box)
            face_residual = Face_G(torch.cat((gen_face, tgt_face_heatmap), dim=1))
            fined_face = F.tanh(gen_face + face_residual)
            paste_face(gen, tgt_face_box, fined_face)

            mask_sum = torch.clamp_max(out[2][:, 1:, :, :].sum(dim=1), 1)
            src_mask_prior = F.softmax(src_mask_prior)
        # print('src_mask_prior.max(), src_mask_prior.min()', src_mask_prior.max(), src_mask_prior.min())
        writer.add_images('genFG/epoch%d'%i, out[0]*0.5+0.5)
        writer.add_images('fined_face/epoch%d'%i, gen*0.5+0.5)
        writer.add_images('y/epoch%d'%i, y*0.5+0.5)
        writer.add_image('src_pose/epoch%d'%i, torch.sum(src_pose[0], 0).unsqueeze(0))
        writer.add_image('tgt_pose/epoch%d'%i, torch.sum(tgt_pose[0], 0).unsqueeze(0))
        writer.add_images('src_img/epoch%d'%i, src_img*0.5+0.5)
        # writer.add_images('agg/epoch%d'%i, torch.cat((src_img, y, out[0], gen), dim=3)*0.5+0.5)
        writer.add_images('agg/epoch%d'%i, torch.cat((src_img, y, gen), dim=3)*0.5+0.5)
        writer.add_images('src_mask/epoch%d'%i, out[2].view((out[2].size(0)*out[2].size(1), 1, out[2].size(2), out[2].size(3))))
        writer.add_images('warped/epoch%d'%i, out[3].view((out[3].size(0)*11, 3, out[3].size(2), out[3].size(3)))*0.5+0.5)
        writer.add_images('mask_sum/epoch%d'%i, mask_sum.unsqueeze(1))
        writer.add_images('src_mask_delta/epoch%d'%i, out[1].view((out[1].size(0)*out[1].size(1), 1, out[1].size(2), out[1].size(3))))
        writer.add_images('src_mask_prior/epoch%d'%i, src_mask_prior.view((src_mask_prior.size(0)*src_mask_prior.size(1), 1, src_mask_prior.size(2), src_mask_prior.size(3))))
        # print('sleep')
        # time.sleep(10)
        # print('wake')
        # break