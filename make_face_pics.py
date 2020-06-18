'''
    整理图片用的脚本
'''
from net import MModel
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
from network import Face_U_Net, U_Net

def crop_face(gen, size, tgt_face_box):
    gen_face = torch.zeros(size).cuda()
    for i in range(gen.size(0)):
        # print(tgt_face_box[i][1], tgt_face_box[i][3], tgt_face_box[i][0], tgt_face_box[i][2])
        face = gen[i, :, tgt_face_box[i][1]:tgt_face_box[i][3], tgt_face_box[i][0]:tgt_face_box[i][2]]
        # print(face.size())
        face = F.interpolate(face.unsqueeze(0), size=256)
        gen_face[i] = face
    return gen_face

if __name__ == '__main__':
    params = get_general_params()
    params['IMG_HEIGHT'] = 256
    params['IMG_WIDTH'] = 256
    params['posemap_downsample'] = 2
    bs = 1
    shuffle = False
    epoch = 0
    ds = mtdataset(params, mini=False, full_y=False, mode='test')
    dl = DataLoader(ds, bs, shuffle)
    model = MModel(params, use_cuda=True).cuda()
    writer = SummaryWriter('runs/makepic')
    model.load_state_dict(torch.load('/versa/kangliwei/motion_transfer/0429-256-gan/g_epoch_720.pth'))
    model.eval()

    Face_G = UNetWithResnet50Encoder(in_ch=6, n_classes=3).cuda()
    Face_G.load_state_dict(torch.load('/versa/kangliwei/motion_transfer/0518-faceg-resunet-256-gan/faceg_epoch_100.pth'))
    Face_G.eval()

    Face_G2 = UNetWithResnet50Encoder(in_ch=4, n_classes=3).cuda()
    Face_G2.load_state_dict(torch.load('/versa/kangliwei/motion_transfer/0603-face-dlib-2/faceg_epoch_2400.pth'))
    Face_G2.eval()

    model2 = MModel(params, use_cuda=True).cuda()
    model2.load_state_dict(torch.load('/versa/kangliwei/motion_transfer/0511-256-face-gan-2/g_epoch_50.pth'), strict=True)
    model2.eval()

    for i, (src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, src_mask_gt, tgt_face, tgt_face_box, src_face_box, tgt_face_heatmap) in enumerate(dl):
        print('epoch:', epoch, 'iter:', i)
        src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, tgt_face, tgt_face_heatmap = src_img.cuda(), y.cuda(), src_pose.cuda(), tgt_pose.cuda(), src_mask_prior.cuda(), x_trans.cuda(), tgt_face.cuda(), tgt_face_heatmap.cuda()
        print('src_pose.size(), tgt_pose.size(), tgt_face_heatmap.size()', src_pose.size(), tgt_pose.size(), tgt_face_heatmap.size())
        with torch.no_grad():
            out = model(src_img, src_pose, tgt_pose, src_mask_prior, x_trans)
            out = model(src_img, src_pose, tgt_pose, src_mask_prior, x_trans)
            gen = out[0]
            gen_face = crop_face(gen, tgt_face.size(), tgt_face_box)
            src_face = crop_face(src_img, tgt_face.size(), src_face_box)
            writer.add_image('weight_loss/%d'%i, gen_face.squeeze(0)*0.5+0.5)
            writer.add_image('src_face/%d'%i, src_face.squeeze(0)*0.5+0.5)
            writer.add_image('tgt_face/%d'%i, tgt_face.squeeze(0)*0.5+0.5)
            writer.add_image('tgt_face_heatmap/%d'%i, tgt_face_heatmap[0])

            writer.add_image('tgt_pose/%d'%i, torch.sum(tgt_pose[0], 0).unsqueeze(0))
            writer.add_images('src_img/epoch%d'%i, src_img*0.5+0.5)
            writer.add_images('tgt_img/epoch%d'%i, y*0.5+0.5)
            writer.add_images('gen/epoch%d'%i, out[0]*0.5+0.5)

            face_residual = Face_G(torch.cat((gen_face, src_face), dim=1))
            fined_face = F.tanh(gen_face + face_residual)

            face_residual2 = Face_G2(torch.cat((gen_face, tgt_face_heatmap), dim=1))
            fined_face2 = F.tanh(gen_face + face_residual2)

            writer.add_image('d+g/%d'%i, fined_face.squeeze(0)*0.5+0.5)
            writer.add_image('dlib/%d'%i, fined_face2.squeeze(0)*0.5+0.5)

            out2 = model2(src_img, src_pose, tgt_pose, src_mask_prior, x_trans)
            gen2 = out2[0]
            gen_face2 = crop_face(gen2, tgt_face.size(), tgt_face_box)

            writer.add_image('d/%d'%i, gen_face2.squeeze(0)*0.5+0.5)
            writer.add_image('agg/%d'%i, torch.cat((src_face.squeeze(0)*0.5+0.5, tgt_face.squeeze(0)*0.5+0.5, gen_face.squeeze(0)*0.5+0.5, gen_face2.squeeze(0)*0.5+0.5, fined_face.squeeze(0)*0.5+0.5, fined_face2.squeeze(0)*0.5+0.5), dim=2))
            # writer.add_image('gen2', gen2.squeeze(0)*0.5+0.5)


        # mask_sum = torch.clamp_max(out[2][:, 1:, :, :].sum(dim=1), 1)
        # src_mask_prior = F.softmax(src_mask_prior)
        # print('src_mask_prior.max(), src_mask_prior.min()', src_mask_prior.max(), src_mask_prior.min())
        # writer.add_images('genFG/epoch%d'%epoch, out[0]*0.5+0.5)
        # writer.add_images('y/epoch%d'%epoch, y*0.5+0.5)
        # writer.add_image('src_pose/epoch%d'%epoch, torch.sum(src_pose[0], 0).unsqueeze(0))
        # writer.add_image('tgt_pose/epoch%d'%epoch, torch.sum(tgt_pose[0], 0).unsqueeze(0))
        # writer.add_images('src_img/epoch%d'%epoch, src_img*0.5+0.5)
        # writer.add_images('src_mask/epoch%d'%epoch, out[2].view((out[2].size(0)*out[2].size(1), 1, out[2].size(2), out[2].size(3))))
        # writer.add_images('warped/epoch%d'%epoch, out[3].view((out[3].size(0)*11, 3, out[3].size(2), out[3].size(3)))*0.5+0.5)
        # writer.add_images('mask_sum/epoch%d'%epoch, mask_sum.unsqueeze(1))
        # writer.add_images('src_mask_delta/epoch%d'%epoch, out[1].view((out[1].size(0)*out[1].size(1), 1, out[1].size(2), out[1].size(3))))
        # writer.add_images('src_mask_prior/epoch%d'%epoch, src_mask_prior.view((src_mask_prior.size(0)*src_mask_prior.size(1), 1, src_mask_prior.size(2), src_mask_prior.size(3))))
        
        # print('sleep')
        # time.sleep(10)
        # print('wake')
        # break

