'''
    Train generator from scratch with vgg loss and l1 loss.
'''
# from net_full_y import MModel
from net import MModel
from dataset import mtdataset
from param import get_general_params
from vgg_loss import VGGPerceptualLoss, L1MaskLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--start_epoch', type=int, default=290)
    parser.add_argument('--model_dir', type=str, default='0604-gan')
    # parser.add_argument('--weight_dir', type=str, default='0604-gan')
    parser.add_argument('--complete', action='store_true')
    parser.add_argument('--full_y', action='store_true')  # if true, ground-truth image is full image, i.e. with background, instead of foreground image
    parser.add_argument('--no_shuffle', action='store_true')

    args = parser.parse_args()

    n_epoch = args.n_epoch
    bs = args.bs
    lr = args.lr
    start_epoch = args.start_epoch
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # weight_dir = args.weight_dir
    mini = not args.complete
    full_y = args.full_y
    shuffle = not args.no_shuffle

    mini = False
    full_y = False
    shuffle = True

    params = get_general_params()
    params['IMG_HEIGHT'] = 256
    params['IMG_WIDTH'] = 256
    params['posemap_downsample'] = 2
    ds = mtdataset(params, mini=mini, full_y=full_y)
    dl = DataLoader(ds, bs, shuffle)
    model = MModel(params, use_cuda=True).cuda()
    model.train()
    # model.load_state_dict(torch.load('/versa/kangliwei/motion_transfer/0424-gan/g_epoch_1670.pth'), strict=False)
    # model.load_state_dict(torch.load('/versa/kangliwei/motion_transfer/0429-256-gan/g_epoch_720.pth'))
    model.load_state_dict(torch.load('/versa/kangliwei/motion_transfer/0604-gan/epoch_290.pth'))
    # if start_epoch != 0:
    #     model.load_state_dict(torch.load('/versa/kangliwei/motion_transfer/'+weight_dir+'/epoch_%d.pth'%(start_epoch-1)))
    vgg_loss = VGGPerceptualLoss().cuda()
    l1mask_loss = L1MaskLoss().cuda()
    model_params = model.parameters()
    optimizer = torch.optim.Adam(model_params, lr=lr)
    writer = SummaryWriter('runs/'+model_dir)

    print('len(dl)', len(dl))
    print('len(ds)', len(ds))
    for epoch in range(start_epoch, n_epoch):
        cnt = 0
        loss_sum = 0
        fg_loss_sum = 0
        mask_loss_sum = 0
        L1_sum = 0
        for i, (src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, src_mask_gt, tgt_face, tgt_face_box, src_face_box) in enumerate(dl):
            optimizer.zero_grad()
            print('epoch:', epoch, 'iter:', i)
            src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, tgt_face = src_img.cuda(), y.cuda(), src_pose.cuda(), tgt_pose.cuda(), src_mask_prior.cuda(), x_trans.cuda(), tgt_face.cuda()
            out = model(src_img, src_pose, tgt_pose, src_mask_prior, x_trans)
            # out: (FG, src_mask_delta, src_mask, warped_stack)
            # channel: 3, 11, 11, 33

            fg_loss = vgg_loss(out[0], y)
            # L1 = l1mask_loss(out[0], y, src_mask_gt)
            L1 = nn.L1Loss()(out[0], y)
            # mask_sum = torch.clamp_max(out[2][:, 1:, :, :].sum(dim=1), 1)
            # mask_loss = nn.BCELoss()(mask_sum.cpu(), src_mask_gt)
            # loss = fg_loss + mask_loss + L1
            if epoch < 100:
                loss = L1 + fg_loss  #  + mask_loss
            else:
                loss = L1 + fg_loss
            loss.backward()
            optimizer.step()

            cnt += 1
            loss_sum += loss.item()
            fg_loss_sum += fg_loss.item()
            # mask_loss_sum += mask_loss.item()
            L1_sum += L1.item()

            # writer.add_images('Image/epoch%d/y'%epoch, y)
            # writer.add_images('Image/epoch%d/gen'%epoch, out)
            if (epoch % 10 == 0 or epoch < 5) and i == 0:
                # writer.add_scalar('Epoch%d/loss'%epoch, loss.item(), i)
                writer.add_images('genFG/epoch%d'%epoch, out[0]*0.5+0.5)
                writer.add_images('y/epoch%d'%epoch, y*0.5+0.5)
                writer.add_images('src_img/epoch%d'%epoch, src_img*0.5+0.5)
                # writer.add_images('src_mask_delta/epoch%d'%epoch, out[1].view((out[1].size(0)*out[1].size(1), 1, out[1].size(2), out[1].size(3))))
                writer.add_images('src_mask/epoch%d'%epoch, out[2].view((out[2].size(0)*out[2].size(1), 1, out[2].size(2), out[2].size(3))))
                # writer.add_images('src_mask_prior/epoch%d'%epoch, src_mask_prior.view((src_mask_prior.size(0)*src_mask_prior.size(1), 1, src_mask_prior.size(2), src_mask_prior.size(3))))
                # writer.add_images('warped/epoch%d'%epoch, out[3].view((out[3].size(0)*11, 3, out[3].size(2), out[3].size(3)))*0.5+0.5)
                # writer.add_images('mask_sum/epoch%d'%epoch, mask_sum.unsqueeze(1))
                # writer.add_histogram('src_mask_hist/epoch%d'%epoch, out[2])
                # writer.add_histogram('src_mask_delta_hist/epoch%d'%epoch, out[1])
                # writer.add_histogram('src_mask_gt/epoch%d'%epoch, src_mask_gt)
                


        writer.add_scalar('Train/loss', loss_sum/cnt, epoch)
        writer.add_scalar('Train/fg_loss', fg_loss_sum/cnt, epoch)
        # writer.add_scalar('Train/mask_loss', mask_loss_sum/cnt, epoch)
        writer.add_scalar('Train/L1_loss', L1_sum/cnt, epoch)
        if epoch % 10 == 0 and epoch != 0 and mini == False:
            torch.save(model.state_dict(), model_dir+'/epoch_%d.pth'%epoch)

