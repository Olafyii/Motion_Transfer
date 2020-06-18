'''
    训练人脸生成器和判别器。
'''
from net import MModel
# from dataset import mtdataset
from dataset_face import mtdataset
from vgg_loss import VGGPerceptualLoss
from torch.utils.tensorboard import SummaryWriter
from discriminator import Discriminator, FaceDisc, PatchFace
from network import Face_U_Net, U_Net
# from discriminator_progressive import Discriminator
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import argparse
from param import get_general_params
from torch.utils.data import DataLoader
from torchvision import transforms
from resnet_unet import UNetWithResnet50Encoder
from detect_face_keypoints import face_detector

class gan(nn.Module):
    # def __init__(self, params, save_dir, g_weight_dir, d_weight_dir, d_update_freq=1, start_epoch=0, g_lr=2e-4, d_lr=2e-4, use_cuda=True):
    def __init__(self, params, args):
        super(gan, self).__init__()
        self.G = MModel(params, use_cuda=True)
        # self.Face_G = Face_U_Net(img_ch=6, output_ch=3)
        # self.Face_G = U_Net(params, output_ch=3)
        self.Face_G = UNetWithResnet50Encoder(in_ch=3+3+1, n_classes=3)
        # self.Face_D = FaceDisc()
        # self.Face_D = Discriminator(params)
        self.Face_D = PatchFace(in_channels=3+3+1)
        self.vgg_loss = VGGPerceptualLoss()
        self.L1_loss = nn.L1Loss()
        if args.use_cuda:
            self.G = self.G.cuda()
            self.Face_G = self.Face_G.cuda()
            self.Face_D = self.Face_D.cuda()
            self.vgg_loss = self.vgg_loss.cuda()
            self.L1_loss = self.L1_loss.cuda()
        
        print('loading g model', args.g_weight_dir)
        self.G.load_state_dict(torch.load(args.g_weight_dir), strict=True)
        self.G.eval()

        if args.face_g_weight_dir:
            print('loading face_g model', args.face_g_weight_dir)
            self.Face_G.load_state_dict(torch.load(args.face_g_weight_dir))
        if args.face_d_weight_dir:
            print('loading face_d model', args.face_d_weight_dir)
            self.Face_D.load_state_dict(torch.load(args.face_d_weight_dir))
        
        self.optimizer_Face_G = torch.optim.Adam(self.Face_G.parameters(), lr=args.face_g_lr)
        self.optimizer_Face_D = torch.optim.Adam(self.Face_D.parameters(), lr=args.face_d_lr)

        self.save_dir = args.save_dir
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        self.d_update_freq = args.d_update_freq
        self.save_freq = args.save_freq
        self.writer = SummaryWriter('runs/'+args.save_dir)
        self.use_cuda = args.use_cuda
        self.start_epoch = args.start_epoch
    
    def G_loss(self, input, target):
        # vgg = self.vgg_loss(input, target)
        L1 = self.L1_loss(input, target)
        # return vgg
        # return vgg + L1
        return L1
    
    def update_Face_D(self, loss, epoch):
        if epoch % self.d_update_freq == 0:
            loss.backward()
            self.optimizer_Face_D.step()

    def update_Face_G(self, loss, epoch):
        # if epoch > 100:
        if True:
            loss.backward()
            self.optimizer_Face_G.step()

    def get_patch_weight(self, pose, size=62):
        heads = pose[:, 0, :, :]
        heads = heads.unsqueeze(1)
        heads = torch.nn.functional.interpolate(heads, size=size)
        heads = heads*5 + torch.ones_like(heads)
        return heads

    def gan_loss(self, out, label, pose):
        # weight = self.get_patch_weight(pose)
        # return nn.BCELoss(weight=weight)(out, torch.ones_like(out) if label==1 else torch.zeros_like(out))
        return nn.BCELoss()(out, torch.ones_like(out) if label==1 else torch.zeros_like(out))
    
    def face_gan_loss(self, out, label):
        return nn.BCELoss()(out, torch.ones_like(out) if label==1 else torch.zeros_like(out))
    
    def crop_face(self, gen, size, tgt_face_box):
        gen_face = torch.zeros(size).cuda()
        for i in range(gen.size(0)):
            # print(tgt_face_box[i][1], tgt_face_box[i][3], tgt_face_box[i][0], tgt_face_box[i][2])
            face = gen[i, :, tgt_face_box[i][1]:tgt_face_box[i][3], tgt_face_box[i][0]:tgt_face_box[i][2]]
            # print(face.size())
            face = F.interpolate(face.unsqueeze(0), size=256)
            gen_face[i] = face
        return gen_face

    def paste_face(self, img, box, face):
        for i in range(img.size(0)):
            img[i][:, box[i][1]:box[i][3], box[i][0]:box[i][2]] = F.interpolate(face[i].unsqueeze(0), size=(box[i][3]-box[i][1], box[i][2]-box[i][0]))
        return

    def train(self, dl, epoch):  # i -- current epoch
        cnt = 0
        # loss_face_D_sum, loss_D_real_sum, loss_D_fake_sum, loss_D_sum, loss_G_gan_sum, loss_G_img_sum, loss_G_sum = 0, 0, 0, 0, 0, 0, 0
        loss_Face_D_sum, loss_G_face_gan_sum, loss_G_face_vgg_sum, loss_Face_G_sum, loss_G_face_L1_sum = 0, 0, 0, 0, 0
        for iter, (src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, src_mask_gt, tgt_face, tgt_face_box, src_face_box, tgt_face_heatmap) in enumerate(dl):
            print('epoch:', epoch, 'iter:', iter)
            if self.use_cuda:
                src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, tgt_face, tgt_face_heatmap = src_img.cuda(), y.cuda(), src_pose.cuda(), tgt_pose.cuda(), src_mask_prior.cuda(), x_trans.cuda(), tgt_face.cuda(), tgt_face_heatmap.cuda()

            # out = self.G(src_img, F.interpolate(src_pose, scale_factor=1/2), F.interpolate(tgt_pose, scale_factor=1/2), src_mask_prior, x_trans)
            with torch.no_grad():
                out = self.G(src_img, src_pose, tgt_pose, src_mask_prior, x_trans)
                gen = out[0]
                gen_face = self.crop_face(gen, tgt_face.size(), tgt_face_box)
                src_face = self.crop_face(src_img, tgt_face.size(), src_face_box)
            # face_residual = self.Face_G(torch.cat((src_face, tgt_face_heatmap), dim=1))
            face_residual = self.Face_G(torch.cat((src_face, gen_face, tgt_face_heatmap), dim=1))
            # fined_face = F.tanh(src_face + face_residual)
            fined_face = F.tanh(gen_face + face_residual)
            # fined_face = self.Face_G(src_face, tgt_pose)

            self.optimizer_Face_D.zero_grad()
            loss_Face_D_real = self.face_gan_loss(self.Face_D(torch.cat((src_face, tgt_face, tgt_face_heatmap), dim=1)), 1)
            loss_Face_D_fake = self.face_gan_loss(self.Face_D(torch.cat((src_face, fined_face.detach(), tgt_face_heatmap), dim=1)), 0)
            loss_Face_D = loss_Face_D_real + loss_Face_D_fake
            loss_Face_D_sum += loss_Face_D.item()
            self.update_Face_D(loss_Face_D, epoch)

            self.optimizer_Face_G.zero_grad()
            loss_G_face_gan = self.face_gan_loss(self.Face_D(torch.cat((src_face, fined_face, tgt_face_heatmap), dim=1)), 1)
            loss_G_face_vgg = self.G_loss(fined_face, tgt_face)
            loss_G_face_L1 = self.G_loss(fined_face, tgt_face)
            loss_Face_G = loss_G_face_L1 + loss_G_face_gan + loss_G_face_vgg
            # loss_Face_G = loss_G_face_L1 + loss_G_face_vgg
            self.update_Face_G(loss_Face_G, epoch)

            with torch.no_grad():
                self.paste_face(gen, tgt_face_box, fined_face)

            # loss_D_real_sum += loss_D_real.item()
            # loss_D_fake_sum += loss_D_fake.item()
            # loss_G_face_vgg_sum += loss_G_face_vgg.item()

            loss_G_face_gan_sum += loss_G_face_gan.item()
            loss_G_face_L1_sum += loss_G_face_L1.item()
            loss_Face_G_sum += loss_Face_G.item()
            cnt += 1

            # if epoch % self.save_freq == 0 and iter < 3:
            #     self.writer.add_images('gen/epoch%d'%epoch, gen*0.5+0.5)
            #     self.writer.add_images('y/epoch%d'%epoch, y*0.5+0.5)
            #     self.writer.add_images('src_mask/epoch%d'%epoch, out[2].view((out[2].size(0)*out[2].size(1), 1, out[2].size(2), out[2].size(3))))
            #     self.writer.add_images('warped/epoch%d'%epoch, out[3].view((out[3].size(0)*11, 3, out[3].size(2), out[3].size(3)))*0.5+0.5)

        # self.writer.add_scalar('loss_D_real', loss_D_real_sum/cnt, epoch)
        # self.writer.add_scalar('loss_D_fake', loss_D_fake_sum/cnt, epoch)
        self.writer.add_scalar('loss_Face_D', loss_Face_D_sum/cnt, epoch)
        self.writer.add_scalars('DG', {'Face_D': loss_Face_D_sum/cnt, 'Face_G': loss_Face_G_sum/cnt}, epoch)
        self.writer.add_scalar('loss_G_face_gan', loss_G_face_gan_sum/cnt, epoch)
        # self.writer.add_scalar('loss_G_face_vgg', loss_G_face_vgg_sum/cnt, epoch)
        self.writer.add_scalar('loss_G_face_L1', loss_G_face_L1_sum/cnt, epoch)
        self.writer.add_scalar('loss_Face_G', loss_Face_G_sum/cnt, epoch)
        
        self.writer.add_images('train_gen_face/epoch%d'%epoch, gen_face*0.5+0.5)
        self.writer.add_images('train_gen/epoch%d'%epoch, gen*0.5+0.5)
        self.writer.add_images('train_fined_face/epoch%d'%epoch, fined_face*0.5+0.5)
        self.writer.add_images('train_tgt_face/epoch%d'%epoch, tgt_face*0.5+0.5)
        self.writer.add_images('train_tgt_face_heatmap/epoch%d'%epoch, tgt_face_heatmap)
        self.writer.add_images('train_src_face/epoch%d'%epoch, src_face*0.5+0.5)
        

        if epoch % self.save_freq == 0:
            torch.save(self.Face_G.state_dict(), os.path.join(self.save_dir, 'faceg_epoch_%d.pth'%epoch))
            if epoch % self.d_update_freq == 0:
                torch.save(self.Face_D.state_dict(), os.path.join(self.save_dir, 'faced_epoch_%d.pth'%epoch))

    def test(self, test_dl, epoch):
        self.G.eval()
        for iter, (src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, src_mask_gt, tgt_face, tgt_face_box, src_face_box, tgt_face_heatmap) in enumerate(test_dl):
            print('test', 'epoch:', epoch, 'iter:', iter)
            if self.use_cuda:
                src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, tgt_face, tgt_face_heatmap = src_img.cuda(), y.cuda(), src_pose.cuda(), tgt_pose.cuda(), src_mask_prior.cuda(), x_trans.cuda(), tgt_face.cuda(), tgt_face_heatmap.cuda()
            with torch.no_grad():
                # src_pose = F.interpolate(src_pose, scale_factor=1/2)
                # tgt_pose = F.interpolate(tgt_pose, scale_factor=1/2)
                out = self.G(src_img, src_pose, tgt_pose, src_mask_prior, x_trans)
                gen = out[0]
                gen_face = self.crop_face(gen, tgt_face.size(), tgt_face_box)
                src_face = self.crop_face(src_img, tgt_face.size(), src_face_box)
                face_residual = self.Face_G(torch.cat((src_face, gen_face, tgt_face_heatmap), dim=1))
                fined_face = F.tanh(gen_face + face_residual)
                self.paste_face(gen, tgt_face_box, fined_face)
            if iter == 0:
                self.writer.add_images('test_gen/epoch%d'%epoch, gen*0.5+0.5)
                self.writer.add_images('test_tgt_img/epoch%d'%epoch, y*0.5+0.5)
                self.writer.add_images('test_src_img/epoch%d'%epoch, src_img*0.5+0.5)
                self.writer.add_images('test_gen_face/epoch%d'%epoch, gen_face*0.5+0.5)
                self.writer.add_images('test_fined_face/epoch%d'%epoch, fined_face*0.5+0.5)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--face_g_lr', type=float, default=1e-5)
    parser.add_argument('--face_d_lr', type=float, default=1e-5)
    # parser.add_argument('--face_d_lr', type=float, default=1e-5)
    parser.add_argument('--n_epoch', type=int, default=5000)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--d_update_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='7ch-face')
    # parser.add_argument('--g_weight_dir', type=str, default='/versa/kangliwei/motion_transfer/0428-vgg/epoch_600.pth')
    parser.add_argument('--g_weight_dir', type=str, default='/versa/kangliwei/motion_transfer/0605-gan/g_epoch_610.pth')
    # parser.add_argument('--face_g_weight_dir', type=str, default='/versa/kangliwei/motion_transfer/0605-face-dlib/faceg_epoch_1050.pth')
    parser.add_argument('--face_g_weight_dir', type=str)
    # parser.add_argument('--d_weight_dir', type=str, default='/versa/kangliwei/motion_transfer/0505-prog-512-gan/d_epoch_230.pth')
    parser.add_argument('--d_weight_dir', type=str)
    # parser.add_argument('--face_d_weight_dir', type=str, default='/versa/kangliwei/motion_transfer/0605-face-dlib/faced_epoch_1050.pth')
    parser.add_argument('--face_d_weight_dir', type=str)
    parser.add_argument('--complete', action='store_true')
    parser.add_argument('--full_y', action='store_true')
    parser.add_argument('--no_shuffle', action='store_true')

    args = parser.parse_args()

    n_epoch = args.n_epoch
    bs = args.bs
    start_epoch = args.start_epoch
    mini = not args.complete
    full_y = args.full_y
    print('mini', mini)
    shuffle = not args.no_shuffle
    args.use_cuda = True

    params = get_general_params()
    params['IMG_HEIGHT'] = 256
    params['IMG_WIDTH'] = 256
    params['posemap_downsample'] = 2
    GAN = gan(params, args)

    ds = mtdataset(params, mini=mini, full_y=full_y)
    dl = DataLoader(ds, bs, shuffle)

    test_ds = mtdataset(params, mini=False, full_y=full_y, mode='test')
    test_dl = DataLoader(test_ds, 16, shuffle=False)

    for epoch in range(args.start_epoch+1, args.n_epoch+args.start_epoch):
        if epoch == 1:
            GAN.test(test_dl, epoch)
        GAN.train(dl, epoch)
        if epoch % 10 == 0:
            GAN.test(test_dl, epoch)