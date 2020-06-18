'''
    加一个人脸判别器来训练大generator，deprecated.
'''
from net import MModel
from dataset import mtdataset
from vgg_loss import VGGPerceptualLoss
from torch.utils.tensorboard import SummaryWriter
from discriminator import Discriminator, FaceDisc
# from discriminator_progressive import Discriminator
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import argparse
from param import get_general_params
from torch.utils.data import DataLoader
from torchvision import transforms

class gan(nn.Module):
    # def __init__(self, params, save_dir, g_weight_dir, d_weight_dir, d_update_freq=1, start_epoch=0, g_lr=2e-4, d_lr=2e-4, use_cuda=True):
    def __init__(self, params, args):
        super(gan, self).__init__()
        self.G = MModel(params, use_cuda=True)
        self.D = Discriminator(params, bias=True)
        self.Face_D = FaceDisc()
        self.vgg_loss = VGGPerceptualLoss()
        self.L1_loss = nn.L1Loss()
        if args.use_cuda:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
            self.Face_D = self.Face_D.cuda()
            self.vgg_loss = self.vgg_loss.cuda()
            self.L1_loss = self.L1_loss.cuda()
        if args.g_weight_dir:
            self.G.load_state_dict(torch.load(args.g_weight_dir), strict=True)
        if args.d_weight_dir:
            self.D.load_state_dict(torch.load(args.d_weight_dir), strict=False)
        
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=args.g_lr)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=args.d_lr)
        self.optimizer_Face_D = torch.optim.Adam(self.Face_D.parameters(), lr=args.face_d_lr)

        self.save_dir = args.save_dir
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        self.d_update_freq = args.d_update_freq
        self.save_freq = args.save_freq
        self.writer = SummaryWriter('runs/'+args.save_dir)
        self.use_cuda = args.use_cuda
        self.start_epoch = args.start_epoch
    
    def set_lr(self, op, lr):
        # op: {D|G|Face_D} or optim object
        if op == 'G':
            for g in self.optimizer_G.param_groups:
                g['lr'] = lr
            return None
        if op == 'D':
            for g in self.optimizer_D.param_groups:
                g['lr'] = lr
            return None
        if op == 'Face_D':
            for g in self.optimizer_Face_D.param_groups:
                g['lr'] = lr
            return None
        for g in op.param_groups:
            g['lr'] = lr

    def G_loss(self, input, target):
        vgg = self.vgg_loss(input, target)
        # L1 = self.L1_loss(input, target)
        return vgg
        # return vgg + L1
    
    def update_D(self, loss, epoch):
        if epoch % self.d_update_freq == 0:
            loss.backward()
            self.optimizer_D.step()
    
    def update_G(self, loss, epoch):
        if epoch >= 10:
            loss.backward()
            self.optimizer_G.step()

    def update_Face_D(self, loss, epoch):
        loss.backward()
        self.optimizer_Face_D.step()

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
    
    def get_face_lambda(self, epoch):
        return 1
        if epoch in range(10, 50):
            return 0.0001
        if epoch in range(50, 70):
            return 0.001
        if epoch in range(70, 80):
            return 0.01
        if epoch in range(80, 90):
            return 0.1
        return 1

    def train(self, dl, epoch):
        cnt = 0
        loss_face_D_sum, loss_D_real_sum, loss_D_fake_sum, loss_D_sum, loss_G_gan_sum, loss_G_img_sum, loss_G_sum = 0, 0, 0, 0, 0, 0, 0
        for iter, (src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, src_mask_gt, tgt_face, tgt_face_box, src_face_box) in enumerate(dl):
            print('epoch:', epoch, 'iter:', iter)
            if self.use_cuda:
                src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, tgt_face = src_img.cuda(), y.cuda(), src_pose.cuda(), tgt_pose.cuda(), src_mask_prior.cuda(), x_trans.cuda(), tgt_face.cuda()

            # out = self.G(src_img, F.interpolate(src_pose, scale_factor=1/2), F.interpolate(tgt_pose, scale_factor=1/2), src_mask_prior, x_trans)
            out = self.G(src_img, src_pose, tgt_pose, src_mask_prior, x_trans)
            gen = out[0]
            gen_face = self.crop_face(gen, tgt_face.size(), tgt_face_box)
            
            self.optimizer_D.zero_grad()
            loss_D_real = self.gan_loss(self.D(y, tgt_pose), 1, tgt_pose)
            loss_D_fake = self.gan_loss(self.D(gen.detach(), tgt_pose), 0, tgt_pose)
            loss_D = loss_D_real + loss_D_fake
            loss_D_sum += loss_D.item()
            self.update_D(loss_D, epoch)

            self.optimizer_Face_D.zero_grad()
            pred_tgt = self.Face_D(tgt_face)
            pred_gen = self.Face_D(gen_face.detach())
            print('pred_tgt', pred_tgt)
            print('pred_gen', pred_gen)
            loss_D_real = self.face_gan_loss(pred_tgt, 1)
            loss_D_fake = self.face_gan_loss(pred_gen, 0)
            loss_face_D = loss_D_real + loss_D_fake
            loss_face_D_sum += loss_face_D.item()
            self.update_Face_D(loss_face_D, epoch)

            self.optimizer_G.zero_grad()
            loss_G_gan = self.gan_loss(self.D(gen, tgt_pose), 1, tgt_pose)
            loss_G_face_gan = self.face_gan_loss(self.Face_D(gen_face), 1)
            loss_G_img = self.G_loss(gen, y)  # vgg_loss + L1_loss
            lmd = self.get_face_lambda(epoch)
            loss_G = loss_G_gan + loss_G_img + lmd * loss_G_face_gan
            # loss_G = loss_G_face_gan
            self.update_G(loss_G, epoch)

            # loss_D_real_sum += loss_D_real.item()
            # loss_D_fake_sum += loss_D_fake.item()
            loss_G_gan_sum += loss_G_gan.item()
            loss_G_img_sum += loss_G_img.item()
            loss_G_sum += loss_G.item()
            cnt += 1

            # if epoch % self.save_freq == 0 and iter < 3:
            #     self.writer.add_images('gen/epoch%d'%epoch, gen*0.5+0.5)
            #     self.writer.add_images('y/epoch%d'%epoch, y*0.5+0.5)
            #     self.writer.add_images('src_mask/epoch%d'%epoch, out[2].view((out[2].size(0)*out[2].size(1), 1, out[2].size(2), out[2].size(3))))
            #     self.writer.add_images('warped/epoch%d'%epoch, out[3].view((out[3].size(0)*11, 3, out[3].size(2), out[3].size(3)))*0.5+0.5)

        # self.writer.add_scalar('loss_D_real', loss_D_real_sum/cnt, epoch)
        # self.writer.add_scalar('loss_D_fake', loss_D_fake_sum/cnt, epoch)
        self.writer.add_scalar('loss_D', loss_D_sum/cnt, epoch)
        self.writer.add_scalar('loss_face_D', loss_face_D_sum/cnt, epoch)
        self.writer.add_scalars('DG', {'D': loss_D/cnt, 'face_D': loss_face_D_sum/cnt, 'G': loss_G_sum/cnt}, epoch)
        self.writer.add_scalar('loss_G_gan', loss_G_gan_sum/cnt, epoch)
        self.writer.add_scalar('loss_G_img', loss_G_img_sum/cnt, epoch)
        self.writer.add_scalar('loss_G', loss_G_sum/cnt, epoch)
        
        self.writer.add_images('train_gen_face/epoch%d'%epoch, gen_face*0.5+0.5)
        self.writer.add_images('train_tgt_face/epoch%d'%epoch, tgt_face*0.5+0.5)
        self.writer.add_images('train_src/epoch%d'%epoch, src_img*0.5+0.5)
        self.writer.add_images('train_tgt/epoch%d'%epoch, y*0.5+0.5)
        self.writer.add_images('train_gen/epoch%d'%epoch, gen*0.5+0.5)        

        if epoch % self.save_freq == 0:
            torch.save(self.G.state_dict(), os.path.join(self.save_dir, 'g_epoch_%d.pth'%epoch))
            torch.save(self.D.state_dict(), os.path.join(self.save_dir, 'd_epoch_%d.pth'%epoch))
            torch.save(self.Face_D.state_dict(), os.path.join(self.save_dir, 'faced_epoch_%d.pth'%epoch))

    def test(self, test_dl, epoch):
        self.G.eval()
        for iter, (src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, src_mask_gt, tgt_face, tgt_face_box, src_face_box) in enumerate(test_dl):
            print('test', 'epoch:', epoch, 'iter:', iter)
            if self.use_cuda:
                src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans = src_img.cuda(), y.cuda(), src_pose.cuda(), tgt_pose.cuda(), src_mask_prior.cuda(), x_trans.cuda()
            with torch.no_grad():
                # src_pose = F.interpolate(src_pose, scale_factor=1/2)
                # tgt_pose = F.interpolate(tgt_pose, scale_factor=1/2)
                out = self.G(src_img, src_pose, tgt_pose, src_mask_prior, x_trans)
            gen = out[0]
            if iter == 0:
                self.writer.add_images('test_gen/epoch%d'%epoch, gen*0.5+0.5)
                self.writer.add_images('test_y/epoch%d'%epoch, y*0.5+0.5)
                self.writer.add_images('test_src/epoch%d'%epoch, src_img*0.5+0.5)
                self.writer.add_images('test_src_mask/epoch%d'%epoch, out[2].view((out[2].size(0)*out[2].size(1), 1, out[2].size(2), out[2].size(3))))
                self.writer.add_images('test_warped/epoch%d'%epoch, out[3].view((out[3].size(0)*11, 3, out[3].size(2), out[3].size(3)))*0.5+0.5)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--g_lr', type=float, default=2e-4)
    parser.add_argument('--d_lr', type=float, default=2e-4)
    parser.add_argument('--face_d_lr', type=float, default=1e-5)
    parser.add_argument('--n_epoch', type=int, default=5000)
    parser.add_argument('--bs', type=int, default=6)
    parser.add_argument('--d_update_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='0513-plan1')
    parser.add_argument('--g_weight_dir', type=str, default='/versa/kangliwei/motion_transfer/0428-vgg/epoch_600.pth')
    # parser.add_argument('--d_weight_dir', type=str, default='/versa/kangliwei/motion_transfer/0505-prog-512-gan/d_epoch_230.pth')
    parser.add_argument('--d_weight_dir', type=str)
    parser.add_argument('--complete', action='store_true')
    parser.add_argument('--full_y', action='store_true')
    parser.add_argument('--no_shuffle', action='store_true')

    args = parser.parse_args()

    n_epoch = args.n_epoch
    bs = args.bs
    start_epoch = args.start_epoch
    mini = not args.complete
    full_y = args.full_y
    print(full_y)
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
        # if epoch == 1:
        #     GAN.test(test_dl, epoch)
        # if epoch in range(10, 50):
        #     GAN.set_lr('Face_D', 1e-8)
        #     print("GAN.set_lr('Face_D', 1e-8)")
        # if epoch in range(50, 70):
        #     GAN.set_lr('Face_D', 1e-7)
        #     print("GAN.set_lr('Face_D', 1e-7)")
        # if epoch in range(70, 80):
        #     GAN.set_lr('Face_D', 1e-6)
        #     print("GAN.set_lr('Face_D', 1e-6)")
        # if epoch in range(80, 90):
        #     GAN.set_lr('Face_D', 1e-5)
        #     print("GAN.set_lr('Face_D', 1e-5)")
        # if epoch in range(90, 100):
        #     GAN.set_lr('Face_D', 1e-4)
        #     print("GAN.set_lr('Face_D', 1e-4)")
        GAN.train(dl, epoch)
        if epoch % 10 == 0:
            GAN.test(test_dl, epoch)