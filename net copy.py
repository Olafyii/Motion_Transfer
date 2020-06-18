'''
    deprecated.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from dataset import mtdataset
from network import U_Net

class MModel(nn.Module):
    def __init__(self, params, use_cuda=False):
        super(MModel, self).__init__()
        self.params = params
        self.src_mask_delta_UNet = U_Net(params, img_ch=3, output_ch=11)
        self.FG_UNet = U_Net(params, img_ch=30, output_ch=3)
        # self.BG_UNet = U_Net(params, img_ch=4, output_ch=3)
        self.use_cuda = use_cuda

    def meshgrid(self, height, width):
        x_t = torch.matmul(torch.ones((height, 1)), \
                            torch.unsqueeze(torch.linspace(0.0, width-1.0, width), 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0,
                        height-1.0, height), 1),
                        torch.ones((1, width)))
        return x_t, y_t

    def repeat(self, x, n_repeats):
        '''
            x       (img_h*img_w,)  values: [0, 1*h*w, 2*h*w, ..., (bs-1)*h*w] dtype: int64
        '''
        rep = torch.transpose(torch.unsqueeze(torch.ones((n_repeats)), 1), 0, 1)  # (1, n_repeats)
        rep = rep.long()
        x = torch.matmul(x.view((-1, 1)), rep)  # (img_h*img_w, n_repeats)
        return x.flatten()  # (img_h*img_w*n_repeats,)
    
    def interpolate(self, im, x, y):
        '''
            im      (bs, 3, img_h, img_w)
            x       (bs, img_h, img_w)  dtype: float32
            y       (bs, img_h, img_w)  dtype: float32
        '''
        # im = F.pad(im, (1,1,1,1), mode='reflect')  # (bs, 3, img_h+2, img_w+2)
        num_batch = im.size(0)
        height = im.size(2)
        width = im.size(3)
        channels = im.size(1)

        out_height = x.size(1)
        out_width = x.size(2)
                    
        x = x.flatten()  # (bs*img_h*img_w)
        y = y.flatten()  # (bs*img_h*img_w)
                    
        x = x+1
        y = y+1
                    
        max_x = width - 1
        max_y = height - 1

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
                    
        x0 = torch.clamp(x0, 0, max_x)  # (bs*img_h*img_w)
        x1 = torch.clamp(x1, 0, max_x)  # (bs*img_h*img_w)
        y0 = torch.clamp(y0, 0, max_y)  # (bs*img_h*img_w)
        y1 = torch.clamp(y1, 0, max_y)  # (bs*img_h*img_w)
                    
        base = self.repeat(torch.arange(0, num_batch)*width*height, (out_height*out_width))  # (bs*img_h*img_w) dtype:int64
        if self.use_cuda:
            base = base.cuda()

        base_y0 = base + y0*width  # (bs*img_h*img_w)
        base_y1 = base + y1*width  # (bs*img_h*img_w)

        idx_a = base_y0 + x0  # (bs*img_h*img_w)
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1
                    
        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im = im.permute(0, 2, 3, 1)  # (bs, img_h, img_w, 3)
        im_flat = im.reshape((-1, channels))  # (bs*img_h*img_w, 3)

        idx_a = torch.repeat_interleave(idx_a.unsqueeze(-1), channels, -1)  # (bs*img_h*img_w, channels)
        idx_b = torch.repeat_interleave(idx_b.unsqueeze(-1), channels, -1)  # (bs*img_h*img_w, channels)
        idx_c = torch.repeat_interleave(idx_c.unsqueeze(-1), channels, -1)  # (bs*img_h*img_w, channels)
        idx_d = torch.repeat_interleave(idx_d.unsqueeze(-1), channels, -1)  # (bs*img_h*img_w, channels)

        Ia = torch.gather(im_flat, dim=0, index=idx_a)  # (bs*img_h*img_w, channels)
        Ib = torch.gather(im_flat, dim=0, index=idx_b)
        Ic = torch.gather(im_flat, dim=0, index=idx_c)
        Id = torch.gather(im_flat, dim=0, index=idx_d)
                    
        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()
                    
        dx = x1_f - x  # (bs*img_h*img_w)
        dy = y1_f - y  # (bs*img_h*img_w)

        wa = torch.unsqueeze((dx * dy), 1)  # (bs*img_h*img_w, 1)
        wb = torch.unsqueeze((dx * (1-dy)), 1)  # (bs*img_h*img_w, 1)
        wc = torch.unsqueeze(((1-dx) * dy), 1)  # (bs*img_h*img_w, 1)
        wd = torch.unsqueeze(((1-dx) * (1-dy)), 1)  # (bs*img_h*img_w, 1)

        output = wa*Ia + wb*Ib + wc*Ic + wd*Id  # (bs*img_h*img_w, 3)
        output = output.view((-1,out_height,out_width,channels))
        
        return output.permute((0, 3, 1, 2))

    def affine_warp(self, im, theta):
        '''
            im      (bs, 3, img_h, img_w)
            theta   (bs, 2, 3)
        '''
        num_batch = im.size(0)
        height = im.size(2)
        width = im.size(3)

        grid = F.affine_grid(theta, im.size())

        return F.grid_sample(im, grid)
    # def affine_warp(self, im, theta):
    #     '''
    #         im      (bs, 3, img_h, img_w)
    #         theta   (bs, 2, 3)
    #     '''
    #     num_batch = im.size(0)
    #     height = im.size(2)
    #     width = im.size(3)

    #     x_t, y_t = self.meshgrid(height, width)  # (img_h, img_w)   (img_h, img_w)
    #     x_t_flat = x_t.view((1, -1))  # (1, img_h*img_w)
    #     y_t_flat = y_t.view((1, -1))  # (1, img_h*img_w)
    #     ones = torch.ones_like(x_t_flat)  # (1, img_h*img_w)
    #     grid = torch.cat((x_t_flat, y_t_flat, ones), 0)  # (3, img_h*img_w)
    #     grid = grid.flatten()  # (3*img_h*img_w)
    #     grid = grid.repeat(num_batch)  # (bs*3*img_h*img_w)
    #     grid = grid.view((num_batch, 3, -1))  # (bs, 3, img_h*img_w)

    #     if self.use_cuda:
    #         T_g = torch.matmul(theta, grid.cuda())  # (bs, 2, img_h*img_w)
    #     else:
    #         T_g = torch.matmul(theta, grid)  # (bs, 2, img_h*img_w)
    #     x_s = T_g[:, 0, :]  # (bs, 1, img_h*img_w)
    #     y_s = T_g[:, 1, :]  # (bs, 1, img_h*img_w)

    #     x_s = x_s.view((num_batch, height, width))  # (bs, img_h, img_w)
    #     y_s = y_s.view((num_batch, height, width))  # (bs, img_h, img_w)

    #     torch_grid = F.affine_grid(theta, im.size())

    #     return self.interpolate(im, x_s, y_s)

    def make_warped_stack(self, mask, src_img, trans):
        '''
            mask        (bs, 11, img_h, img_w)
            src_img     (bs, 3, img_h, img_w)
            trans       (bs, 11, 2, 3)
        '''
        for i in range(11):
            mask_i = torch.repeat_interleave(torch.unsqueeze(mask[:, i, :, :], 1), 3, 1)  # (bs, 3, img_h, img_w)
            src_masked = torch.mul(mask_i, src_img)  # (bs, 3, img_h, img_w)

            if i == 0:
                warps = src_masked
            else:
                warp_i = self.affine_warp(src_masked, trans[:, i, :, :])  # (bs, 3, img_h, img_w)
                print('src_masked.max(), warp_i.max()', src_masked.max(), warp_i.max())
                warps = torch.cat((warps, warp_i), 1)
        return warps

    
    def forward(self, src_img, src_posemap, tgt_posemap, src_mask_prior, x_trans):
        src_mask_delta = self.src_mask_delta_UNet(src_img, src_posemap)
        src_mask = torch.add(src_mask_delta, src_mask_prior)  # (bs, 11, img_h, img_w)
        src_mask = F.softmax(src_mask)
        src_mask = src_mask_prior

        warped_stack = self.make_warped_stack(src_mask, src_img, x_trans)  # (bs, 33, img_h, img_w)
        warped_stack_limbs = warped_stack[:, 3:, :, :]  # (bs, 30, img_h, img_w)
        # bg_src = warped_stack[:, :3, :, :]  # (bs, 3, img_h, img_w)
        # bg_src_mask = src_mask[:, 0, :, :].unsqueeze(0)  # (bs, 1, img_h, img_w)

        # bg_tgt = self.BG_UNet(torch.cat((bg_src, bg_src_mask), dim=1), src_posemap)

        FG = self.FG_UNet(warped_stack_limbs, tgt_posemap)
        FG = F.tanh(FG)

        
        return FG, src_mask_delta, src_mask, warped_stack


if __name__ == '__main__':
    import cv2
    from param import get_general_params
    params = get_general_params()

    ds = mtdataset(params, full_y=False)
    dl = DataLoader(ds, 1, True)

    model = MModel(params, use_cuda=True).cuda()
    
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('runs/debug')
    for src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, src_mask_gt in dl:
        src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, src_mask_gt = src_img.cuda(), y.cuda(), src_pose.cuda(), tgt_pose.cuda(), src_mask_prior.cuda(), x_trans.cuda(), src_mask_gt.cuda()
        out = model(src_img, src_pose, tgt_pose, src_mask_prior, x_trans)
        writer.add_images('src_img', src_img*0.5+0.5)
        warped = out[-1][0].view(11, 3, out[-1][0].size(1), out[-1][0].size(2))
        # print(warped.size(), warped.max(), warped.min(), warped.dtype)
        # print(src_img.size(), src_img.max(), src_img.min(), src_img.dtype)
        # writer.add_images('warped_stack', warped[0].unsqueeze(0))
        writer.add_images('warped_stack', warped*0.5+0.5)
        idx = torch.Tensor([1,2,3,4,5,6,7,8,9,10]).long()
        warped = out[-1][0]
        img = torch.cat((warped[idx*3].sum(dim=0).unsqueeze(0), warped[idx*3+1].sum(dim=0).unsqueeze(0), warped[idx*3+2].sum(dim=0).unsqueeze(0)), dim=0)
        writer.add_image('sum_warped_stack', img*0.5+0.5)
        writer.add_images('y', y*0.5+0.5)
        writer.add_images('src_mask', out[2].view(11,1,256,256))
        print(warped.max(dim=1)[0].max(dim=1))
        print(warped.min(dim=1)[0].min(dim=1))
        print(warped.mean(dim=1).mean(dim=1))
        
        for i in range(10000000000):
            36**3
        break
        # for i in range(11):
        #     img = out[0, i*3:i*3+3, :, :].permute(1, 2, 0).detach().numpy()
        #     cv2.imwrite('%d.jpg'%i, img)


