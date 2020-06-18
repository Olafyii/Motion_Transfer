'''
    just for debug
'''
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
import numpy as np
import scipy.io as sio
from param import get_general_params

class mtdataset(Dataset):
    def __init__(self, params, user_path, user_id, dancer_path):
        super(mtdataset, self).__init__()
        '''
        mode = {train | test}
        type = {posewarp | Versa_masked | Versa_mask}
        '''
        data_path = '/versa/kangliwei/motion_transfer/data/posewarp/test/frames'
        if user_path == r"[Golf with Aimee] Aimee's Golf Lesson 022_ Lagging & Casting":
            self.user_path = data_path+'/'+user_path+'/'+str(user_id)+'.png'
        else:            
            self.user_path = glob.glob(data_path+'/'+user_path+'/'+str(user_id)+'*')[0]
        self.dancer_path_list = sorted(self.myglob(data_path+'/'+dancer_path), key=lambda s: int(os.path.split(s)[-1].split('.')[0]))
        self.user_id = user_id
        self.params = params

        dancer_mat = sio.loadmat(self.img2mat_path(self.dancer_path_list[0]))['data']['bbox'][0][0]
        self.left = dancer_mat[:, 0].min()
        self.top = dancer_mat[:, 1].min()
        self.right = (dancer_mat[:, 0] + dancer_mat[:, 2]).max()
        self.bottom = (dancer_mat[:, 1] + dancer_mat[:, 3]).max()
        center = [(self.right+self.left)/2, (self.bottom+self.top)/2]
        img = Image.open(self.dancer_path_list[0]).convert('RGB')
        radius = min(int(max(self.right-self.left, self.bottom-self.top) * 1.2 / 2), center[0], img.width - center[0], center[1], img.height - center[1])
        self.left = center[0]-radius
        self.right = center[0]+radius
        self.top = center[1]-radius
        self.bottom = center[1]+radius

        self.transform3 = transforms.Compose([
            transforms.Resize(params['IMG_HEIGHT']),
            transforms.ToTensor(), 
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # set the range of image tensor to (-1, 1)
        ])

        self.transform1 = transforms.Compose([
            transforms.Resize(params['IMG_HEIGHT']),
            transforms.ToTensor(), 
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.5], std=[0.5])  # set the range of image tensor to (-1, 1)
        ])

    def img2mat_path(self, path):
        path = os.path.split(path)[0]
        path = 'info'.join(path.split('frames'))
        path = path + '.mat'
        return path

    def __len__(self):
        return len(self.dancer_path_list)
    def __getitem__(self, idx):
        info = sio.loadmat(self.img2mat_path(self.user_path))

        # load .mat file, get bbox width and height, and joints
        box = info['data']['bbox'][0][0]  # bbox width and height, size (n_frames, 4)
        # (left_corner_x, left_corner_y, delta_x, delta_y)
        x = info['data']['X'][0][0]  # joints, size(14, 2, n_frames) (14 joints)

        src_img = Image.open(self.user_path).convert('RGB')
        src_mask_gt = Image.open('Versa_mask'.join(self.user_path.split('posewarp'))).convert('L')
        bbox = box[self.user_id-1]
        src_center = [int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2)]
        src_radius = min(int(max(bbox[2], bbox[3]) * 1.2 / 2), src_center[0], src_img.width - src_center[0], src_center[1], src_img.height - src_center[1])

        left = src_center[0]-src_radius
        top = src_center[1]-src_radius
        right = src_center[0]+src_radius
        bottom = src_center[1]+src_radius

        src_img = src_img.crop((left, top, right, bottom))
        src_mask_gt = src_mask_gt.crop((left, top, right, bottom))
        # print('src ltrb:', left, top, right, bottom)
        # print('src_img_size', src_img.size)
        src_joints = x[:, :, self.user_id-1] - 1.0  # (14, 2)
        src_joints[:, 0] = (src_joints[:, 0] - left) * (self.params['IMG_WIDTH']/(2*src_radius))
        src_joints[:, 1] = (src_joints[:, 1] - top) * (self.params['IMG_HEIGHT']/(2*src_radius))
        src_posemap = self.make_joint_heatmaps(self.params['IMG_HEIGHT'], self.params['IMG_WIDTH'], src_joints, self.params['sigma_joint'], self.params['posemap_downsample'])
        src_limb_masks = self.make_limb_masks(self.params['limbs'], src_joints, self.params['IMG_WIDTH'], self.params['IMG_HEIGHT'])
        src_bg_mask = np.expand_dims(1.0 - np.amax(src_limb_masks, axis=2), 2)
        # src_masks_prior = np.log(np.concatenate((src_bg_mask, src_limb_masks), axis=2) + 1e-10)
        src_masks_prior = np.log(np.concatenate((src_bg_mask, src_limb_masks), axis=2) + 1e-10)

        # DEBUG
        # print('src_masks_prior', src_masks_prior[:, :, 0].max(), src_masks_prior[:, :, 0].min(), src_masks_prior.shape)
        # print('src_limb_masks', src_limb_masks.max(), src_limb_masks.min(), src_limb_masks.shape)
        # print('src_bg_mask', src_bg_mask.max(), src_bg_mask.min(), src_bg_mask.shape)
        # Image.fromarray((src_posemap.sum(axis=2)*255).astype(np.uint8)).save('src_posemap.jpg')
        # Image.fromarray((src_limb_masks[:, :, 0]*255).astype(np.uint8)).save('src_limb_masks.jpg')
        # Image.fromarray((src_masks_prior.sum(axis=2)*255).astype(np.uint8)).save('src_masks_prior.jpg')
        # Image.fromarray((src_masks_prior[:, :, 0]*255).astype(np.uint8)).save('src_masks_prior.jpg')
        # Image.fromarray((src_bg_mask.sum(axis=2)*255).astype(np.uint8)).save('src_bg_mask.jpg')
        # src_img.save('src_img.jpg')
        
        info = sio.loadmat(self.img2mat_path(self.dancer_path_list[idx]))
        box = info['data']['bbox'][0][0]
        x = info['data']['X'][0][0]

        tgt_img = Image.open('Versa_masked'.join(self.dancer_path_list[idx].split('posewarp'))).convert('RGB')

        tgt_img = tgt_img.crop((self.left, self.top, self.right, self.bottom))
        tgt_joints = x[:, :, idx] - 1.0
        tgt_joints[:, 0] = (tgt_joints[:, 0] - self.left) * (self.params['IMG_WIDTH']/(self.right-self.left))
        tgt_joints[:, 1] = (tgt_joints[:, 1] - self.top) * (self.params['IMG_HEIGHT']/(self.bottom-self.top))
        tgt_posemap = self.make_joint_heatmaps(self.params['IMG_HEIGHT'], self.params['IMG_WIDTH'], tgt_joints, self.params['sigma_joint'], self.params['posemap_downsample'])

        x_trans = np.zeros((2, 3, self.params['n_limbs'] + 1))
        x_trans[:, :, 0] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        x_trans[:, :, 1:] = self.get_limb_transforms(self.params['limbs'], src_joints, tgt_joints)

        # src_img, y(tgt_img), src_pose, tgt_pose, src_mask, x_trans, src_mask_gt
        # torch.Size([3, 256, 256])
        # torch.Size([3, 256, 263])
        # torch.Size([14, 256, 256])
        # torch.Size([14, 256, 256])
        # torch.Size([11, 256, 256])
        # torch.Size([11, 2, 3])
        # torch.Size([1, 256, 256])
        return (self.transform3(src_img), self.transform3(tgt_img), torch.from_numpy(src_posemap).permute(2, 0, 1).float(), torch.from_numpy(tgt_posemap).permute(2, 0, 1).float(), torch.from_numpy(src_masks_prior).permute(2, 0, 1).float(), torch.from_numpy(x_trans).permute(2, 0, 1).float(), self.transform1(src_mask_gt))

    def make_joint_heatmaps(self, height, width, joints, sigma, pose_dn):
        height = int(height / pose_dn)
        width = int(width / pose_dn)
        n_joints = joints.shape[0]
        var = sigma ** 2
        joints = joints / pose_dn

        H = np.zeros((height, width, n_joints))

        for i in range(n_joints):
            if (joints[i, 0] <= 0 or joints[i, 1] <= 0 or joints[i, 0] >= width - 1 or
                    joints[i, 1] >= height - 1):
                continue

            H[:, :, i] = self.make_gaussian_map(width, height, joints[i, :], var, var, 0.0)

        return H

    def make_gaussian_map(self, img_width, img_height, center, var_x, var_y, theta):
        xv, yv = np.meshgrid(np.array(range(img_width)), np.array(range(img_height)),
                            sparse=False, indexing='xy')

        a = np.cos(theta) ** 2 / (2 * var_x) + np.sin(theta) ** 2 / (2 * var_y)
        b = -np.sin(2 * theta) / (4 * var_x) + np.sin(2 * theta) / (4 * var_y)
        c = np.sin(theta) ** 2 / (2 * var_x) + np.cos(theta) ** 2 / (2 * var_y)

        return np.exp(-(a * (xv - center[0]) * (xv - center[0]) +
                        2 * b * (xv - center[0]) * (yv - center[1]) +
                        c * (yv - center[1]) * (yv - center[1])))

    def make_limb_masks(self, limbs, joints, img_width, img_height):
        n_limbs = len(limbs)
        mask = np.zeros((img_height, img_width, n_limbs))

        # Gaussian sigma perpendicular to the limb axis.
        sigma_perp = np.array([11, 11, 11, 11, 11, 11, 11, 11, 11, 13]) ** 2

        for i in range(n_limbs):
            n_joints_for_limb = len(limbs[i])
            p = np.zeros((n_joints_for_limb, 2))

            for j in range(n_joints_for_limb):
                p[j, :] = [joints[limbs[i][j], 0], joints[limbs[i][j], 1]]

            if n_joints_for_limb == 4:
                p_top = np.mean(p[0:2, :], axis=0)
                p_bot = np.mean(p[2:4, :], axis=0)
                p = np.vstack((p_top, p_bot))

            center = np.mean(p, axis=0)

            sigma_parallel = np.max([5, (np.sum((p[1, :] - p[0, :]) ** 2)) / 1.5])
            theta = np.arctan2(p[1, 1] - p[0, 1], p[0, 0] - p[1, 0])

            mask_i = self.make_gaussian_map(img_width, img_height, center, sigma_parallel, sigma_perp[i], theta)
            mask[:, :, i] = mask_i / (np.amax(mask_i) + 1e-6)

        return mask

    def get_limb_transforms(self, limbs, joints1, joints2):
        n_limbs = len(limbs)

        Ms = np.zeros((2, 3, n_limbs))

        for i in range(n_limbs):
            n_joints_for_limb = len(limbs[i])
            p1 = np.zeros((n_joints_for_limb, 2))
            p2 = np.zeros((n_joints_for_limb, 2))

            for j in range(n_joints_for_limb):
                p1[j, :] = [joints1[limbs[i][j], 0], joints1[limbs[i][j], 1]]
                p2[j, :] = [joints2[limbs[i][j], 0], joints2[limbs[i][j], 1]]

            tform = self.make_similarity(p2, p1, False)
            Ms[:, :, i] = np.array([[tform[1], -tform[3], tform[0]], [tform[3], tform[1], tform[2]]])

        return Ms

    def make_similarity(self, src, dst,flip=False):
        '''
        Determine parameters of 2D similarity transformation in the order:
            a0, a1, b0, b1
        where the transformation is defined as:
            X = a0 + a1*x - b1*y
            Y = b0 + b1*x + a1*y
        You can determine the over-, well- and under-determined parameters
        with the least-squares method.

        Explicit parameters are in the order:
            a0, b0, m, alpha [radians]
        where the transformation is defined as:
            X = a0 + m*x*cos(alpha) - m*y*sin(alpha)
            Y = b0 + m*x*sin(alpha) + m*y*cos(alpha)

        :param src: :class:`numpy.array`
            Nx2 coordinate matrix of source coordinate system
        :param src: :class:`numpy.array`
            Nx2 coordinate matrix of destination coordinate system

        :returns: params, params_explicit
        '''

        xs = src[:,0]
        ys = src[:,1]
        rows = src.shape[0]
        A = np.zeros((rows*2, 4))
        A[:rows,0] = 1
        A[:rows,1] = xs
        A[:rows,3] = -ys
        A[rows:,2] = 1
        A[rows:,3] = xs
        A[rows:,1] = ys
        
        if(flip):
            A[:rows,3] *= -1.0
            A[rows:,1] *= -1.0

        b = np.zeros((rows*2,))
        b[:rows] = dst[:,0]
        b[rows:] = dst[:,1]
        params = np.linalg.lstsq(A, b)[0]
        '''
        #: determine explicit params
        a0, b0 = params[0], params[2]
        alpha = math.atan2(params[3], params[1])
        m = params[1] / math.cos(alpha)
        params_explicit = np.array([a0, b0, m, alpha])
        '''    

        return params #, params_explicit

    def myglob(self, path):
        ret = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if 'DS_Store' not in file and file[0] != '.':
                    ret += [os.path.join(root, file)]
        return ret

if __name__ == '__main__':
    from net import MModel
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torch.utils.tensorboard import SummaryWriter
    import time

    params = get_general_params()
    params['IMG_HEIGHT'] = 256
    params['IMG_WIDTH'] = 256
    params['posemap_downsample'] = 2
    net = MModel(params, use_cuda=True)
    net.load_state_dict(torch.load('/versa/kangliwei/motion_transfer/0424-gan/g_epoch_2000.pth'), strict=True)
    net = net.cuda()
    net.eval()
    # ds = mtdataset(params, 'Standing Yoga Poses for Hips - Day 10 - The 30 Days of Yoga Challenge', 1, 'CHARLEY HULL 4K UHD SLOW MOTION FACE ON DRIVER GOLF SWING_1')
    ds = mtdataset(params, 'Standing Yoga Poses for Hips - Day 10 - The 30 Days of Yoga Challenge', 1, 'Tennis Tip_ Proper Weight Transfer On Topspin Groundstrokes')
    dl = DataLoader(ds, 1, False)
    writer = SummaryWriter(log_dir='runs/0428-test')
    save_dir = '0428-ours-test-1'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, (src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, src_mask_gt) in enumerate(dl):
        print(i)
        torch.cuda.empty_cache()
        src_img, src_pose, tgt_pose, src_mask_prior, x_trans = src_img.cuda(), src_pose.cuda(), tgt_pose.cuda(), src_mask_prior.cuda(), x_trans.cuda()
        with torch.no_grad():
            out = net(src_img, src_pose, tgt_pose, src_mask_prior, x_trans)
        transforms.ToPILImage()(out[0][0].cpu()*0.5+0.5).save(save_dir+'/generated_%d.jpg'%(i+1))
        transforms.ToPILImage()(src_img[0].cpu()*0.5+0.5).save(save_dir+'/src_%d.jpg'%(i+1))
        transforms.ToPILImage()(y[0].cpu()*0.5+0.5).save(save_dir+'/ground-truth_%d.jpg'%(i+1))
    #     if i == 0:
    #         outs = out[0].cpu()
    #         src_imgs = src_img.cpu()
    #         ys = y.cpu()
    #     else:
    #         outs = torch.cat((outs, out[0].cpu()), dim=0)
    #         src_imgs = torch.cat((src_imgs, src_img.cpu()), dim=0)
    #         ys = torch.cat((ys, y.cpu()), dim=0)
    #     print(outs.device)
    # writer.add_images('out', outs*0.5+0.5)
    # writer.add_images('src_img', src_imgs*0.5+0.5)
    # writer.add_images('y', ys*0.5+0.5)
    # time.sleep(30)
    