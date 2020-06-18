'''
    数据集。
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
import math

class mtdataset(Dataset):
    def __init__(self, params, mode='train', type='posewarp', mini=False, full_y=True):
        super(mtdataset, self).__init__()
        '''
        mode = {train | test}
        type = {posewarp | Versa_masked | Versa_mask}
        '''
        data_path = os.path.join('/versa/kangliwei/motion_transfer/data', type, mode)
        frames = sorted(glob.glob(os.path.join(data_path, 'frames/*')))
        self.frames = [sorted(self.myglob(frames[idx]), key=lambda s: int(os.path.split(s)[-1].split('.')[0])) \
            for idx in range(len(frames))]
        self.info = []
        for f in self.frames:
            self.info.append('/'.join('info'.join(f[0].split('frames')).split('/')[:-1])+'.mat')
        # e.g.
        # /versa/kangliwei/motion_transfer/data/posewarp/train/frames/Ben An golf swing (Driver) face-on view - BMW PGA Wentworth 2016_1/1.jpg
        # |
        # v
        # /versa/kangliwei/motion_transfer/data/posewarp/train/info/Ben An golf swing (Driver) face-on view - BMW PGA Wentworth 2016_1.mat

        self.transform3 = transforms.Compose([
            transforms.Resize((params['IMG_HEIGHT'], params['IMG_WIDTH'])),
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

        self.params = params
        self.mini = mini
        self.full_y = full_y

    def __len__(self):
        if self.mini:
            return 32
        return len(self.frames)

    def get(self, idx, frame_idx=[-1, -1]):
        if self.mini:
            idx = idx%32

        ##################
        # idx = 1
        ##################

        frames = self.frames[idx]
        # print(frames[0])
        info = self.info[idx]

        # select two frames
        if frame_idx[0] == -1 and frame_idx[1] == -1:
            n_frames = len(frames)
            frame_idx = np.random.choice(n_frames, 2, replace=False)
            while abs(frame_idx[0] - frame_idx[1]) / (n_frames * 1.0) <= 0.02:
                frame_idx = np.random.choice(n_frames, 2, replace=False)

        # frame_idx = [0, 23]
        # print('idx:', idx, 'n_frames:', len(frames), 'frame_idx:', frame_idx)
        # print('frames', frames)
        # print('info', info)

        # load .mat file, get bbox width and height, and joints
        info = sio.loadmat(info)
        box = info['data']['bbox'][0][0]  # bbox width and height, size (n_frames, 4)
        # (left_corner_x, left_corner_y, delta_x, delta_y)
        x = info['data']['X'][0][0]  # joints, size(14, 2, n_frames) (14 joints)

        src_img = Image.open(frames[frame_idx[0]]).convert('RGB')
        src_mask_gt = Image.open('Versa_mask'.join(frames[frame_idx[0]].split('posewarp'))).convert('L')
        bbox = box[frame_idx[0]]
        src_center = [int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2)]
        # src_radius = min(int(max(bbox[2], bbox[3]) * 1.2 / 2), src_center[0], src_img.width - src_center[0], src_center[1], src_img.height - src_center[1])
        src_d = min(src_img.width, src_img.height, int(max(bbox[2], bbox[3]) * 1.2))
        src_radius = int(src_d / 2)

        if src_center[0] - src_radius < 0:
            src_center[0] = src_radius
        if src_center[0] + src_radius > src_img.width - 1:
            src_center[0] = src_img.width - src_radius - 1
        if src_center[1] - src_radius < 0:
            src_center[1] = src_radius
        if src_center[1] + src_radius > src_img.height - 1:
            src_center[1] = src_img.height - src_radius - 1

        left = src_center[0]-src_radius
        top = src_center[1]-src_radius
        right = src_center[0]+src_radius
        bottom = src_center[1]+src_radius

        src_img = src_img.crop((left, top, right, bottom))
        src_mask_gt = src_mask_gt.crop((left, top, right, bottom))
        # print('src ltrb:', left, top, right, bottom)
        # print('src_img_size', src_img.size)
        src_joints = x[:, :, frame_idx[0]] - 1.0  # (14, 2)
        src_joints[:, 0] = (src_joints[:, 0] - left)
        src_joints[:, 1] = (src_joints[:, 1] - top)
        src_face_box = self.get_face_box(src_joints, src_img.width, src_img.height)
        src_joints[:, 0] *= (self.params['IMG_WIDTH']/(2*src_radius))
        src_joints[:, 1] *= (self.params['IMG_HEIGHT']/(2*src_radius))
        src_posemap = self.make_joint_heatmaps(self.params['IMG_HEIGHT'], self.params['IMG_WIDTH'], src_joints, self.params['sigma_joint'], self.params['posemap_downsample'])
        src_limb_masks = self.make_limb_masks(self.params['limbs'], src_joints, self.params['IMG_WIDTH'], self.params['IMG_HEIGHT'])
        src_bg_mask = np.expand_dims(1.0 - np.amax(src_limb_masks, axis=2), 2)
        # src_masks_prior = np.log(np.concatenate((src_bg_mask, src_limb_masks), axis=2) + 1e-10)
        src_masks_prior = np.log(np.concatenate((src_bg_mask, src_limb_masks), axis=2) + 1e-10)

        src_face_box = torch.tensor(src_face_box)
        src_face_box[0] *= (self.params['IMG_WIDTH']/(2*src_radius))
        src_face_box[2] *= (self.params['IMG_WIDTH']/(2*src_radius))
        src_face_box[1] *= (self.params['IMG_HEIGHT']/(2*src_radius))
        src_face_box[3] *= (self.params['IMG_HEIGHT']/(2*src_radius))
        src_face_box = src_face_box.long()

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
        
        if self.full_y:
            tgt_img = Image.open(frames[frame_idx[1]]).convert('RGB')
        else:
            tgt_img = Image.open('Versa_masked'.join(frames[frame_idx[1]].split('posewarp'))).convert('RGB')
        bbox = box[frame_idx[1]]
        tgt_center = [int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2)]
        # tgt_radius = min(int(max(bbox[2], bbox[3]) * 1.2 / 2), tgt_center[0], tgt_img.width - tgt_center[0], tgt_center[1], tgt_img.height - tgt_center[1])
        tgt_d = min(tgt_img.width, tgt_img.height, int(max(bbox[2], bbox[3]) * 1.2))
        tgt_radius = int(tgt_d / 2)

        if tgt_center[0] - tgt_radius < 0:
            tgt_center[0] = tgt_radius
        if tgt_center[0] + tgt_radius > tgt_img.width - 1:
            tgt_center[0] = tgt_img.width - tgt_radius - 1
        if tgt_center[1] - tgt_radius < 0:
            tgt_center[1] = tgt_radius
        if tgt_center[1] + tgt_radius > tgt_img.height - 1:
            tgt_center[1] = tgt_img.height - tgt_radius - 1

        left = tgt_center[0]-tgt_radius
        top = tgt_center[1]-tgt_radius
        right = tgt_center[0]+tgt_radius
        bottom = tgt_center[1]+tgt_radius

        tgt_img = tgt_img.crop((left, top, right, bottom))
        tgt_joints = x[:, :, frame_idx[1]] - 1.0
        tgt_joints[:, 0] = (tgt_joints[:, 0] - left)
        tgt_joints[:, 1] = (tgt_joints[:, 1] - top)
        tgt_face_box = self.get_face_box(tgt_joints, tgt_img.width, tgt_img.height)
        # print(tgt_face_box)
        tgt_joints[:, 0] = tgt_joints[:, 0] * (self.params['IMG_WIDTH']/(2*tgt_radius))
        tgt_joints[:, 1] = tgt_joints[:, 1] * (self.params['IMG_HEIGHT']/(2*tgt_radius))
        tgt_posemap = self.make_joint_heatmaps(self.params['IMG_HEIGHT'], self.params['IMG_WIDTH'], tgt_joints, self.params['sigma_joint'], self.params['posemap_downsample'])
        # print('tgt_face_box', tgt_face_box)
        tgt_face = tgt_img.crop(tgt_face_box)
        # h, w = tgt_face.height, tgt_face.width
        # tgt_face = transforms.Resize((int(h*(256/(2*tgt_radius))), int(w*(256/(2*tgt_radius)))))(tgt_face)
        tgt_face_box = torch.tensor(tgt_face_box)
        tgt_face_box[0] *= (self.params['IMG_WIDTH']/(2*tgt_radius))
        tgt_face_box[2] *= (self.params['IMG_WIDTH']/(2*tgt_radius))
        tgt_face_box[1] *= (self.params['IMG_HEIGHT']/(2*tgt_radius))
        tgt_face_box[3] *= (self.params['IMG_HEIGHT']/(2*tgt_radius))
        tgt_face_box = tgt_face_box.long()
        # print(tgt_face.width, tgt_face.height)

        x_trans = np.zeros((2, 3, self.params['n_limbs'] + 1))
        x_trans[:, :, 0] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        x_trans[:, :, 1:] = self.get_limb_transforms(self.params['limbs'], src_joints, tgt_joints)

        # src_img, y(tgt_img), src_pose, tgt_pose, src_mask, x_trans, src_mask_gt, tgt_face, tgt_face_box, src_face_box
        # torch.Size([3, 256, 256])
        # torch.Size([3, 256, 256])
        # torch.Size([14, 128, 128])
        # torch.Size([14, 128, 128])
        # torch.Size([11, 256, 256])
        # torch.Size([11, 2, 3])
        # torch.Size([1, 256, 256])
        # torch.Size([3, 256, 256])
        # torch.Size([4])  (left, top, right, bottom)
        # torch.Size([4])  (left, top, right, bottom)
        return self.transform3(src_img), self.transform3(tgt_img), torch.from_numpy(src_posemap).permute(2, 0, 1).float(), torch.from_numpy(tgt_posemap).permute(2, 0, 1).float(), torch.from_numpy(src_masks_prior).permute(2, 0, 1).float(), torch.from_numpy(x_trans).permute(2, 0, 1).float(), self.transform1(src_mask_gt), self.transform3(tgt_face), tgt_face_box, src_face_box

    def __getitem__(self, idx):
        if self.mini:
            idx = idx%32

        ##################
        # idx = 1
        ##################

        frames = self.frames[idx]
        # print(frames[0])
        info = self.info[idx]

        # select two frames
        n_frames = len(frames)
        frame_idx = np.random.choice(n_frames, 2, replace=False)
        while abs(frame_idx[0] - frame_idx[1]) / (n_frames * 1.0) <= 0.02:
            frame_idx = np.random.choice(n_frames, 2, replace=False)

        # frame_idx = [0, n_frames-2]
        # print('idx:', idx, 'n_frames:', len(frames), 'frame_idx:', frame_idx)
        # print('frames', frames)
        # print('info', info)

        # load .mat file, get bbox width and height, and joints
        info = sio.loadmat(info)
        box = info['data']['bbox'][0][0]  # bbox width and height, size (n_frames, 4)
        # (left_corner_x, left_corner_y, delta_x, delta_y)
        x = info['data']['X'][0][0]  # joints, size(14, 2, n_frames) (14 joints)

        src_img = Image.open(frames[frame_idx[0]]).convert('RGB')
        src_mask_gt = Image.open('Versa_mask'.join(frames[frame_idx[0]].split('posewarp'))).convert('L')
        bbox = box[frame_idx[0]]
        src_center = [int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2)]
        # src_radius = min(int(max(bbox[2], bbox[3]) * 1.2 / 2), src_center[0], src_img.width - src_center[0], src_center[1], src_img.height - src_center[1])
        src_d = min(src_img.width, src_img.height, int(max(bbox[2], bbox[3]) * 1.2))
        src_radius = int(src_d / 2)

        if src_center[0] - src_radius < 0:
            src_center[0] = src_radius
        if src_center[0] + src_radius > src_img.width - 1:
            src_center[0] = src_img.width - src_radius - 1
        if src_center[1] - src_radius < 0:
            src_center[1] = src_radius
        if src_center[1] + src_radius > src_img.height - 1:
            src_center[1] = src_img.height - src_radius - 1

        left = src_center[0]-src_radius
        top = src_center[1]-src_radius
        right = src_center[0]+src_radius
        bottom = src_center[1]+src_radius

        src_img = src_img.crop((left, top, right, bottom))
        src_mask_gt = src_mask_gt.crop((left, top, right, bottom))
        # print('src ltrb:', left, top, right, bottom)
        # print('src_img_size', src_img.size)
        src_joints = x[:, :, frame_idx[0]] - 1.0  # (14, 2)
        src_joints[:, 0] = (src_joints[:, 0] - left)
        src_joints[:, 1] = (src_joints[:, 1] - top)
        src_face_box = self.get_face_box(src_joints, src_img.width, src_img.height)
        src_joints[:, 0] *= (self.params['IMG_WIDTH']/(2*src_radius))
        src_joints[:, 1] *= (self.params['IMG_HEIGHT']/(2*src_radius))
        src_posemap = self.make_joint_heatmaps(self.params['IMG_HEIGHT'], self.params['IMG_WIDTH'], src_joints, self.params['sigma_joint'], self.params['posemap_downsample'])
        src_limb_masks = self.make_limb_masks(self.params['limbs'], src_joints, self.params['IMG_WIDTH'], self.params['IMG_HEIGHT'])
        src_bg_mask = np.expand_dims(1.0 - np.amax(src_limb_masks, axis=2), 2)
        # src_masks_prior = np.log(np.concatenate((src_bg_mask, src_limb_masks), axis=2) + 1e-10)
        src_masks_prior = np.log(np.concatenate((src_bg_mask, src_limb_masks), axis=2) + 1e-10)

        src_face_box = torch.tensor(src_face_box)
        src_face_box[0] *= (self.params['IMG_WIDTH']/(2*src_radius))
        src_face_box[2] *= (self.params['IMG_WIDTH']/(2*src_radius))
        src_face_box[1] *= (self.params['IMG_HEIGHT']/(2*src_radius))
        src_face_box[3] *= (self.params['IMG_HEIGHT']/(2*src_radius))
        src_face_box = src_face_box.long()

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
        
        if self.full_y:
            tgt_img = Image.open(frames[frame_idx[1]]).convert('RGB')
        else:
            tgt_img = Image.open('Versa_masked'.join(frames[frame_idx[1]].split('posewarp'))).convert('RGB')
        bbox = box[frame_idx[1]]
        tgt_center = [int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2)]
        # tgt_radius = min(int(max(bbox[2], bbox[3]) * 1.2 / 2), tgt_center[0], tgt_img.width - tgt_center[0], tgt_center[1], tgt_img.height - tgt_center[1])
        tgt_d = min(tgt_img.width, tgt_img.height, int(max(bbox[2], bbox[3]) * 1.2))
        tgt_radius = int(tgt_d / 2)

        if tgt_center[0] - tgt_radius < 0:
            tgt_center[0] = tgt_radius
        if tgt_center[0] + tgt_radius > tgt_img.width - 1:
            tgt_center[0] = tgt_img.width - tgt_radius - 1
        if tgt_center[1] - tgt_radius < 0:
            tgt_center[1] = tgt_radius
        if tgt_center[1] + tgt_radius > tgt_img.height - 1:
            tgt_center[1] = tgt_img.height - tgt_radius - 1

        left = tgt_center[0]-tgt_radius
        top = tgt_center[1]-tgt_radius
        right = tgt_center[0]+tgt_radius
        bottom = tgt_center[1]+tgt_radius

        tgt_img = tgt_img.crop((left, top, right, bottom))
        tgt_joints = x[:, :, frame_idx[1]] - 1.0
        tgt_joints[:, 0] = (tgt_joints[:, 0] - left)
        tgt_joints[:, 1] = (tgt_joints[:, 1] - top)
        tgt_face_box = self.get_face_box(tgt_joints, tgt_img.width, tgt_img.height)
        # print(tgt_face_box)
        tgt_joints[:, 0] = tgt_joints[:, 0] * (self.params['IMG_WIDTH']/(2*tgt_radius))
        tgt_joints[:, 1] = tgt_joints[:, 1] * (self.params['IMG_HEIGHT']/(2*tgt_radius))
        tgt_posemap = self.make_joint_heatmaps(self.params['IMG_HEIGHT'], self.params['IMG_WIDTH'], tgt_joints, self.params['sigma_joint'], self.params['posemap_downsample'])
        # print('tgt_face_box', tgt_face_box)
        tgt_face = tgt_img.crop(tgt_face_box)
        # h, w = tgt_face.height, tgt_face.width
        # tgt_face = transforms.Resize((int(h*(256/(2*tgt_radius))), int(w*(256/(2*tgt_radius)))))(tgt_face)
        tgt_face_box = torch.tensor(tgt_face_box)
        tgt_face_box[0] *= (self.params['IMG_WIDTH']/(2*tgt_radius))
        tgt_face_box[2] *= (self.params['IMG_WIDTH']/(2*tgt_radius))
        tgt_face_box[1] *= (self.params['IMG_HEIGHT']/(2*tgt_radius))
        tgt_face_box[3] *= (self.params['IMG_HEIGHT']/(2*tgt_radius))
        tgt_face_box = tgt_face_box.long()
        # print(tgt_face.width, tgt_face.height)

        x_trans = np.zeros((2, 3, self.params['n_limbs'] + 1))
        x_trans[:, :, 0] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        x_trans[:, :, 1:] = self.get_limb_transforms(self.params['limbs'], src_joints, tgt_joints)

        # src_img, y(tgt_img), src_pose, tgt_pose, src_mask, x_trans, src_mask_gt, tgt_face, tgt_face_box, src_face_box
        # torch.Size([3, 256, 256]), src_img
        # torch.Size([3, 256, 256]), tgt_img
        # torch.Size([14, 128, 128]), src_img的关键点的heatmap，一个通道一个点，共14个点
        # torch.Size([14, 128, 128]), tgt_img的关键点的heatmap，一个通道一个点，共14个点
        # torch.Size([11, 256, 256]), src_mask的prior信息，根据关键点计算得到。
        # torch.Size([11, 2, 3]), 由src和tgt的关键点算出来的transformation matrix
        # torch.Size([1, 256, 256]), src_mask的ground-truth，由versa的segmentation模型计算得到
        # torch.Size([3, 256, 256]), 从tgt_img中crop出来的tgt_face，resize到了256*256
        # torch.Size([4])  (left, top, right, bottom), src_face在src_img中的bounding box
        # torch.Size([4])  (left, top, right, bottom), tgt_face在tgt_img中的bounding box
        return self.transform3(src_img), self.transform3(tgt_img), torch.from_numpy(src_posemap).permute(2, 0, 1).float(), torch.from_numpy(tgt_posemap).permute(2, 0, 1).float(), torch.from_numpy(src_masks_prior).permute(2, 0, 1).float(), torch.from_numpy(x_trans).permute(2, 0, 1).float(), self.transform1(src_mask_gt), self.transform3(tgt_face), tgt_face_box, src_face_box

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

    def get_face_box(self, pose, max_w, max_h):
        center = (pose[0] + pose[1]) / 2
        d = math.sqrt((pose[0][0] - pose[1][0])**2+(pose[0][1] - pose[1][1])**2)
        r = d*0.6
        # return int(center[0]-r), int(center[1]-r), int(center[0]+r), int(center[1]+r)
        return max(center[0]-r, 0), max(center[1]-r, 0), min(center[0]+r, max_w), min(center[1]+r, max_h)
        

def get_patch_weight(pose, size=62):
    heads = pose[:, 0, :, :]
    heads = heads.unsqueeze(1)
    heads = torch.nn.functional.interpolate(heads, size=size)
    heads = heads*5 + torch.ones_like(heads)
    return heads

# if __name__ == '__main__':
#     params = get_general_params()
#     ds = mtdataset(params, mode='train')
#     for i, folder in enumerate(ds.frames):
#         print(i, folder[0])

if __name__ == '__main__':
    # demo
    from net import MModel
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torch.utils.tensorboard import SummaryWriter
    import time

    params = get_general_params()
    params['IMG_HEIGHT'] = 256
    params['IMG_WIDTH'] = 256
    params['posemap_downsample'] = 2
    # net = MModel(params, use_cuda=False)
    # net.load_state_dict(torch.load('/versa/kangliwei/motion_transfer/0424-gan/g_epoch_10.pth'), strict=True)
    # net.eval()
    ds = mtdataset(params, mode='train')
    dl = DataLoader(ds, 16, False)
    writer = SummaryWriter(log_dir='runs/dataset')
    while True:
        # for item in ds[0]:
        #     print(item.size())
        # break
        for i, (src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, src_mask_gt, tgt_face, tgt_face_box) in enumerate(dl):
            continue
        writer.add_images('src_img/%d'%i, src_img*0.5+0.5)
        writer.add_images('y/%d'%i, y*0.5+0.5)
        writer.add_images('tgt_face/%d'%i, tgt_face*0.5+0.5)
    time.sleep(2)

#     # DEBUG
#     # Image.fromarray((np.swapaxes(np.swapaxes(out[0].numpy(), 0, 2), 0, 1)*255).astype(np.uint8)).save('%d_out0.jpg'%i)
#     # Image.fromarray((np.swapaxes(np.swapaxes(out[1].numpy(), 0, 2), 0, 1)*255).astype(np.uint8)).save('%d_out1.jpg'%i)
#     # Image.fromarray((out[2].numpy().sum(axis=2)*255).astype(np.uint8)).save('%d_out2.jpg'%i)
#     # Image.fromarray((out[3].numpy().sum(axis=2)*255).astype(np.uint8)).save('%d_out3.jpg'%i)
#     # for j in range(11):
#     #     Image.fromarray((out[4][:, :, j].numpy()*255).astype(np.uint8)).save('out4_%d.jpg'%j)
#     # Image.fromarray((out[4].numpy().sum(axis=2)*255).astype(np.uint8)).save('out4.jpg')