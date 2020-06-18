
'''
    dlib检测人脸关键点并保存。
'''
import dlib
from dataset import mtdataset
from vgg_loss import VGGPerceptualLoss
from torch.utils.tensorboard import SummaryWriter
from param import get_general_params
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
import time

def crop_face(gen, size, tgt_face_box):
    gen_face = torch.zeros(size).cuda()
    for i in range(gen.size(0)):
        # print(tgt_face_box[i][1], tgt_face_box[i][3], tgt_face_box[i][0], tgt_face_box[i][2])
        face = gen[i, :, tgt_face_box[i][1]:tgt_face_box[i][3], tgt_face_box[i][0]:tgt_face_box[i][2]]
        # print(face.size())
        face = F.interpolate(face.unsqueeze(0), size=256)
        gen_face[i] = face
    return gen_face

class face_detector():
    def __init__(self):
        self.predictor = dlib.shape_predictor('/versa/kangliwei/motion_transfer/MT/dlib/shape_predictor_68_face_landmarks.dat')
        self.detector = dlib.get_frontal_face_detector()

    def to_numpy(self, img):
        # img = img.squeeze(0)
        img = img.permute(1, 2, 0)
        return ((img.squeeze(0).cpu().numpy()*0.5+0.5)*255).astype(np.uint8)

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

    def detectobj_to_nparray(self, obj):
        ret = np.zeros((obj.num_parts, 2))
        for i in range(obj.num_parts):
            ret[i][0] = obj.part(i).x
            ret[i][1] = obj.part(i).y
        return ret

    def get_face_heatmap(self, face):
        bs = face.size(0)
        ret = torch.zeros(face.size(0), 1, face.size(2), face.size(3))
        for i in range(bs):
            f = face[i]
            img = self.to_numpy(f)
            rect = dlib.rectangle(0,0,255,255)
            dets = self.detector(img, 1)
            if len(dets) != 0:
                shape = self.predictor(img, dets[0])
            else:
                return None
                shape = self.predictor(img, rect)
            face_keypoints = self.detectobj_to_nparray(shape)
            face_heatmap = self.make_joint_heatmaps(256, 256, face_keypoints, 7/4.0, 1)
            face_heatmap = torch.from_numpy(face_heatmap)
            face_heatmap = torch.sum(face_heatmap, dim=2).unsqueeze(0)
            ret[i] = face_heatmap
            ret = torch.clamp(ret, 0, 1)
        return ret

if __name__ == '__main__':
    params = get_general_params()
    params['IMG_HEIGHT'] = 256
    params['IMG_WIDTH'] = 256
    params['posemap_downsample'] = 2
    bs = 1
    epoch = 0
    use_cuda = True
    mode = 'test'
    ds = mtdataset(params, mini=False, full_y=False, mode=mode)
    dl = DataLoader(ds, bs, shuffle=False)
    writer = SummaryWriter(log_dir='runs/debug')
    dsi = ds[0]
    dli = next(iter(dl))
    detector = face_detector()
    for i, item in enumerate(dsi):
        print('ds', i, item.size())
    frames = ds.frames

    total = 0
    good = 0
    for i in range(len(ds)):
        for j in range(len(frames[i])):
            total += 1
            src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, src_mask_gt, tgt_face, tgt_face_box, src_face_box = ds.get(i, [0, j])
            tgt_face = tgt_face.unsqueeze(0)
            ret = detector.get_face_heatmap(tgt_face)
            if ret != None:
                print(i, j, 'good')
                good += 1
                ret = ret.numpy()
                np.save('/versa/kangliwei/motion_transfer/MT/dlib/face_landmarks_'+mode+'/%d_%d.npy'%(i, j), ret)
            else:
                print(i, j, 'bad')
    print(total, good, good/total)




    # for iter, (src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, src_mask_gt, tgt_face, tgt_face_box, src_face_box) in enumerate(dl):
    #     print('epoch:', epoch, 'iter:', iter)
    #     if use_cuda:
    #         src_img, y, src_pose, tgt_pose, src_mask_prior, x_trans, tgt_face = src_img.cuda(), y.cuda(), src_pose.cuda(), tgt_pose.cuda(), src_mask_prior.cuda(), x_trans.cuda(), tgt_face.cuda()
    #     with torch.no_grad():
    #         src_face = crop_face(src_img, tgt_face.size(), src_face_box)
    #     detector = face_detector()
    #     face_heatmap = detector.get_face_heatmap(tgt_face)
    #     print(face_heatmap.size())
    #     writer.add_images('face_hp/%d'%iter, face_heatmap)
    #     writer.add_images('face/%d'%iter, tgt_face*0.5+0.5)
    #     # print('sleep')
    #     # time.sleep(10)
    #     # print('wake')
    #     # break