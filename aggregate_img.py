'''
    整理图片用的脚本
'''
import os
import cv2
import numpy as np
import glob

ours_root_dir = '/versa/kangliwei/motion_transfer/0428-ours-test-1'
org_root_dir = '/versa/kangliwei/motion_transfer/posewarp-cvpr2018/0428-org-test-1'
n_rows = 0
width, height = 256, 256
for root, dirs, files in os.walk(ours_root_dir):
    n_rows = len(files)/3

n_rows = int(n_rows)
print(n_rows)

fps = 6
size = (256*3, 256)
videowriter = cv2.VideoWriter('1-tennis.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

agg = np.zeros((n_rows*height, 3*width, 3)).astype(np.uint8)
for i in range(n_rows):
    ours_gen = cv2.imread(os.path.join(ours_root_dir, 'generated_%d.jpg'%(i+1)))
    org_gen = cv2.imread(os.path.join(org_root_dir, 'generated_%d.jpg'%(i+1)))
    y = cv2.imread(os.path.join(ours_root_dir, 'ground-truth_%d.jpg'%(i+1)))
    cat = np.concatenate((ours_gen, org_gen, y), axis=1)
    videowriter.write(cat)
    agg[i*height:i*height+height, :width, :] = ours_gen
    agg[i*height:i*height+height, width:2*width, :] = org_gen
    agg[i*height:i*height+height, 2*width:, :] = y
cv2.imwrite('agg.jpg', agg)
videowriter.release()