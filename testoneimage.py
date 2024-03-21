from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile

from dataset import ImageDataset
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere

def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--lfw', default='../../dataset/face/lfw/lfw.zip', type=str)
parser.add_argument('--model','-m', default='./model/sphere20a_20171020.pth', type=str)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
predicts=[]
net = getattr(net_sphere,args.net)()
net.load_state_dict(torch.load(args.model))
net.to(device)
net.eval()
net.feature = True

def predict_single_image(image_path, model, landmark):
    """
    对单张图像进行预测的函数。
    :param image_path: 单张图像的路径。
    :param model: 加载了预训练权重的模型。
    :param landmark: 单张图像的landmark信息。
    :return: 预测结果。
    """
    # 读取并处理图像
    src_img = cv2.imread(image_path)
    aligned_img = alignment(src_img, landmark)

    # 预处理图像以适配模型输入
    img_processed = cv2.resize(aligned_img, (96, 112)).transpose(2, 0, 1).reshape(1, 3, 112, 96)
    img_processed = (img_processed - 127.5) / 128.0
    img_processed = torch.from_numpy(img_processed).float().cuda()

    # 进行预测
    with torch.no_grad():
        img_variable = Variable(img_processed)
        print(img_variable)
        output = net(img_variable)
        # 假设你需要的输出是特征或其他形式的结果
        return output.data

# 使用示例
image_path = '/home/chiyunli/dataset/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg'
landmark_example = [107, 107, 147, 111, 124, 143, 103, 157, 139, 161]  # 示例landmark，替换为实际值
output = predict_single_image(image_path, net, landmark_example)
print(output)