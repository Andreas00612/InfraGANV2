import torch.nn as nn
import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import util.util as util
def open_img(rgb_img_path, infra=True):
    rgb_img = Image.open(rgb_img_path)
    if not infra:
        rgb_img = np.transpose(np.asarray(rgb_img), (2, 0, 1))
        rgb_img = np.expand_dims(np.asarray(rgb_img), 0)
    else:
        rgb_img = np.expand_dims(np.asarray(rgb_img), 0)
        rgb_img = np.expand_dims(rgb_img, 0)

    rgb_img = torch.tensor(rgb_img).float().cuda()
    result = sobel_conv(rgb_img)

    for img in result:
        img1 = util.thermal_tensor2im(img.detach())
        cv2.imshow('My Image', img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return rgb_img


def sobel_conv(input):
    b, c, h, w = input.shape
    conv_op = nn.Conv2d(3, 1, 3, bias=False)
    sobel_kernel = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    sobel_kernel = sobel_kernel.expand((1, c, 3, 3))
    conv_op.weight.data = sobel_kernel
    # for param in conv_op.parameters():
    #     param.requires_grad = False
    # conv_op.to(opt.gpu_ids[0])
    edge_detect = conv_op.cuda()(input)

    conv_hor = nn.Conv2d(3, 1, 3, bias=False)
    hor_kernel = torch.Tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])

    hor_kernel = hor_kernel.expand((1, c, 3, 3))
    conv_hor.weight.data = hor_kernel
    # for param in conv_hor.parameters():
    #     param.requires_grad = False
    conv_hor.cuda()
    hor_detect = conv_hor(input)

    conv_ver = nn.Conv2d(3, 1, 3, bias=False)
    ver_kernel = torch.Tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
    ver_kernel = ver_kernel.expand((1, c, 3, 3))
    conv_ver.weight.data = ver_kernel
    # for param in conv_ver.parameters():
    #     param.requires_grad = False
    conv_ver.cuda()
    ver_detect = conv_ver(input)

    return [edge_detect, hor_detect, ver_detect]



if __name__ == '__main__':


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    infra_img_path = r'D:\InfraGAN\InfraGAN\datasets\KAIST\KAIST-dataset\kaist-cvpr15\images\set00\V000/lwir/I00000.jpg'
    rgb_img_path = r'D:\InfraGAN\InfraGAN\datasets\KAIST\KAIST-dataset\kaist-cvpr15\images\set00\V000/visible/I00000.jpg'

    open_img(rgb_img_path, infra=False)
    open_img(infra_img_path,infra=True)


