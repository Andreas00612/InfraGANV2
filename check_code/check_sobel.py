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


def canny(img_path):
    # img = cv2.imread(img_path, 0)

    img = Image.open(rgb_img_path)
    img = np.asarray(img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # grayscale conversion
    edges = cv2.Canny(img, 50, 100)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def add_pepper_noise(img, n=5):
    # 添加pepper noise
    noise_density = 0.025  # noise density設為5%
    height, width, channel = img.shape

    noise_mask = np.random.choice((0, 1), size=(height, width), p=[1 - noise_density, noise_density])
    img_with_noise = np.copy(img)
    img_with_noise[noise_mask == 1] = 255  # 0代表黑色

    pepper_noise_mask = np.random.choice((0, 1), size=(height, width), p=[1 - noise_density, noise_density])
    img_with_noise_pepper = np.copy(img_with_noise)
    img_with_noise_pepper[pepper_noise_mask == 1] = 0

    cv2.imshow('image_noise', img_with_noise_pepper)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def add_Gaussion_noise(img):
    # 添加Gaussian noise
    mean = 0
    variance = 0.01  # variance設為0.01
    std_dev = np.sqrt(variance)
    noise = np.random.normal(mean, std_dev, img.shape)
    noise_level = 100
    img_with_noise = np.clip(img + noise * 100, 0, 255).astype(np.uint8)
    # 顯示圖片
    cv2.imshow('image with noise', img_with_noise)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    infra_img_path = r'D:\InfraGAN\InfraGAN\datasets\KAIST\KAIST-dataset\kaist-cvpr15\images\set00\V000/lwir/I00000.jpg'
    rgb_img_path = r'D:\InfraGAN\InfraGAN\datasets\KAIST\KAIST-dataset\kaist-cvpr15\images\set00\V000/visible/I00000.jpg'

    infra_img = Image.open(infra_img_path)
    rgb_img = np.asarray(Image.open(rgb_img_path))
    add_Gaussion_noise(rgb_img)
    # infar_img = np.asarray(infra_img)
    # infra_img = np.expand_dims(np.asarray(infra_img), 0)
    # infra_img = np.expand_dims(infra_img, 0)
