import os
import random
import numpy as np
import cv2
from models.wavelet import get_wav,get_wav_two
import torch
from PIL import Image
import matplotlib.pyplot as plt



def normalize_img(img_):
    img_ = np.asarray(img_)
    normalized_img = cv2.normalize(img_, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_img = normalized_img.astype('uint8')
    cv2.imshow('Normalized Image', normalized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img_LL_wavelet(img):
    rgb_img = torch.permute(torch.unsqueeze(torch.tensor(img), 0), (0, 3, 1, 2))
    rgb_img_tensor = torch.as_tensor(rgb_img, dtype=torch.float32)
    LL_rgb_img = LL(rgb_img_tensor)
    LL_rgb_img = torch.permute(torch.squeeze(LL_rgb_img), (1, 2, 0))

    pil_image = Image.fromarray(LL_rgb_img.mul(255).byte().numpy())
    normalize_img(pil_image)
    return pil_image

def img_HH_wavelet(img):
    rgb_img = torch.permute(torch.unsqueeze(torch.tensor(img), 0), (0, 3, 1, 2))
    rgb_img_tensor = torch.as_tensor(rgb_img, dtype=torch.float32)
    LL, LH, HL, HH = get_wav_two(in_channels=3, out_channels=3)

    LH_img = LH(rgb_img_tensor)
    HL_img = HL(rgb_img_tensor)
    HH_img = HH(rgb_img_tensor)

    H_img = LH_img + HL_img + HH_img[0]
    H_img = torch.permute(torch.squeeze(H_img), (1, 2, 0))
    #normalize_img(H_img)
    pil_image = Image.fromarray(H_img.mul(255).byte().numpy())
    return pil_image

if __name__ == '__main__':
    LL, LH, HL, HH = get_wav_two(in_channels=3, out_channels=3)
    save_foler = r'D:\InfraGANV2\check_code\wavelet_result'
    night_img_folder = r'D:\InfraGAN\InfraGAN\datasets\KAIST\KAIST-dataset\kaist-cvpr15\images\set04\V000\visible'
    for path in os.listdir(night_img_folder):
        img_path = os.path.join(night_img_folder,path)

        img_path = r'D:\InfraGAN\InfraGAN\dog.jpg'
        img = cv2.imread(img_path)
        wavelet_image = img_HH_wavelet(img)
        wavelet_image.show()
        # 绘制图像
        # cht = np.asarray(wavelet_image)
        # fig, ax = plt.subplots()
        # im = ax.imshow(wavelet_image, cmap='jet', extent=[0, 1, 0, 1])
        # ax.set_aspect('equal')
        #
        # # 添加颜色条
        # cbar = ax.figure.colorbar(im, ax=ax)
        #
        # # 显示图像
        # plt.show()


        save_path_name = 'HH_' + path
        save_path = os.path.join(save_foler,save_path_name)
        print(save_path)
        #pil_image.save(save_path)

