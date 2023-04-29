import os
import random
import numpy as np
import cv2
from models.wavelet import get_wav
import torch
from PIL import Image


def img_LL_wavelet(img):
    rgb_img = torch.permute(torch.unsqueeze(torch.tensor(img), 0), (0, 3, 1, 2))
    rgb_img_tensor = torch.as_tensor(rgb_img, dtype=torch.float32)
    LL_rgb_img = LL(rgb_img_tensor)
    LL_rgb_img = torch.permute(torch.squeeze(LL_rgb_img), (1, 2, 0))
    pil_image = Image.fromarray(LL_rgb_img.mul(255).byte().numpy())
    return pil_image

def img_H_wavelet(img):
    rgb_img = torch.permute(torch.unsqueeze(torch.tensor(img), 0), (0, 3, 1, 2))
    rgb_img_tensor = torch.as_tensor(rgb_img, dtype=torch.float32)

    LH_img = LH(rgb_img_tensor)
    HL_img = HL(rgb_img_tensor)
    HH_img = HH(rgb_img_tensor)

    H_img = LH_img + HL_img + HH_img
    H_img = torch.permute(torch.squeeze(H_img), (1, 2, 0))
    pil_image = Image.fromarray(H_img.mul(255).byte().numpy())
    return pil_image

if __name__ == '__main__':
    LL, LH, HL, HH = get_wav(in_channels=3, out_channels=3)
    save_foler = r'D:\InfraGANV2\check_code\wavelet_result'
    night_img_folder = r'D:\InfraGANV2\datasets\KAIST\KAIST-dataset\kaist-cvpr15\images\set03\V000\visible'
    for path in os.listdir(night_img_folder):
        img_path = os.path.join(night_img_folder,path)
        img = cv2.imread(img_path)
        pil_image = img_LL_wavelet(img)
        save_path_name = 'LL_' + path
        save_path = os.path.join(save_foler,save_path_name)
        print(save_path)
        pil_image.save(save_path)

