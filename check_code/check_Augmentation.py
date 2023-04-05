import torch.nn as nn
import torch
from PIL import Image
import numpy as np
import cv2
from data.build import *
from data.build import Gaussion_noise
def diff_image(img1,img2):

    img1_gray = img1.convert('L')
    img2_gray = img2.convert('L')
    img1_array = np.array(img1_gray)
    img2_array = np.array(img2_gray)
    diff = np.abs(img1_array - img2_array)
    diff_img = Image.fromarray(diff)
    return diff_img

if __name__ == '__main__':

    infra_img_path = r'D:\InfraGAN\InfraGAN\datasets\KAIST\KAIST-dataset\kaist-cvpr15\images\set00\V000/lwir/I00000.jpg'
    rgb_img_path = r'D:\InfraGAN\InfraGAN\datasets\KAIST\KAIST-dataset\kaist-cvpr15\images\set00\V000/visible/I00000.jpg'

    infra_img = Image.open(infra_img_path)
    rgb_img = Image.open(rgb_img_path)

    rgb = transforms.CenterCrop(512)(rgb_img)
    Gaussion_noise = Gaussion_noise(p=0.7)
    trans_rgb_img = Gaussion_noise(rgb)


    diff_result = diff_image(trans_rgb_img,rgb)


    w,h = trans_rgb_img.size
    result = Image.new(trans_rgb_img.mode,(w*2,h))
    result.paste(rgb_img, box = (0,0))
    result.paste(trans_rgb_img, box = (w,0))
    result.save('Gaussion_Augment_img.png')
    pass



