import os
import torch.nn as nn
import torch
from PIL import Image
import numpy as np
import cv2
from data.build import *
from data.build import Gaussion_noise
import shutil


def make_thermal_dataset_kaist(mode, path=None, text_path=None, val_text_path=None):
    if path is None:
        path = 'D:/InfraGAN/InfraGAN/datasets/KAIST/KAIST-dataset/kaist-cvpr15/images/'
    if text_path is None:
        # text_path = '/cta/users/mehmet/rgbt-ped-detection/data/scripts/imageSets/train-all-04.txt'
        text_path = 'D:/InfraGAN/InfraGAN/datasets/KAIST/KAIST-dataset/kaist-cvpr15/imageSets/train-all-04.txt'

    if mode == 'test':
        text_path = val_text_path
    assert os.path.isfile(text_path), '%s is not a valid file' % text_path
    assert os.path.isdir(path), '%s is not a valid directory' % path
    images = []
    with open(text_path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.split()[0]
        line = line.split('/')
        path_rgb = os.path.join(path, line[0])
        path_rgb = os.path.join(path_rgb, line[1])
        path_ir = os.path.join(path_rgb, 'lwir')
        path_ir = os.path.join(path_ir, line[2] + '.jpg')
        path_rgb = os.path.join(path_rgb, 'visible')
        path_rgb = os.path.join(path_rgb, line[2] + '.jpg')
        assert os.path.isfile(path_rgb), '%s is not a valid file' % path_rgb
        assert os.path.isfile(path_ir), '%s is not a valid file' % path_ir
        images.append({'A': path_rgb, 'B': path_ir})
    np.random.seed(12)
    np.random.shuffle(images)
    return images


def add_new_text(dir_path, add_txt):
    txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
    for txt in txt_files:
        with open(os.path.join(dir_path, txt), 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            all_sentence = (add_txt + ', ') + content
            f.write(all_sentence)
    print('Add new text finish')


def random_choince_img(_list, num):
    return list(random.sample(_list, num))


def copy_AtoB(random_list, dest_dir=None):

    visible_dest_dir = os.path.join(dest_dir,'visible')
    os.makedirs(visible_dest_dir,exist_ok=True)

    infra_dest_dir = os.path.join(dest_dir,'infra')
    os.makedirs(infra_dest_dir,exist_ok=True)

    for file_name in random_list:

        source_file_visible, source_file_infra = file_name['A'],file_name['B']

        save_img_path = ''
        visible_filename = os.path.basename(source_file_visible)
        image_path_list = source_file_visible.split('/')[-1].split('\\')
        for name in image_path_list[:-1]:
            save_img_path += (name + '_')

        visble_dest_file = os.path.join(visible_dest_dir, save_img_path+visible_filename)
        print(source_file_infra,'->', visble_dest_file)
        shutil.copyfile(source_file_visible, visble_dest_file)

        save_img_path = ''
        infra_filename = os.path.basename(source_file_infra)
        image_path_list = source_file_infra.split('/')[-1].split('\\')
        for name in image_path_list[:-1]:
            save_img_path += (name + '_')
        infra_dest_file = os.path.join(infra_dest_dir, save_img_path+infra_filename)
        print(source_file_infra, '->', infra_dest_file)
        shutil.copyfile(source_file_infra, infra_dest_file)

    print('Files copied to target folder')

def cpoy_infra_and_txt(txt_dir,infra_dir):

    txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]
    infra_list = [f for f in os.listdir(infra_dir) if f.endswith('.jpg')]

    for idx,ori_file_name in enumerate(infra_list):

        infra_filename = os.path.basename(ori_file_name)
        new_filename = '{:05d}-0-{}'.format(idx + 1, infra_filename)

        print(ori_file_name +' -> ' + new_filename)
        os.rename(os.path.join(infra_dir, ori_file_name), os.path.join(infra_dir, new_filename))



    for txt_name in txt_files:
        txt_path = os.path.join(txt_dir,txt_name)

        new_txt_path = os.path.join(infra_dir,txt_name)
        print(txt_path + '->' + new_txt_path)
        shutil.copyfile(txt_path, new_txt_path)
    print('Files copied to target folder')


if __name__ == '__main__':
    dir_path = r'C:\Users\YZU\Desktop\KAIST_Lora\preprocess'
    train_dir_path = r'C:\Users\YZU\Desktop\KAIST_Lora\train'
    ###__1__##
    images = make_thermal_dataset_kaist(mode='train')
    random_list = random_choince_img(images, 100)
    copy_AtoB(random_list=random_list,dest_dir = train_dir_path)

    ##__2__##
    #add_new_text(dir_path=r'C:\Users\YZU\Desktop\KAIST_Lora\preprocess_rgb', add_txt='kaist_infrared')


    ##__3__##
    # cpoy_infra_and_txt(txt_dir= r'C:\Users\YZU\Desktop\KAIST_Lora\preprocess_rgb',
    #                    infra_dir = r'C:\Users\YZU\Desktop\KAIST_Lora\preprocess_infra')

