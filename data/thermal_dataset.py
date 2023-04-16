import os.path
import torchvision.transforms as transforms
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from data.image_folder import make_thermal_dataset, is_image_file
from PIL import Image, ImageOps
import cv2
from data.build import DataAugmentation
def make_thermal_dataset_kaist(mode, path=None, text_path=None, val_text_path=None):
    if path is None:
        path = 'datasets/KAIST/KAIST-dataset/kaist-cvpr15/images/'
    if text_path is None:
        # text_path = '/cta/users/mehmet/rgbt-ped-detection/data/scripts/imageSets/train-all-04.txt'
        text_path = 'datasets/KAIST/KAIST-dataset/kaist-cvpr15/imageSets/test-all-20.txt'

    if mode == 'test':
        text_path = val_text_path
    assert os.path.isfile(text_path), '%s is not a valid file' % text_path
    assert os.path.isdir(path), '%s is not a valid directory' % path
    images = []
    with open(text_path) as f:
        lines = f.readlines()
    # D:\InfraGAN\KAIST\KAIST - dataset\kaist - cvpr15\images\set06\V000\visible
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
        images.append({'A': path_rgb, 'B': path_ir, "annotation_file": os.path.join(path,
                                                                                    "..",
                                                                                    "annotations",
                                                                                    line[0],
                                                                                    line[1],
                                                                                    line[2] + '.txt')
                       })
    np.random.seed(12)
    np.random.shuffle(images)
    return images

def make_thermal_dataset_VEDAI(path):
    images = []
    assert os.path.isdir(path), '%s is not a valid directory' % path

    for fname in sorted(os.listdir(path)):
        if is_image_file(fname) and fname.endswith("co.png"):
            path_tv = os.path.join(path, fname)
            path_ir = fname[:-6] + "ir.png"
            path_ir = os.path.join(path, path_ir)
            annotation_file = os.path.join(path, "..", "Annotations1024", fname[:-7] + ".txt")
            images.append({'A': path_tv, 'B': path_ir, "annotation_file": annotation_file})
    return images


class ThermalDataset(BaseDataset):
    def initialize(self, opt, mode='train',val_text_path=None,no_trans=None,Resize=False):
        print('ThermalDataset')
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.val_text_path = val_text_path if val_text_path else opt.val_text_path
        if not no_trans:
            self.trans = DataAugmentation(mode=mode,opt=opt)


        if opt.dataset_mode == 'VEDAI':
            self.AB_paths = make_thermal_dataset_VEDAI(os.path.join(opt.dataroot, mode))
        elif opt.dataset_mode == 'KAIST':
            print("[%s] dataset is loading..." % mode)

            self.AB_paths = make_thermal_dataset_kaist(mode, path=opt.dataroot, text_path=opt.text_path,
                                                       val_text_path=self.val_text_path)
        assert (opt.resize_or_crop == 'resize_and_crop')

        self.Night_dir_list = ['set03','set04','set05','set09','set10','set11']
        self.Day_dir_list = ['set00', 'set01', 'set02', 'set06', 'set07', 'set08']
    def __getitem__(self, index):
        # AB_path = self.AB_paths[index]
        A_path = self.AB_paths[index]['A']
        B_path = self.AB_paths[index]['B']
        ann_path = self.AB_paths[index]['annotation_file']
        rgb_img = Image.open(A_path).convert('RGB')
        rgb_img = transforms.Resize(self.opt.loadSize)(rgb_img)

        B = Image.open(B_path)
        infra_img = ImageOps.grayscale(B)
        infra_img = transforms.Resize(self.opt.loadSize)(infra_img)

        infra_img,rgb_img = self.trans(infra=infra_img,rgb=rgb_img,Day_Night=self.Day_or_Night(A_path))


        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = rgb_img[0, ...] * 0.299 + rgb_img[1, ...] * 0.587 + rgb_img[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        return {'A': rgb_img, 'B': infra_img,
                'A_paths': A_path, 'B_paths': B_path, "annotation_file": ann_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'ThermalDataset'

    def Day_or_Night(self,path):

        for dir_num in  self.Night_dir_list:
            if dir_num in path:
                return 'Night'
        return 'Day'



