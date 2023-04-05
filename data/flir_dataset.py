import os.path
import random
import torchvision.transforms as transforms
import numpy as np
from data.base_dataset import BaseDataset
import os
import glob
from PIL import Image
import json


class FlirDataset(BaseDataset):
    def initialize(self, opt, mode='train', test_map=None):
        print('ThermalDataset')
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        if test_map == None:
            self.test_map = self.opt.test_map
        else:
            self.test_map = test_map

        # TODO
        ###應該是舊版的FLIR dataset
        # if test:
        #     self.A_data = np.load(os.path.join(self.root, "grayscale_test_data.npy"))
        #     self.B_data = np.load(os.path.join(self.root, "thermal_test_data.npy"))
        # else:
        #     self.A_data = np.load(os.path.join(self.root, "grayscale_training_data.npy"))
        #     self.B_data = np.load(os.path.join(self.root, "thermal_training_data.npy"))

        if mode == 'train':
            with open(os.path.join(self.root, 'images_rgb_train/train_data.json')) as f:
                self.A_data = json.load(f)
            self.A_data_path = self.root + '/images_rgb_train/data'
            self.A_data = [os.path.join(self.A_data_path, path) for path in self.A_data]
            self.B_data_path = self.root + '/images_thermal_train/data'
            self.B_data = sorted(glob.glob(os.path.join(self.B_data_path, "*.jpg")))




        elif mode == 'val':
            self.A_data_path = self.root + '/images_rgb_val/data'
            self.A_data = self.delete_small_data(self.A_data_path)
            # self.A_data = sorted(glob.glob(os.path.join(self.A_data_path, "*.jpg")))

            self.B_data_path = self.root + '/images_thermal_val/data'
            self.B_data = sorted(glob.glob(os.path.join(self.B_data_path, "*.jpg")))



        elif mode == 'test':
            with open(self.test_map) as f:
                self.test_pair = json.load(f)
            self.A_data_path = self.root + '/video_rgb_test/data'
            self.A_data = sorted(glob.glob(os.path.join(self.A_data_path, "*.jpg")))
            self.B_data_path = self.root + '/video_thermal_test/data'
            self.B_data = [os.path.join(self.B_data_path, self.test_pair[os.path.basename(path)]) for path in
                           self.A_data]

            print(f"testing data : {len(self.B_data)}")
        else:
            raise ValueError(f"phase {mode} is not recognized.")

    def delete_small_data(self, root):
        B_data = sorted(glob.glob(os.path.join(root, "*.jpg")))
        size_B = []
        for index, img_path in enumerate(B_data):
            print(f"{index}/{len(B_data)}")
            B_shape = np.asarray(Image.open(img_path)).shape
            if not (B_shape[0] < 512 or B_shape[1] < 512):
                size_B.append(img_path)
        return size_B

    def get_transform(self, phase):
        if phase == 'train':
            pass
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([512, 512]),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        return transform

    def img2gray(self, img):
        rgb_to_grayscale = transforms.Grayscale()
        PIL_image = transforms.ToPILImage()(img)
        gray_img = rgb_to_grayscale(PIL_image)
        return transforms.ToTensor()(gray_img)

    def __getitem__(self, index):
        A_img_path = self.A_data[index]  # .transpose(2, 0, 1)
        B_img_path = self.B_data[index]  # .transpose(2, 0, 1)

        A = Image.open(A_img_path).convert("RGB")
        B = Image.open(B_img_path)

        # A = A.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A.copy()).float()
        B = transforms.ToTensor()(B.copy()).float()
        if self.opt.isTrain:
            w_total = A.size(2)
            w = int(w_total)
            h = A.size(1)
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
            # IR 大小只有 (1,512,640)
            h_b_offset = random.randint(0, max(0, B.size(1) - self.opt.fineSize - 1))
            w_b_offset = random.randint(0, max(0, B.size(2) - self.opt.fineSize - 1))

            A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            B = B[:, h_b_offset:h_b_offset + self.opt.fineSize, w_b_offset:w_b_offset + self.opt.fineSize]

            A_gray = self.img2gray(A)
            A = transforms.Normalize([0.5], [0.5])(A)
            B = transforms.Normalize([0.5], [0.5])(B)
            A_gray = transforms.Normalize([0.5], [0.5])(A_gray)
        else:
            self.transform = self.get_transform(phase='test')
            A_gray = self.img2gray(A)
            A = self.transform(A)
            B = self.transform(B)
            A_gray = self.transform(A_gray)

            # A = transforms.Resize([512,512])(A)
            # B = transforms.Resize([512,512])(B)
            # A = transforms.Normalize([0.5], [0.5])(A)
            # B = transforms.Normalize([0.5], [0.5])(B)

        return {'A': A, 'B': B, 'A_paths': A_img_path, 'B_paths': B_img_path, 'A_gray': A_gray}

    def __len__(self):
        return len(self.A_data)

    def name(self):
        return 'FLIR DATASET'
