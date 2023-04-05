from torchvision import datasets, transforms
from PIL import Image
import random
import numpy as np
import torchvision.transforms.functional as F


class DataAugmentation():
    def __init__(self, opt=None, mode='train',Resize=False):
        self.opt = opt
        self.mode = mode
        self.RandomRotation = My_Rotation((-5, 5))
        self.Gaussion_noise = Gaussion_noise(p=1)
        self.RandomHorizontalFlip = My_flip(p=0.5)
        self.BrightAug = My_BrightAug(p=0.5)

        crop_size = self.opt.fineSize if opt else 512
        self.CenterCrop = My_CenterCrop(crop_size)
        self.final_trans = My_Normalize()

    def get_transform_fn(self):
        transform_list = []
        if self.mode == 'train':
            transform_list.append(self.RandomHorizontalFlip)
            transform_list.append(self.RandomRotation)

        else:
            pass
        return transform_list

    def __call__(self, infra, rgb, Day_Night):
        if self.mode != 'train':
            return self.final_trans(infra=infra,rgb=rgb)

        self.transform = self.get_transform_fn()
        for opr in self.transform:
            infra, rgb = opr(infra, rgb)

        if not self.opt.no_bright:
            rgb = self.BrightAug(rgb)

        if Day_Night == 'Day' and not self.opt.no_add_noise:
            rgb = self.Gaussion_noise(rgb)

        return self.final_trans(infra=infra,rgb=rgb)


class power_low_transform(object):
    def __init__(self, p=0.5):
        self.prop = p

    def __call__(self, img):
        do_it = random.random() <= self.prop
        if not do_it:
            return img
        else:
            img = np.asarray(img)
            img = np.array(255 * (img / 255) ** 0.5, dtype='uint8')
            img = Image.fromarray(img)
            return img


class Gaussion_noise(object):
    def __init__(self, p=0.5, mean=0, variance=0.01):
        self.prop = p
        self.mean = mean
        self.variance = variance
        self.noise_level = random.random

    def __call__(self, img):
        noise_level = random.randint(30, 100)
        if not random.random() <= self.prop:
            return img
        else:
            img = np.asarray(img)
            std_dev = np.sqrt(self.variance)
            noise = np.random.normal(self.mean, std_dev, img.shape)
            img_with_noise = np.clip(img + noise * noise_level, 0, 255).astype(np.uint8)
            return Image.fromarray(img_with_noise)


class My_Rotation(transforms.RandomRotation):
    def __init__(self, angel):
        super().__init__(angel)

    def __call__(self, img1, img2):
        angle = self.get_params(self.degrees)
        img1 = img1.rotate(angle)
        img2 = img2.rotate(angle)
        return img1, img2


class My_flip(transforms.RandomHorizontalFlip):
    def __init__(self, p):
        super().__init__(p)
        pass

    def __call__(self, img1, img2):
        if random.random() < self.p:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)
        return img1, img2


class My_CenterCrop(transforms.CenterCrop):
    def __init__(self, size):
        super().__init__(size)
        pass

    def __call__(self, img1, img2):
        img1 = F.center_crop(img1, self.size)
        img2 = F.center_crop(img2, self.size)
        return img1, img2


class My_BrightAug():
    def __init__(self, p=0.5):
        self.prop = p

    def __call__(self,img):
        do_it = random.random() <= self.prop
        if not do_it:
            return img
        else:
            return(transforms.ColorJitter(brightness=0.3)(img))


class My_Normalize():
    def __init__(self):
        pass

    def __call__(self, infra, rgb, Resize=False):
        if Resize:
            infra  = transforms.Resize(512)(infra)
            rgb = transforms.Resize(512)(rgb)
        else:
            infra = transforms.CenterCrop(512)(infra)
            rgb = transforms.CenterCrop(512)(rgb)

        infra = transforms.ToTensor()(infra.copy()).float()
        rgb = transforms.ToTensor()(rgb.copy()).float()

        infra = transforms.Normalize(mean=[0.5], std=[0.5])(infra)
        rgb = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(rgb)

        return infra,rgb


