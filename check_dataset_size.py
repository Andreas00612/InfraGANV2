import os
import glob
from PIL import Image
import numpy as np
from torchvision import transforms

def check_FLIR_data(root):
    A_data_path = root + '\images_rgb_train\data'
    A_data = sorted(glob.glob(os.path.join(A_data_path, "*.jpg")))

    B_data_path = root + '\images_thermal_train\data'
    B_data = sorted(glob.glob(os.path.join(B_data_path, "*.jpg")))

    size_B = []
    for index,img_path in enumerate(B_data):
        size_B.append(np.asarray(Image.open(img_path)).shape)
        print(f"{index}/{len(B_data)}")

    dict = {}
    count = 0
    for key in size_B:
        count+=1
        dict[key] = dict.get(key,0) +1
    print(dict)
def delete_small_data(root):
    root = root + '\images_rgb_train\data'
    B_data = sorted(glob.glob(os.path.join(root, "*.jpg")))
    size_B = []
    for index, img_path in enumerate(B_data):
        print(f"{index}/{len(B_data)}")
        B_shape = np.asarray(Image.open(img_path)).shape
        if not (B_shape[0] < 512 or B_shape[1] < 512 ):
            size_B.append(img_path)
    return size_B

def img_to_gray(img):
    rgb_to_grayscale = transforms.Grayscale()
    #PIL_image = transforms.ToPILImage()(img).convert('RGB')
    gray_img = rgb_to_grayscale(img)
    gray_img.show()
    return transforms.ToTensor()(gray_img)



img_path = 'video-BzZspxAweF8AnKhWK-frame-000754-D3gFAecmjLtFbZeha.jpg'
img_dir = r'D:\InfraGAN\FLIR_ADAS_v2\video_rgb_test\data'
img_path = os.path.join(img_dir,img_path)
img = Image.open(img_path)
img_to_gray(img)