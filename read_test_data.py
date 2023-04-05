import json
import os
import glob
import numpy as np
from PIL import Image


def read_test_data():
    with open('rgb_to_thermal_vid_map.json') as f:
        test_pair = json.load(f)

    root = r'D:\InfraGAN\FLIR_ADAS_v2'
    A_data_path = root + r'\video_rgb_test\data'
    A_data = sorted(glob.glob(os.path.join(A_data_path, "*.jpg")))
    B_data_path = root + r'\video_thermal_test\data'
    B_data = [os.path.join(B_data_path, test_pair[os.path.basename(path)]) for path in A_data]


def read_train_data_to_json(root):
    A_data_path = root + '\images_rgb_train\data'
    A_data = delete_small_data(A_data_path)
    with open('../FLIR_ADAS_v2/images_rgb_train/train_data.json', 'w', newline='') as jsonfile:
        json.dump(A_data, jsonfile)
        print(jsonfile, type(jsonfile))


def delete_small_data(root):
    B_data = sorted(glob.glob(os.path.join(root, "*.jpg")))
    size_B = []
    for index, img_path in enumerate(B_data):
        print(f"{index}/{len(B_data)}")
        B_shape = np.asarray(Image.open(img_path)).shape
        if not (B_shape[0] < 512 or B_shape[1] < 512):
            size_B.append(os.path.basename(img_path))
    return size_B


if __name__ == '__main__':
    root = r'D:\InfraGAN\FLIR_ADAS_v2'
    read_train_data_to_json(root)
