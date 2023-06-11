import os
import random
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import pywt


def normalize_img(img_):
    img_ = np.asarray(img_)
    # normalized_img = cv2.normalize(img_, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    img_ = img_.astype('uint8')

    normalized_img = np.transpose(img_, (1, 2, 0))
    # PIL_image = Image.fromarray(normalized_img)
    # PIL_image.show()
    # cv2.imshow('Normalized Image', normalized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return normalized_img


def img_LL_wavelet(img):
    LL, LH, HL, HH = get_wav_two(in_channels=3, out_channels=3, pool=False)
    rgb_img = torch.permute(torch.unsqueeze(torch.tensor(img), 0), (0, 3, 1, 2))
    rgb_img_tensor = torch.as_tensor(rgb_img, dtype=torch.float32)
    LL_rgb_img = LL(rgb_img_tensor)

    LL_rgb_img = torch.squeeze(LL_rgb_img)

    # pil_image = unloader(LL_rgb_img.cpu())
    # normalize_img(LL_rgb_img)
    return normalize_img(LL_rgb_img)


def img_H_wavelet(img):
    LL, LH, HL, HH = get_wav_two(in_channels=3, out_channels=3, pool=False)
    rgb_img = torch.permute(torch.unsqueeze(torch.tensor(img), 0), (0, 3, 1, 2))
    rgb_img_tensor = torch.tensor(rgb_img, dtype=torch.float32)

    LH_img = LH(rgb_img_tensor)
    HL_img = HL(rgb_img_tensor)
    HH_img = HH(rgb_img_tensor)

    H_img = LH_img + HL_img + HH_img
    H_img = torch.squeeze(H_img)
    # output = unloader(H_img.cpu())
    # output = normalize_img(H_img)

    return normalize_img(H_img)


def norm_pywt_(img):
    # Perform 2D DWT with Daubechies 4 wavelet
    coeffs = pywt.dwt2(img, 'db4')
    LL, (LH, HL, HH) = coeffs
    LL = np.sqrt(np.square(LL))
    LH = np.sqrt(np.square(LH))
    HL = np.sqrt(np.square(HL))
    HH = np.sqrt(np.square(HH))

    H_img = HL + HH + LH

    # Scale coefficients to the range [0, 255]
    LL = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX)
    LH = cv2.normalize(LH, None, 0, 255, cv2.NORM_MINMAX)
    HL = cv2.normalize(HL, None, 0, 255, cv2.NORM_MINMAX)
    HH = cv2.normalize(HH, None, 0, 255, cv2.NORM_MINMAX)
    H_img_norm = cv2.normalize(H_img, None, 0, 1, cv2.NORM_MINMAX)

    # Convert coefficients to uint8 data type
    LL = np.uint8(LL)
    LH = np.uint8(LH)
    HL = np.uint8(HL)
    HH = np.uint8(HH)

    plt.save()

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].imshow(LL, )
    axs[0, 0].set_title('LL')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(LH)
    axs[0, 1].set_title('LH')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(HL)
    axs[1, 0].set_title('HL')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(H_img_norm, cmap='gray', vmin=0, vmax=1)
    axs[1, 1].set_title('H_img')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def B_pywt_(img):
    # Perform 2D DWT with Daubechies 4 wavelet



    coeffs = pywt.dwt2(img, 'db4')
    LL, (LH, HL, HH) = coeffs

    # Square each element and then take square root
    LL = np.sqrt(np.square(LL))
    LH = np.sqrt(np.square(LH))
    HL = np.sqrt(np.square(HL))
    HH = np.sqrt(np.square(HH))

    # Plot and save the images
    plt.imshow(LL, cmap='gray')
    plt.axis('off')
    plt.savefig('LL_image.png', bbox_inches='tight', pad_inches=0)

    plt.imshow(LH, cmap='gray')
    plt.axis('off')
    plt.savefig('LH_image.png', bbox_inches='tight', pad_inches=0)

    plt.imshow(HL, cmap='gray')
    plt.axis('off')
    plt.savefig('HL_image.png', bbox_inches='tight', pad_inches=0)

    plt.imshow(HH, cmap='gray')
    plt.axis('off')
    plt.savefig('HH_image.png', bbox_inches='tight', pad_inches=0)

    plt.close()

    H3img = LH + HL + HH

    H2img = LH + HL
    # Invert colors of the sub-bands

    # LL = 255 - LL
    # LH = 255 - LH
    # HL = 255 - HL
    # HH = 255 - HH
    # Himg = 255 - Himg

    # Visualize the sub-bands
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))


    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('midium_blur_7')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(LL, cmap='gray')
    axs[0, 1].set_title('LL')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(HL, cmap='gray')
    axs[0, 2].set_title('HL')
    axs[0, 2].axis('off')

    axs[1, 0].imshow(LH, cmap='gray')
    axs[1, 0].set_title('LH')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(H2img, cmap='gray')
    axs[1, 1].set_title('LH + HL')
    axs[1, 1].axis('off')

    axs[1, 2].imshow(H3img, cmap='gray')
    axs[1, 2].set_title('LH + HL + HH')
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


def pywt_(img):
    coeffs = pywt.dwt2(img, 'db4')
    LL, (LH, HL, HH) = coeffs

    LL = np.sqrt(np.square(LL))
    LH = np.sqrt(np.square(LH))
    HL = np.sqrt(np.square(HL))
    HH = np.sqrt(np.square(HH))

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].imshow(LL, )
    axs[0, 0].set_title('LL')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(LH)
    axs[0, 1].set_title('LH')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(HL)
    axs[1, 0].set_title('HL')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(HH)
    axs[1, 1].set_title('HH')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    save_foler = r'D:\InfraGANV2\check_code\wavelet_result'
    night_img_folder = r'D:\InfraGANV2\datasets\KAIST\KAIST-dataset\kaist-cvpr15\images\set02\V001\visible'
    for path in os.listdir(night_img_folder):
        img_path = os.path.join(night_img_folder, path)

        # img_path = r'D:\InfraGANV2\check_code\dog.jpg'
        img_path = r'D:\InfraGANV2\check_code\I00662.jpg'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        B_pywt_(img)
        wavelet_image = img_H_wavelet(img)

        cht = np.asarray(wavelet_image)
        fig, ax = plt.subplots()
        im = ax.imshow(wavelet_image, cmap='jet', extent=[0, 1, 0, 1])
        ax.set_aspect('equal')

        cbar = ax.figure.colorbar(im, ax=ax)
        plt.show()

        save_path_name = 'HH_' + path
        save_path = os.path.join(save_foler, save_path_name)
        print(save_path)
        # pil_image.save(save_path)
