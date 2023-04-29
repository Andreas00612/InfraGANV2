import pywt
import numpy as np
import cv2


def padding_img(img):
    # 調整圖像大小為最接近原始大小的二的次幂
    h, w = img.shape
    new_h = 2 ** int(np.ceil(np.log2(h)))
    new_w = 2 ** int(np.ceil(np.log2(w)))
    padding_h = new_h - h
    padding_w = new_w - w
    top = padding_h // 2
    bottom = padding_h - top
    left = padding_w // 2
    right = padding_w - left
    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)

    return padded_img


if __name__ == '__main__':
    rgb_img = cv2.imread(r'D:\InfraGANV2\datasets\KAIST\KAIST-dataset\kaist-cvpr15\images\set00\V000\lwir\I00000.jpg',
                         cv2.IMREAD_GRAYSCALE)
    infra_img = cv2.imread(
        r'D:\InfraGANV2\datasets\KAIST\KAIST-dataset\kaist-cvpr15\images\set00\V000\visible\I00000.jpg')
    rgb_img = padding_img(rgb_img)
    h, w = rgb_img.shape
    coeffs = pywt.swt2(rgb_img, wavelet='haar', level=3)

    LL, (HL, LH, HH) = coeffs[0], coeffs[1:]

    LL_arr = np.resize(np.asarray(LL), (h, w))
    HL_arr = np.resize(np.asarray(HL), (h, w))
    LH_arr = np.resize(np.asarray(LH), (h, w))
    HH_arr = np.resize(np.asarray(HH), (h, w))

    summed = HL_arr + LH_arr + HH_arr
    # summed = np.clip(summed, 0, 255).astype('uint8').transpose(1, 2, 0)
    #
    # HH_arr = np.clip(HH_arr, 0, 255).astype('uint8').transpose(1, 2, 0)
    # HL_arr = np.clip(HL_arr, 0, 255).astype('uint8').transpose(1, 2, 0)
    # LH_arr = np.clip(LH_arr, 0, 255).astype('uint8').transpose(1, 2, 0)

    # 保存 HL 和 HH 圖像
    cv2.imwrite('rgb_img.png', rgb_img.astype('uint8'))
    cv2.imwrite('HL_img.png', HL_arr.astype('uint8'))
    cv2.imwrite('HH_img.png', HH_arr.astype('uint8'))
    cv2.imwrite('LH_img.png', HH_arr.astype('uint8'))
    cv2.imwrite('summed.png', summed)

    # 小波轉換
    # #coeffs = pywt.wavedec(padding_img(rgb_img), 'haar', level=3)
    # coeffs = pywt.swt2(rgb_img, wavelet='haar', level=2)
    #
    # # 提取 HL、LH 和 HH 子帶
    # HL_arr = np.asarray(coeffs[0][1])
    # LH_arr = np.asarray(coeffs[1][0])
    # HH_arr = np.asarray(coeffs[1][1])
    #
    # HH_arr = np.clip(HH_arr, 0, 255).astype('uint8')
    # HL_arr = np.clip(HL_arr, 0, 255).astype('uint8')
    #
    # summed = HL_arr + LH_arr + HH_arr
    # summed = np.clip(summed, 0, 255).astype('uint8')
    # cv2.imwrite('HL.png', np.uint8(HL_arr))
    # cv2.imwrite('LH.png', np.uint8(LH_arr))
    # cv2.imwrite('HH.png', np.uint8(HH_arr))
    #
    # cv2.imwrite('summed.png', summed)
