from PIL import Image
import cv2
import numpy as np
rgb_img_path = r'C:\Users\YZU\Desktop\Control_net_input\I01000.jpg'
rgb_img = Image.open(rgb_img_path)
blur_rgb = Image.fromarray(cv2.medianBlur(np.asarray(rgb_img),7))

blur_rgb.save("blur_I01000_7.jpg")
