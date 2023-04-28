from PIL import Image

img = Image.open(r'D:\InfraGAN\InfraGAN\datasets\KAIST\KAIST-dataset\kaist-cvpr15\images\set02\V003\visible\I00000.jpg')

gray_img = img.convert('L')
gray_img.save(r'C:\Users\YZU\Desktop\KAIST_Lora\gray_img.png')