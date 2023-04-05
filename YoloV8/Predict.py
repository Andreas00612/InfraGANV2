from ultralytics import YOLO
from data.thermal_dataset import make_thermal_dataset_kaist,ThermalDataset
from options.train_options import TrainOptions
import os


# Predict with the model
#results = model('datasets/KAIST/KAIST-dataset/kaist-cvpr15/images/set00/V000/lwir/I00000.jpg')  # predict on an image

# print("[%s] dataset is loading..." % mode)
# AB_paths = make_thermal_dataset_kaist('train', path=dataroot, text_path=opt.text_path,
#                                            val_text_path=self.val_text_path)


class yolov8_KAIST():
    def __init__(self,opt):
        # self.initialize(opt = opt,no_trans=True)
        self.dataroot = opt.dataroot
        self.root = 'D:\InfraGAN\InfraGAN'
        self.text_path = os.path.join(self.root,opt.text_path)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.val_text_path = os.path.join(self.root, opt.val_text_path)
        self.AB_paths = make_thermal_dataset_kaist(mode='train', path=opt.dataroot, text_path=self.text_path,
                                                   val_text_path=self.val_text_path)

    def yoloDetect(self):
        pass


if __name__ == '__main__':
    model = YOLO('yolov8n.pt')  # load an official model
    opt = TrainOptions().parse()
    KAIST_dataset = yolov8_KAIST(opt)
    KAIST_dataset.yoloDetect()



