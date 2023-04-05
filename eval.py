from ssim import MSSSIM, SSIM
from data.thermal_dataset import ThermalDataset
from data.flir_dataset import FlirDataset
import torch.utils.data

from options.test_options import TestOptions
from tqdm import tqdm
from lpips.lpips import LPIPS
from util.visualizer import Visualizer
from models.models import create_model
from collections import OrderedDict


class Evalulate:
    def __init__(self, opt):
        mode = 'test'
        if opt.dataset_mode == 'VEDAI':
            dataset = ThermalDataset()
            dataset.initialize(opt, mode="test")
        elif opt.dataset_mode == 'KAIST':
            dataset = ThermalDataset()
            # mode = '/cta/users/mehmet/rgbt-ped-detection/data/scripts/imageSets/test-all-20.txt'
            dataset.initialize(opt, mode=mode)
        elif opt.dataset_mode == 'FLIR':
            dataset = FlirDataset()
            dataset.initialize(opt, mode=mode, test_map=opt.test_map)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,  # opt.batchSize , original setting is 1
            # shuffle=False,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            drop_last=True)
        # TODO: batchSize, shuffle

        dataset_size = len(dataset)
        print('#validation images = %d' % dataset_size)
        print()
        # TODO: No flip
        # without augmentation for KAIST

    def eval(self, model, visualizer, opt,save_image=False):
        # model = create_model(opt)
        # opt.no_html = True
        # opt.display_id = 0
        # create website
        # test
        # mssim_obj = MSSSIM(channel=1)
        ssim_obj = SSIM()
        lpips_obj = LPIPS(net=opt.lpips_model)
        mssim, ssim, lpips, i = 0, 0, 0, 0
        for i, data in enumerate(self.dataloader):
            if opt.dataset_mode == 'FLIR':
                model.set_input(data)
            elif opt.dataset_mode == 'KAIST':
                model.set_KAIST_input(data)
            model.test()
            errors = model.get_current_errors()
            visualizer.add_errors(errors)
            # mssim = (mssim*i + mssim_obj(model.real_B, model.fake_B).item())/(i+1)
            lpips = (lpips * i + lpips_obj(model.real_B.cpu().clone(), model.fake_B.cpu().clone()).mean().item()) / (i + 1)
            ssim = (ssim * i + ssim_obj(model.real_B.clone(), model.fake_B.clone()).item()) / (i + 1)

        visualizer.append_error_hist(i+1 , val=True)
        # keep it
        model.netG.train()
        model.netD.train()

        return ssim, lpips  # , mssim


# mssim /= (i + 1)
# ssim /= (i + 1)
# print("ok,mssim:{},ssim{}".format(mssim, ssim))

if __name__ == '__main__':
    opt = TestOptions().parse()
    eval = Evalulate(opt)
    visualizer = Visualizer(opt)
    model = create_model(opt)
    print(eval.dataloader.__len__())  # 907/batch_size
    eval.eval(model=model, visualizer=visualizer, save_image=True)
