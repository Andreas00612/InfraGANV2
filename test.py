import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import numpy as np
from util.visualizer import Visualizer
from util import html, util
import torch
from data.thermal_dataset import ThermalDataset, AirsimDataset
from data.flir_dataset import FlirDataset
from lpips.lpips import LPIPS
from ssim import MSSSIM, SSIM
import torchvision.transforms as transforms
import draq_pic

# import pydevd_pycharm
# pydevd_pycharm.settrace('10.201.182.31', port=2525, stdoutToServer=True, stderrToServer=True)

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.D_input = 'cat'
    mode = "test"

    from evaluate import evaluate

    # evaluate(Resize=False)

    if opt.dataset_mode == 'VEDAI':
        dataset = ThermalDataset()
        dataset.initialize(opt, mode="test")
    elif opt.dataset_mode == 'KAIST':
        dataset = ThermalDataset()
        # mode = '/cta/users/mehmet/rgbt-ped-detection/data/scripts/imageSets/test-all-20.txt'
        dataset.initialize(opt, mode=mode)
    elif opt.dataset_mode == 'FLIR':
        dataset = FlirDataset()
        dataset.initialize(opt, mode="test")
    elif opt.dataset_mode == 'AirSim':
        dataset = AirsimDataset()
        dataset.initialize(opt, test_path=r'C:\Users\YZU\Desktop\Airsim')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads))
    model = create_model(opt)
    # opt.no_html = True
    # opt.display_id = 0
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s_%s' % (opt.dataset_mode, opt.phase, opt.which_epoch))
    print('image_dir: ', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    ssim_obj = SSIM()
    lpips_obj = LPIPS(net=opt.lpips_model)
    mssim, ssim, lpips, i = 0, 0, 0, 0
    step, ssim_sum = 0, 0
    # test
    all_SSIM = []
    all_LPIPS = []
    all_data_path = []
    for i, data in enumerate(dataloader):
        all_data_path.append(data['A_paths'][0])
        if i >= len(dataset):
            break
        if opt.dataset_mode == 'FLIR':
            model.set_input(data)
        elif opt.dataset_mode == 'KAIST':
            model.set_KAIST_input(data)
        elif opt.dataset_mode == 'AirSim':
            model.set_AirSim_input(data)
            model.AirSim_test(inference=True)
            image_dir = r'C:\Users\YZU\Desktop\AirSim_IR_result'
            image_name = 'IR_' + str(model.path_name[0])
            save_path = os.path.join(image_dir, image_name)
            img = util.tensor2im(model.fake_B)
            util.save_image(np.asarray(img), save_path)
        else:
            raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)
        if opt.dataset_mode != 'AirSim':
            model.test(inference=True)
            #model.fake_B = util.normalize_image(model.fake_B )
            lpips_now = lpips_obj(model.real_B.cpu().clone(), model.fake_B.cpu().clone()).mean().item()
            lpips = (lpips * i + lpips_now) / (i + 1)
            ssim_now = ssim_obj(model.real_B.clone(), model.fake_B.clone()).item()
            ssim = (ssim * i + ssim_now) / (i + 1)
            all_SSIM.append(ssim_now)
            all_LPIPS.append(lpips_now)

            visuals = model.get_current_visuals(normalize=True)
            img_path = model.get_image_paths()[0]
            visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

            print('%04d/%d: process image... %s -> ssim:%.4f , lpips:%.4f, mean_ssim:%.5f, mean_lpips:%.5f' % (
                i, len(dataset), img_path, ssim_now, lpips_now, ssim, lpips))



    draq_pic.draq_and_analysis(all_SSIM, all_data_path, save_dir=web_dir, mode='ssim')
    draq_pic.draq_and_analysis(all_LPIPS, all_data_path, save_dir=web_dir, mode='lpips')

    with open(os.path.join(web_dir, 'result.txt'), "a") as log_file:
        log_file.write(f'ssim:{ssim:.5f},lpips:{lpips:.5f} \n')
    txt_dir = os.path.join(opt.results_dir, opt.name, '%s_%s_%s' % (opt.dataset_mode, opt.phase, opt.which_epoch))
    print("\n txt_dir : ", txt_dir)

    if opt.dataset_mode != 'AirSim':
        webpage.save()

    from evaluate import evaluate

    #evaluate(Resize=False)

    # TODO: make inferences to a video
    # import cv2
    # # img_dir = r'checkpoints\inference_results\experiment_name\test_latest'  # set image dir
    # img_dir = web_dir
    #
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fake_writer = cv2.VideoWriter(os.path.join(img_dir, 'fake_B.avi'), fourcc, 15.0, (512, 512))
    # real_writer = cv2.VideoWriter(os.path.join(img_dir, 'real_B.avi'), fourcc, 15.0, (512, 512))
    # img_dir = os.path.join(img_dir, 'images')
    # for paths in os.listdir(img_dir):
    #     file_check = paths.split('_')
    #     if file_check[-2] == 'fake':
    #         paths = os.path.join(img_dir, paths)
    #         frame = cv2.imread(paths)
    #         fake_writer.write(frame)
    #         cv2.imshow('fake_frame', frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     elif file_check[-1] == 'B.png':
    #         paths = os.path.join(img_dir, paths)
    #         frame = cv2.imread(paths)
    #         real_writer.write(frame)
    #         cv2.imshow('real_frame', frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #
    # fake_writer.release()
    # real_writer.release()
