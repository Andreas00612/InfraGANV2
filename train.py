import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from collections import OrderedDict
import torch
import os
from eval import Evalulate


'''
To resolve the multi-threaded problem.
if __name__ == '__main__':(for all code)
or just set num_worker=0
'''
if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    # dataset = data_loader.load_data()                                         # only dataset or dataloader (same one)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    hist_error = model.get_errors(opt)
    hist_error = dict(sorted(hist_error.items()))
    visualizer.plot_data = {'train': OrderedDict((k, []) for k in hist_error.keys()),
                            'val': OrderedDict((k, []) for k in hist_error.keys()),
                            'legend': list(hist_error.keys())}
    eval = Evalulate(opt)  ##valdation 初始化設定
    if opt.continue_train:
        print("---------network is continue train------------")
        p = os.path.join(model.save_dir, "history.pth")
        hist = torch.load(p)
        visualizer.plot_data = hist['plot_data']
        visualizer.metric_data = hist['metric']
    ssim_best = 0
    lpips_best = 1
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.data_error = [0 for _ in hist_error.keys()]
        for i, data in enumerate(data_loader.dataloader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            if opt.dataset_mode == 'FLIR':
                model.set_input(data)
            elif opt.dataset_mode == 'KAIST':
                model.set_KAIST_input(data)
            model.optimize_parameters(total_steps=total_steps)

            # if total_steps % opt.display_freq == 0:
            #     save_result = total_steps % opt.update_html_freq == 0
            #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            errors = model.get_current_errors()
            visualizer.add_errors(errors)
            if total_steps % opt.print_freq == 0:
                # errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data, total_steps,
                                                len(data_loader.dataloader) * opt.batchSize)
                # if opt.display_id > 0:
                #   visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

            iter_data_time = time.time()
        del data
        if opt.display_id > 0:
            visualizer.append_error_hist(i + 1)
            visualizer.data_error = [0 for _ in hist_error.keys()]
            train_a_path = model.get_image_paths()[0]
            visualizer.display_current_results(model.get_current_visuals(), epoch, True)

            ##這裡開始Valdataion
            ssim, lpips = eval.eval(model, visualizer, opt)

            # put behind eval (get val images)
            val_a_path = model.get_image_paths()[0]
            visualizer.display_current_results(model.get_current_visuals(), epoch, True, [train_a_path, val_a_path],
                                               val=True)

            ##tensorboard
            visualizer.plot_current_errors(epoch)
            ##tensorbard-evaluation
            visualizer.plot_current_metrics(ssim, lpips, epoch)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            # TODO: hist only save at every 5 epochs
            hist = {'plot_data': visualizer.plot_data,
                    'metric': visualizer.metric_data}
            p = os.path.join(model.save_dir, "history.pth")
            torch.save(hist, p)
            model.save(epoch)
        if ssim > ssim_best:
            model.save('ssim_best')
            print('saving the ssim_best model (epoch %d, total_steps %d)' % (epoch, total_steps))
        if lpips < lpips_best:
            model.save('lpips_best')
            print('saving the lpips_best model (epoch %d, total_steps %d)' % (epoch, total_steps))

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        lr = model.update_learning_rate()
        with open(visualizer.log_name, "a") as log_file:
            log_file.write('Learning Rate = %f\n' % lr)
            log_file.write('================End of epoch %d / %d \t Time Taken: %d sec  ================\n' %
                           (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    visualizer.train_log_writer.close()
    visualizer.val_log_writer.close()