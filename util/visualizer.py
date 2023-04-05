import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from skimage.transform import resize
from collections import OrderedDict
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def draw_result(epoch, y, title, path, y2=None):
    fig = plt.figure()
    plt.plot(epoch, y, 'r', label='val')
    if y2 is not None:
        plt.plot(epoch, y2, 'b', label='training')
    plt.xlabel("Epoch")
    plt.legend()
    plt.title(title)
    # save image
    plt.savefig(os.path.join(path, title + ".png"))  # should before show method
    plt.close(fig)


class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        # self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        # if self.display_id > 0:
        #     import visdom
        #     self.vis = visdom.Visdom(port=opt.display_port)
        self.img_path_list = []
        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================Training Loss (%s) ================\n' % now)
        self.metric_data = OrderedDict([('SSIM', []), ('LPIPS', []), ('MSSIM', [])])
        # tensorboard
        self.train_log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'tensorboard_logs', 'train')
        self.val_log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'tensorboard_logs', 'val')
        self.train_log_writer = SummaryWriter(self.train_log_dir)
        self.val_log_writer = SummaryWriter(self.val_log_dir)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    # TODO: html image path save
    def display_current_results(self, visuals, epoch, save_result, path=None, val=False):
        # if self.display_id > 0:  # show images in the browser
        #     ncols = self.opt.display_single_pane_ncols
        #     if ncols > 0:
        #         h, w = next(iter(visuals.values())).shape[:2]
        #         table_css = """<style>
        #                 table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
        #                 table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
        #                 </style>""" % (w, h)
        #         title = self.name
        #         label_html = ''
        #         label_html_row = ''
        #         nrows = int(np.ceil(len(visuals.items()) / ncols))
        #         images = []
        #         idx = 0
        #         for label, image_numpy in visuals.items():
        #             label_html_row += '<td>%s</td>' % label
        #             images.append(image_numpy.transpose([2, 0, 1]))
        #             idx += 1
        #             if idx % ncols == 0:
        #                 label_html += '<tr>%s</tr>' % label_html_row
        #                 label_html_row = ''
        #         white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
        #         while idx % ncols != 0:
        #             images.append(white_image)
        #             label_html_row += '<td></td>'
        #             idx += 1
        #         if label_html_row != '':
        #             label_html += '<tr>%s</tr>' % label_html_row
        #         # pane col = image row
        #         self.vis.images(images, nrow=ncols, win=self.display_id + 1,
        #                         padding=2, opts=dict(title=title + ' images'))
        #         label_html = '<table>%s</table>' % label_html
        #         self.vis.text(table_css + label_html, win=self.display_id + 2,
        #                       opts=dict(title=title + ' labels'))
        #     else:
        #         idx = 1
        #         for label, image_numpy in visuals.items():
        #             self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
        #                            win=self.display_id + idx)
        #             idx += 1webpage
        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image_numpy in visuals.items():
                if val:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_val.png' % (epoch, label))  # save images here
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_train.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            if val:
                webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
                for n in range(epoch, 0, -1):
                    webpage.add_header('epoch [%d]' % n)
                    ims = []
                    txts = []
                    links = []

                    for label, image_numpy in visuals.items():
                        img_path = 'epoch%.3d_%s_train.png' % (n, label)
                        ims.append(img_path)
                        txts.append(label + '_train')
                        links.append(img_path)
                        if n == epoch:
                            self.img_path_list.append(path[0])
                    # if val:
                    for label, image_numpy in visuals.items():
                        img_path = 'epoch%.3d_%s_val.png' % (n, label)
                        ims.append(img_path)
                        txts.append(label + '_val')
                        links.append(img_path)
                        if n == epoch:
                            self.img_path_list.append(path[1])

                    webpage.add_images(ims, txts, links, self.img_path_list[6 * (n - 1):6 * (1 + n - 1)],
                                       width=self.win_size)
                webpage.save()

    # errors: dictionary of error labels and values

    def add_errors(self, errors):
        self.data_error = [errors[k].cpu().detach().numpy() + self.data_error[i] for i, k in enumerate(sorted(errors.keys()))]

    def append_error_hist(self, total_iter, val=False):
        for i, leg in enumerate(self.plot_data['legend']):
            if not val:
                self.plot_data['train'][leg].append(self.data_error[i] / total_iter)
            else:
                self.plot_data['val'][leg].append(self.data_error[i] / total_iter)

    def plot_current_errors(self, epochs):
        for i, leg in enumerate(self.plot_data['legend']):
            y = [[k, l] for k, l in zip(self.plot_data['train'][leg], self.plot_data['val'][leg])]
            x = np.stack([np.array(range(len(y)))] * 2, 1)
            # TODO: add opt.next_step => use opt.epoch_count(epochs)

            # print('{}:{}'.format(leg,self.plot_data['train'][leg][-1]))
            self.train_log_writer.add_scalar('train/' + leg, self.plot_data['train'][leg][-1], epochs)
            self.val_log_writer.add_scalar('val/' + leg, self.plot_data['val'][leg][-1], epochs)
            draw_result(np.array(range(len(y))) + 1,
                        [k for k in self.plot_data['val'][leg]],
                        leg,
                        self.web_dir,
                        [k for k in self.plot_data['train'][leg]])

    def plot_current_metrics(self, ssim, lpips, epochs):

        # use tensorboard
        self.val_log_writer.add_scalar('evaluation/SSIM_val', ssim, epochs)
        self.val_log_writer.add_scalar('evaluation/LPIPS_val',lpips, epochs)


        self.metric_data['SSIM'].append(ssim)
        self.metric_data['LPIPS'].append(lpips)
        y = np.array(self.metric_data['SSIM'])
        epoch = range(1, len(self.metric_data['SSIM']) + 1)
        draw_result(epoch, self.metric_data['SSIM'], "SSIM", self.web_dir)
        draw_result(epoch, self.metric_data['LPIPS'], "LPIPS", self.web_dir)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t, t_data, total_steps, len_data):
        message = '(epoch: %d, iters: %d / %d , time: %.3f, data: %.3f) ' % (epoch, i, len_data, t, t_data)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)
            # TODO: step losses
            self.train_log_writer.add_scalar('Train_Step/' + k, v, total_steps)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path, aspect_ratio=1.0,normalize=False):
        image_dir = webpage.get_image_dir()
        save_img_path = ' '
        # short_path = ntpath.basename(image_path[0])
        # name = os.path.splitext(short_path)[0]

        # fix image file name for test.py
        # print('image_path', image_path)
        # TODO: al console fix to '/'
        image_path_list = image_path.split('\\')
        for name in image_path_list[1:-1]:
            save_img_path += (name + '_')

        name = save_img_path +  os.path.basename(image_path).split('.')[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, im in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            h, w, _ = im.shape
            if aspect_ratio > 1.0:
                im = resize(im, (h, int(w * aspect_ratio)), order=3)  # interp='bicubic')
            if aspect_ratio < 1.0:
                im = resize(im, (int(h / aspect_ratio), w), order=3)  # interp='bicubic')
            util.save_image(im, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, links, width=self.win_size)
