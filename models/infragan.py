import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from ssim import SSIM, MSSSIM
import time
from lpips.lpips import LPIPS
import models.loss as loss
import torch.nn as nn
import models.loss_sobel as loss_sobel


class InfraGAN(BaseModel):
    def name(self):
        return 'InfraGAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain  # or True,
        # self.num_Gen = opt.n_Gen

        # load/define networks
        G_input_nc = opt.input_nc + 1 if self.opt.canny else opt.input_nc
        self.netG = networks.define_G(G_input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type,
                                      self.gpu_ids, opt)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan

            ##原本是opt.input_nc+opt.output_nc
            D_input_nc = opt.input_nc + opt.output_nc if self.opt.D_input == 'cat' else opt.output_nc
            print('D_input_nc=', D_input_nc)
            self.netD = networks.define_D(D_input_nc, opt.ndf,
                                          opt.which_model_netD, opt,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                          resolution=self.opt.fineSize if opt.dataset_mode == 'FLIR' else 512)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:  # or True:
                self.load_network(self.netD, 'D', opt.which_epoch)

        self.fake_AB_pool = ImagePool(opt.pool_size)
        # define loss functions
        self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
        self.criterionL1 = torch.nn.L1Loss()

        if opt.loss_ssim == 'ssim':
            self.ssim = SSIM()
        elif opt.loss_ssim == 'msssim':
            self.ssim = MSSSIM()
        else:
            self.ssim = None

        self.lpips = LPIPS(net=opt.lpips_model)
        self.MoNCELoss = loss.MoNCELoss(opt,f_net_name=opt.MoNCE_Net)
        self.perceptual_loss = loss.StyleLoss()
        self.CCPL_loss = loss.CCPLoss(opt)
        self.sobel_loss = loss_sobel.GradLoss()
        self.MSE_loss = nn.MSELoss()

        # initialize optimizers
        if self.isTrain:
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr / 100, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        # # TODO: don't print network
        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_A_gray = input['A_gray']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
            input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)
            input_A_gray = input_A_gray.cuda(self.gpu_ids[0], non_blocking=True)
        if self.opt.input_nc == 1:
            self.input_A = input_A_gray
        else:
            self.input_A = input_A
        self.input_B = input_B

        self.input_A_gray = input_A_gray
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_KAIST_input(self, input, Resize=False):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        # input_A_Canny = input['A_Canny']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
            input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)
            # input_A_Canny = input_A_Canny.cuda(self.gpu_ids[0], non_blocking=True)


        self.input_A = input_A
        self.input_B = input_B
        # self.input_A_Canny = input_A_Canny
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        if self.opt.canny:
            self.real_A = Variable(torch.cat((self.input_A, self.input_A_Canny), 1))
        else:
            self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

        # if self.opt.loss_struc:  ##只有這條件會需要用到A_gray
        #     self.real_A_gray = Variable(self.input_A_gray)

        # if self.opt.input_nc == 3 and self.opt.loss_identity:
        #     self.identity_fake_B = self.netG(self.oneD2threeD(self.real_B))
        #
        # elif self.opt.input_nc == 1 and self.opt.loss_identity:
        #     self.identity_fake_B = self.netG(self.real_B)

    def oneD2threeD(self, x):
        if x.shape[1] != 3:
            x_3 = torch.stack([x, x, x], dim=1).squeeze()
        return x_3

    # no backprop gradients
    def test(self, inference=False):
        if not self.opt.is_test:
            self.netG.eval()
            self.netD.eval()
        else:
            self.netG.eval()
        with torch.no_grad():
            # adjust input dimension for test.py (batchSize=1)
            # if inference:
            #     self.input_A = self.input_A.unsqueeze(0)
            #     self.input_B = self.input_B.unsqueeze(0)

            # if self.opt.canny:
            #     self.real_A = Variable(torch.cat((self.input_A, self.input_A_Canny), 1))

            self.real_A = Variable(self.input_A)
            t = time.time()
            self.fake_B = self.netG(self.real_A)
            t = time.time() - t
            self.real_B = Variable(self.input_B)

            # TODO: val loss
            if not inference:
                self.cal_G_loss()
                self.g_losses = self.loss_G
                self.cal_D_loss()
                self.d_losses = self.loss_D

        return t

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        self.cal_D_loss()
        self.run_discriminator_one_step(self.loss_D)

    def backward_G(self):
        self.loss_G = {}
        # with torch.autograd.set_detect_anomaly(True):
        self.cal_G_loss()
        self.run_generator_one_step(self.loss_G)

    def cal_G_loss(self):

        if self.opt.D_input == 'cat':
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        else:
            fake_AB = self.fake_B
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G['loss_G_GAN'] = self.loss_G_GAN

        if self.opt.loss_lpips:
            self.loss_G_lpips = self.lpips(self.real_B.cpu().clone(), self.fake_B.cpu().clone()).mean().cuda()
            self.loss_G['loss_G_lpips'] = self.opt.lambda_lpips * self.loss_G_lpips

        if self.opt.loss_l1:
            self.loss_G_L1 = self.criterionL1(self.fake_B.clone(), self.real_B.clone())
            self.loss_G['loss_G_l1'] = self.opt.lambda_l1 * self.loss_G_L1

        if self.opt.loss_monce:
            self.loss_G_MoNCE = self.MoNCELoss(self.fake_B.clone(), self.real_B.clone()).mean()
            self.loss_G['loss_G_MoNCE'] = self.opt.lambda_monce * self.loss_G_MoNCE

        if self.opt.loss_identity:
            self.loss_G_identity = self.criterionL1(self.real_B, self.identity_fake_B)
            self.loss_G['loss_G_identity'] = self.opt.lambda_id * self.loss_G_identity

        if self.opt.loss_sobel:
            real_sobels = self.sobel_conv(self.real_B)
            fake_sobels = self.sobel_conv(self.fake_B)
            self.loss_G_sobel = 0
            for i in range(len(fake_sobels)):
                self.loss_G_sobel += self.criterionL1(fake_sobels[i], real_sobels[i].detach())
            # self.loss_G_sobel = self.sobel_loss(self.real_B.cpu().clone(), self.fake_B.cpu().clone()).cuda()
            self.loss_G['loss_G_sobel'] = self.opt.lambda_sobel * self.loss_G_sobel

        if self.opt.loss_ssim != None:
            self.loss_G_ssim = 1. - self.ssim(self.real_B.clone(), self.fake_B.clone())
            self.loss_G['loss_G_ssim'] = self.opt.lambda_ssim * self.loss_G_ssim

        if self.opt.loss_perceptual:
            self.loss_G_perceptual = self.perceptual_loss(self.real_A, self.oneD2threeD(self.fake_B))
            self.loss_G['perceptual'] = self.opt.lambda_perceptual * self.loss_G_perceptual

        if self.opt.loss_CCP:
            if self.opt.dataset_mode == 'KAIST':
                self.loss_G_CCP = self.CCPL_loss(self.real_B, self.fake_B)
            elif self.opt.dataset_mode == 'FLIR':
                self.loss_G_CCP = self.CCPL_loss(self.real_A, self.fake_B)
            self.loss_G['CCP'] = self.opt.lambda_CCP * self.loss_G_CCP

        if self.opt.loss_mse:
            self.loss_G_MSE = self.MSE_loss(self.real_B, self.fake_B)
            self.loss_G['MSE'] = self.opt.lambda_MSE * self.loss_G_MSE

        if self.opt.loss_tv:
            diff_i = torch.sum(torch.abs(self.fake_B[:, :, :, 1:] - self.fake_B[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(self.fake_B[:, :, 1:, :] - self.fake_B[:, :, :-1, :]))
            self.loss_G_tv = self.lambda_tv * (diff_i + diff_j)

    def cal_D_loss(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        self.loss_D = {}

        if self.opt.D_input == 'cat':
            fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        else:
            fake_AB = self.fake_B
            real_AB = self.real_B

        # pred_fake, pred_middle_fake = self.netD(fake_AB.detach())
        # self.loss_D_fake = self.criterionGAN(pred_fake, False) + self.criterionGAN(pred_middle_fake, False)
        # self.loss_D['loss_D_fake'] = self.loss_D_fake * 0.5
        # # Real
        # pred_real, pred_middle_real = self.netD(real_AB)
        # self.loss_D_real = self.criterionGAN(pred_real, True) + self.criterionGAN(pred_middle_real, True)
        # self.loss_D['loss_D_real'] = self.loss_D_real * 0.5

        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D['loss_D_fake'] = self.loss_D_fake * 0.5

        # Real
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D['loss_D_real'] = self.loss_D_real * 0.5

    def run_generator_one_step(self, data):
        self.g_losses = data
        g_loss = sum(data.values()).mean()
        g_loss.backward()

    def run_discriminator_one_step(self, data):
        self.d_losses = data
        d_loss = sum(data.values()).mean()
        d_loss.backward()

    def optimize_parameters(self, total_steps):
        self.total_steps = total_steps
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # if (self.total_steps / self.opt.batchSize) % self.num_Gen == 0:
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return {**self.g_losses, **self.d_losses}

    @staticmethod
    def get_errors(opt):
        dict_errors = {'loss_G_GAN': 0,
                       'loss_D_real': 0,
                       'loss_D_fake': 0,
                       }
        if opt.loss_l1:
            dict_errors['loss_G_l1'] = 0

        if opt.loss_monce:
            dict_errors['loss_G_MoNCE'] = 0

        if opt.loss_identity:
            dict_errors['loss_G_identity'] = 0

        if opt.loss_ssim:
            dict_errors['loss_G_ssim'] = 0

        if opt.loss_perceptual:
            dict_errors['loss_G_perceptual'] = 0

        if opt.loss_CCP:
            dict_errors['loss_G_CCP'] = 0

        if opt.loss_lpips:
            dict_errors['loss_G_lpips'] = 0

        if opt.loss_sobel:
            dict_errors['loss_G_sobel'] = 0

        # if opt.loss_MSE:
        #     dict_errors['loss_G_MSE'] = 0

        return OrderedDict(dict_errors)

        # return OrderedDict([('loss_G_GAN', 0),
        #                     ('loss_G_identity', 0),
        #                     # ('loss_G_MoNCE', 0),
        #                     ('loss_D_real', 0),
        #                     ('loss_D_fake', 0),
        #                     ('loss_G_sobel', 0),
        #                     ])

    def get_current_visuals(self,normalize=False):
        real_A = util.tensor2im(self.real_A.data)
        if normalize:
            fake_B = util.thermal_tensor2im_Normalize(self.fake_B.data)
            real_B = util.thermal_tensor2im_Normalize(self.real_B.data)
        else:
            fake_B = util.thermal_tensor2im(self.fake_B.data)
            real_B = util.thermal_tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)



    def sobel_conv(self, input):
        b, c, h, w = input.shape
        conv_op = nn.Conv2d(3, 1, 3, bias=False)
        sobel_kernel = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        sobel_kernel = sobel_kernel.expand((1, c, 3, 3))
        conv_op.weight.data = sobel_kernel
        for param in conv_op.parameters():
            param.requires_grad = False
        conv_op.to(self.opt.gpu_ids[0])
        edge_detect = conv_op(input)

        conv_hor = nn.Conv2d(3, 1, 3, bias=False)
        hor_kernel = torch.Tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])

        hor_kernel = hor_kernel.expand((1, c, 3, 3))
        conv_hor.weight.data = hor_kernel
        for param in conv_hor.parameters():
            param.requires_grad = False
        conv_hor.to(self.opt.gpu_ids[0])
        hor_detect = conv_hor(input)

        conv_ver = nn.Conv2d(3, 1, 3, bias=False)
        ver_kernel = torch.Tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
        ver_kernel = ver_kernel.expand((1, c, 3, 3))
        conv_ver.weight.data = ver_kernel
        for param in conv_ver.parameters():
            param.requires_grad = False
        conv_ver.to(self.opt.gpu_ids[0])
        ver_detect = conv_ver(input)

        return [edge_detect, hor_detect, ver_detect]
