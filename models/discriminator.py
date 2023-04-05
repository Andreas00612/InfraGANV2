"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
#from models.networks.base_network import BaseNetwork
#from models.networks.normalization import get_nonspade_norm_layer

import util.util as util

class SesameMultiscaleDiscriminator(nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='sesame_n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt, input_nc = None):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt, input_nc)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt, input_nc = None):
        subarch = opt.netD_subarch
        if subarch == 'sesame_n_layer':
            netD = SesameNLayerDiscriminator(opt, input_nc)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input, input_angle=None):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            if self.opt.load_angle:
                out = D(input, input_angle)
            else:
                out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)
            if self.opt.load_angle and not self.opt.angle_embedding:
                input_angle = self.downsample(input_angle)
        # result 2筆資料 原始輸出和input downsample後輸出
        return result


# Defines the SESAME discriminator with the specified arguments.
class SesameNLayerDiscriminator(nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt, input_nc=None, nf=64, kw=4):
        super().__init__()
        self.opt = opt
        self.sigmoid = nn.Sigmoid()
        padw = int(np.ceil((kw - 1.0) / 2))
        if input_nc is None:
            input_nc = self.compute_D_input_nc(opt)

        branch = []
        sizes = (1,input_nc - 3, 3)
        original_nf = nf
        for input_nc in sizes: 
            nf = original_nf
            norm_layer = get_nonspade_norm_layer(opt, 'spectralinstance')
            sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                         nn.LeakyReLU(0.2, False)]]

            for n in range(1, opt.n_layers_D):
                nf_prev = nf
                nf = min(nf * 2, 512)
                stride = 1 if n == opt.n_layers_D - 1 else 2
                sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                                   stride=stride, padding=padw)),
                              nn.LeakyReLU(0.2, False)
                              ]]

            branch.append(sequence)


        sem_sequence = nn.ModuleList()
        for n in range(len(branch[0])):
            sem_sequence.append(nn.Sequential(*branch[1][n]))
        sem_sequence.append(self.sigmoid)
        self.sem_sequence = nn.Sequential(*sem_sequence)

        sequence = branch[2]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw),self.sigmoid]]

        # We divide the layers into groups to extract intermediate layer outputs
        self.img_sequence = nn.ModuleList()
        for n in range(len(sequence)):
            self.img_sequence.append(nn.Sequential(*sequence[n]))

        self.finalConv = nn.Conv2d(1,1, kernel_size=3)

    def compute_D_input_nc(self, opt):
        label_nc = opt.label_nc
        input_nc = label_nc + opt.output_nc
        if opt.contain_dontcare_label:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        if not opt.no_inpaint:
            input_nc += 1
        return input_nc

    def forward(self, input, angle=None):
        img, sem = input[:,-3:], input[:,:-3]

        sem_results = self.sem_sequence(sem)
        results = [img]
        # img_sequence 負責img的conv 有n層 default:n=5 64,128,256,512,1
        for submodel in self.img_sequence[:-1]:
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)
            #results最後有 3,64,128,256,512
        # intermediate_output:512img和512sem經過my_dot
        # !!!!!這可以用在acgan!!!!!
        intermediate_output = self.my_dot(intermediate_output, sem_results)
        results.append(self.img_sequence[-1](intermediate_output))

        #get_intermediate_features = not self.opt.no_ganFeat_loss
        # if get_intermediate_features:
        #     # results: img的64,128,256,512和最後輸出(b,1,h,w)共5筆資料
        #     return results[1:]
        # else:

        return results[-1]

    def my_dot(self, x, y, z=None):
        return x + x * y.sum(1).unsqueeze(1)

    def get_angle_embedding(self, image, angle):
        bs, _, h, w = image.size()
        n_angle = self.opt.num_angle
        embedding = torch.nn.Embedding(n_angle, h*w).cuda()
        angle_embedding = embedding(angle.long())
        angle_embedding = torch.cat([angle_embedding, angle_embedding], dim=0) # 為了跟fake_concat real_concat維度相同
        angle_embedding_reshape = torch.reshape(angle_embedding, (bs, 1, h, w))
        return angle_embedding_reshape


# Returns a function that creates a normalization function
# that does not condition on semantic map
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        try:
            subnorm_type
        except:
            subnorm_type = 'instance'

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE
class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out