from packaging import version
import torch
from torch import nn
from options.train_options import TrainOptions
import math
import torchvision
# from .sinkhorn import OT

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class CoNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.opt = TrainOptions().parse()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.l2_norm = Normalize(2)

    def forward(self, feat_q, feat_k, i):
        # k原始, q生成
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        # Therefore, we will include the negatives from the entire minibatch.
        # if self.opt.nce_includes_all_negatives_from_minibatch:
        #     batch_dim_for_bmm = 1
        # else:
        #batch_dim_for_bmm = self.opt.batchSize // len(self.opt.gpu_ids)
        batch_dim_for_bmm = self.opt.batchSize

        # ot_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        # ot_k = feat_k.view(batch_dim_for_bmm, -1, dim).detach()

        #### optimal transport ###
        # f = OT(ot_q, ot_k, eps=1.0, max_iter=50)
        # f = f.permute(0, 2, 1) * self.opt.ot_weight + 1e-8
        # f_max = torch.max(f, -1)[0].view(batchSize, 1)

        feat_k = feat_k.detach()
        # pos logit calculate v * v+
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))  # feat_q(256,1,dim) * feat_k(256,dim,1)
        # if i == 4:
        #     l_pos = l_pos.view(batchSize, 1) + torch.log(f_max) * 0.07
        # else:
        l_pos = l_pos.view(batchSize, 1)  # shape(256,1)

        # reshape features to batch size
        # calculate v * v-
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
        # if i == 4:
        #     l_neg_curbatch = l_neg_curbatch + torch.log(f) * 0.07

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0) # shape(1,256,256)
        l_neg = l_neg_curbatch.view(-1, npatches)  # shape(256,256)

        out = torch.cat((l_pos, l_neg), dim=1) / 0.07 # shape(256,257)

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))


        return loss
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out