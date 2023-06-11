from pytorch_wavelets import DWTForward
import torch
from options.train_options import TrainOptions

class Wavelet():
    def __init__(self, opt, wkd_level=4, wkd_basis='haar'):
        self.opt = opt
        self.xfm = DWTForward(J=wkd_level, mode='zero', wave=wkd_basis).cuda()

    def get_wavelet_loss(self, student, teacher):
        student_l, student_h = self.xfm(student)
        teacher_l, teacher_h = self.xfm(teacher)
        loss = 0.0
        if 'low' in self.opt.wavelet_mode:
            loss += torch.nn.functional.l1_loss(teacher_l, student_l)
        if 'high' in self.opt.wavelet_mode:
            for index in range(len(student_h)):
                loss += torch.nn.functional.l1_loss(teacher_h[index], student_h[index])
        return loss


class Wavelet_exp():
    def __init__(self, wkd_level=4, wkd_basis='haar'):
        self.xfm = DWTForward(J=wkd_level, mode='zero', wave=wkd_basis).cuda()

    def get_wavelet_loss(self, student, teacher):
        student_l, student_h = self.xfm(student)
        teacher_l, teacher_h = self.xfm(teacher)
        loss_low = torch.nn.functional.l1_loss(teacher_l, student_l)
        loss_high0 = torch.nn.functional.l1_loss(teacher_h[0], student_h[0])
        loss_high1 = torch.nn.functional.l1_loss(teacher_h[1], student_h[1])
        loss_high2 = torch.nn.functional.l1_loss(teacher_h[2], student_h[2])
        loss_high3 = torch.nn.functional.l1_loss(teacher_h[3], student_h[3])
        l1 = {'loss_low': float(loss_low), 'loss_high0': float(loss_high0), 'loss_high1': float(loss_high1), 'loss_high2': float(loss_high2),
              'loss_high3': float(loss_high3)}
        return l1


if __name__ == '__main__':
    opt = TrainOptions().parse()
    wkd = Wavelet(opt=opt)
    x = torch.randn(3, 3, 24, 24).cuda()
    y = torch.randn(3, 3, 24, 24).cuda()
    loss = wkd.get_wavelet_loss(x, y)
    pass
