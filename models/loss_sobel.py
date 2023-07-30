import torch
import torch.nn as nn
import torch.nn.functional as F


def sobel_loss(input, target):
    """
    Calculates the Sobel loss between the input and target images.
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

    input = F.pad(input, (1, 1, 1, 1), mode='reflect')
    target = F.pad(target, (1, 1, 1, 1), mode='reflect')

    input_dx = F.conv2d(input.unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0))
    input_dy = F.conv2d(input.unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0))
    target_dx = F.conv2d(target.unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0))
    target_dy = F.conv2d(target.unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0))

    input_grad = torch.sqrt(torch.pow(input_dx, 2) + torch.pow(input_dy, 2))
    target_grad = torch.sqrt(torch.pow(target_dx, 2) + torch.pow(target_dy, 2))

    loss = torch.mean(torch.abs(input_grad - target_grad))

    return loss


class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        kernel_h = torch.FloatTensor(sobel_x).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(sobel_y).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):

        if x.shape[1] == 3:
            x = self.get_gray(x)
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        return x

class GradLoss(nn.Module):

    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()
    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad)

if __name__ == "__main__":

    import cv2
    import numpy as np
    
    net = GradLayer()
    infra_img_path = r'D:\InfraGAN\InfraGAN\datasets\KAIST\KAIST-dataset\kaist-cvpr15\images\set00\V000/lwir/I00000.jpg'
    img = cv2.imread(infra_img_path)
    a = img.shape # (256, 256, 3)
    
    img = (img / 255.0).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = net(img) # input img: data range [0, 1]; data type torch.float32; data shape [1, 3, 256, 256]
    b = img.shape # torch.Size([1, 1, 256, 256])
    img = (img[0, :, :, :].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    
    c = img.shape # (256, 256, 1)
    cv2.imshow('pytorch sobel', img)
    cv2.waitKey(0)
