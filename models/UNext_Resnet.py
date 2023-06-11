import torch
from torch import nn
import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
#from UNeXt import shiftedBlock, OverlapPatchEmbed, shiftedBlock_L
from wavelet import get_wav, WaveUnpool
from util.visualize import feature_visualization


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            ConvBlock(in_channels=channels, out_channels=channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Tokenized_MLP_Block(nn.Module):
    def __init__(self, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], shift_type='shiftedBlock', shift_size=5):
        super().__init__()
        self.shift_type = shift_type
        Rnet_embed_dims = [input_channels]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.patch_embed_R1 = OverlapPatchEmbed(img_size=img_size, patch_size=3, stride=1, in_chans=input_channels,
                                                embed_dim=input_channels)

        self.blockR1 = nn.ModuleList([shiftedBlock(
            dim=Rnet_embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0], shift_type=shift_type, shift_size=shift_size)])

        self.normR1 = norm_layer(Rnet_embed_dims[0])
        if self.shift_type == 'shiftedBlock_L2':
            self.conv1d = nn.Conv1d(in_channels=input_channels, out_channels=input_channels, kernel_size=(3, 3),
                                    padding=1)

    def forward(self, x):
        B = x.shape[0]
        out, H, W = self.patch_embed_R1(x)
        for i, blk in enumerate(self.blockR1):
            out = blk(out, H, W)
        out = self.normR1(out)
        x = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        if self.shift_type == 'shiftedBlock_L2':
            x = self.conv1d(x)
        return x


class Tokenized_MLP_Block_L(nn.Module):
    def __init__(self, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], shift_size=5):
        super().__init__()
        Rnet_embed_dims = [input_channels]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.blockR1 = nn.ModuleList([shiftedBlock_L(
            dim=Rnet_embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0], shift_size=shift_size)])
        self.conv2D_1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=3, stride=1,
                                  padding=1)
        # self.normR1 = norm_layer((Rnet_embed_dims[0], 128, 128))
        self.normR2 = nn.InstanceNorm2d(Rnet_embed_dims[0])

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.conv2D_1(x)
        out = self.blockR1[0](out, H, W)
        out = self.normR2(out)
        return out


class Tokenized_ResNet(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=0, num_mlp_block=9, out_channels=3, img_size=512,
                 token_channel=256,
                 opt=None, shift_type='Tokenized_MLP_Block', shift_size=5):
        super().__init__()
        self.opt = opt
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1),
            ]
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        self.up = nn.Conv1d(in_channels=num_features * 4, out_channels=token_channel, kernel_size=(3, 3), padding=1)
        self.down = nn.Conv1d(in_channels=token_channel, out_channels=num_features * 4, kernel_size=(3, 3), padding=1)

        self.UNext_blocks = nn.Sequential(
            *[Tokenized_MLP_Block(input_channels=token_channel, img_size=img_size // 4, shift_type=shift_type,
                                  shift_size=shift_size) for _ in
              range(num_mlp_block)]
        )

        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1),
                ConvBlock(num_features * 2, num_features * 1, down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1),
            ]
        )

        self.last = nn.Conv2d(num_features * 1, out_channels, kernel_size=7, stride=1, padding=3,
                              padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)

        x = self.up(x)

        # if len(self.res_blocks):
        #     x = self.res_blocks(x)

        if len(self.UNext_blocks):
            x = self.UNext_blocks(x)

        x = self.down(x)

        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))


class WavePool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels, out_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class wavelet_Genetator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=0, num_mlp_block=9, out_channels=3, img_size=512,
                 token_channel=256,
                 opt=None, shift_type='Tokenized_MLP_Block', shift_size=5):
        super().__init__()
        self.opt = opt
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )

        self.down_block_1 = ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1)
        self.down_block_2 = ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1)
        self.up = nn.Conv1d(in_channels=num_features * 4, out_channels=token_channel, kernel_size=(3, 3), padding=1)
        self.down = nn.Conv1d(in_channels=token_channel, out_channels=num_features * 4, kernel_size=(3, 3), padding=1)

        self.UNext_blocks = nn.Sequential(
            *[Tokenized_MLP_Block(input_channels=token_channel, img_size=img_size // 4, shift_type=shift_type,
                                  shift_size=shift_size) for _ in
              range(num_mlp_block)]
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks_1 = ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1,
                                     output_padding=1)

        self.up_blocks_2 = ConvBlock(num_features * 2, num_features * 1, down=False, kernel_size=3, stride=2, padding=1,
                                     output_padding=1)

        self.last = nn.Conv2d(num_features * 1, out_channels, kernel_size=7, stride=1, padding=3,
                              padding_mode="reflect")

        self.pool1 = WavePool(in_channels=64, out_channels=128).cuda()
        self.pool2 = WavePool(in_channels=128, out_channels=256).cuda()

        self.recon_block1 = WaveUnpool(256, 128, 'sum').cuda()
        self.recon_block2 = WaveUnpool(128, 64, 'sum').cuda()

        self.Conv1 = nn.Conv1d(in_channels=num_features * 4, out_channels=num_features * 2, kernel_size=(3, 3),
                               padding=1)
        self.Conv2 = nn.Conv1d(in_channels=num_features * 2, out_channels=num_features, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        skips = {}
        x = self.initial(x)
        feature_visualization(x, module_type='wavelet', stage=1)
        LL1, LH1, HL1, HH1 = self.pool1(x)
        skips['pool1'] = [LH1, HL1, HH1]
        x = self.down_block_1(x)

        LL2, LH2, HL2, HH2 = self.pool2(x)
        skips['pool2'] = [LH2, HL2, HH2]
        x = self.down_block_2(x + LL1)

        x = self.up(x)

        if len(self.UNext_blocks):
            x = self.UNext_blocks(x)

        if len(self.res_blocks):
            x = self.res_blocks(x)
        x1 = self.down(x)

        x2 = self.up_blocks_1(x1)
        x_deconv = self.recon_block1(LL2, LH2, HL2, HH2)
        # x_deconv = self.Conv1(x_deconv)
        x3 = x2 + x_deconv

        x4 = self.up_blocks_2(x3)
        x_deconv = self.recon_block2(LL1, LH1, HL1, HH1)
        # x_deconv = self.Conv2(x_deconv)
        x5 = x4 + x_deconv

        return torch.tanh(self.last(x5))


class deep_wavelet_Genetator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=0, num_mlp_block=9, out_channels=3, img_size=512,
                 token_channel=256,
                 opt=None, shift_type='Tokenized_MLP_Block', shift_size=5):
        super().__init__()
        self.opt = opt
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )

        self.down_block_1 = ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1)
        self.down_block_2 = ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1)
        self.up = nn.Conv1d(in_channels=num_features * 4, out_channels=token_channel, kernel_size=(3, 3), padding=1)
        self.down = nn.Conv1d(in_channels=token_channel, out_channels=num_features * 4, kernel_size=(3, 3), padding=1)

        self.UNext_blocks = nn.Sequential(
            *[Tokenized_MLP_Block(input_channels=token_channel, img_size=img_size // 4, shift_type=shift_type,
                                  shift_size=shift_size) for _ in
              range(num_mlp_block)]
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks_1 = ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1,
                                     output_padding=1)

        self.up_blocks_2 = ConvBlock(num_features * 2, num_features * 1, down=False, kernel_size=3, stride=2, padding=1,
                                     output_padding=1)

        self.last = nn.Conv2d(num_features * 1, out_channels, kernel_size=7, stride=1, padding=3,
                              padding_mode="reflect")

        self.pool1 = WavePool(in_channels=64, out_channels=128).cuda()
        self.pool2 = WavePool(in_channels=128, out_channels=256).cuda()
        self.pool3 = WavePool(in_channels=256, out_channels=512).cuda()

        self.recon_block3 = WaveUnpool(512, 256, 'sum').cuda()
        self.recon_block2 = WaveUnpool(256, 128, 'sum').cuda()
        self.recon_block1 = WaveUnpool(128, 64, 'sum').cuda()


    def forward(self, x):
        skips = {}
        x = self.initial(x)

        LL1, LH1, HL1, HH1 = self.pool1(x)
        x = self.down_block_1(x)

        LL2, LH2, HL2, HH2 = self.pool2(x)
        x = self.down_block_2(x)

        LL3, LH3, HL3, HH3 = self.pool3(x)
        x = self.up(x)

        if len(self.UNext_blocks):
            for idx, m in enumerate(self.UNext_blocks):
                if idx == 6:
                    x = m(x)
                elif idx == 7:
                    x = m(x)
                elif idx == 8:
                    x = m(x)
                else:
                    x = m(x)

        x0 = self.down(x)
        x_deconv = self.recon_block3(LL3, LH3, HL3, HH3)
        x1 = x0 + x_deconv

        x2 = self.up_blocks_1(x1)
        x_deconv = self.recon_block2(LL2, LH2, HL2, HH2)
        x3 = x2 + x_deconv

        x4 = self.up_blocks_2(x3)
        x_deconv = self.recon_block1(LL1, LH1, HL1, HH1)
        x5 = x4 + x_deconv

        return torch.tanh(self.last(x5))


def test():
    img_channels = 3
    x = torch.randn((4, img_channels, 512, 512)).cuda()

    # gen = Tokenized_ResNet(img_channels=3, num_features=64, num_residuals=9, out_channels=1, img_size=512)
    gen = deep_wavelet_Genetator(3, num_features=64, num_residuals=0, num_mlp_block=9, out_channels=1, img_size=512,
                                 shift_type='shiftedBlock_L2', token_channel=256)

    gen.cuda()
    print(gen(x).shape)
    print_network(gen)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    # print(net)
    print(f'{net.__class__.__name__} -> Total number of parameters: {num_params}')


if __name__ == "__main__":
    test()
