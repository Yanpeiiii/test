import copy
import torch
from torch import nn
from model.spatialtransformer import SpatialTransformer
from model.channeltransformer import ChannelTransformer


def extract_name_kwargs(obj):
    if isinstance(obj, dict):
        obj    = copy.copy(obj)
        name   = obj.pop('name')
        kwargs = obj
    else:
        name   = obj
        kwargs = {}

    return (name, kwargs)


def get_activ_layer(activ):
    name, kwargs = extract_name_kwargs(activ)

    if (name is None) or (name == 'linear'):
        return nn.Identity()

    if name == 'gelu':
        return nn.GELU(**kwargs)

    if name == 'relu':
        return nn.ReLU(**kwargs)

    if name == 'leakyrelu':
        return nn.LeakyReLU(**kwargs)

    if name == 'tanh':
        return nn.Tanh()

    if name == 'sigmoid':
        return nn.Sigmoid()

    raise ValueError("Unknown activation: '%s'" % name)


class Pre_process(nn.Module):
    def __init__(self, img_dim, in_dim):
        super(Pre_process, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(img_dim, in_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class Post_process(nn.Module):
    def __init__(self, out_dim, img_dim):
        super(Post_process, self).__init__()

        self.layer = nn.Conv2d(out_dim, img_dim, kernel_size=1)

    def forward(self, x):
        x = self.layer(x)

        return x


class UnetBasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UnetBasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.InstanceNorm2d(in_dim),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),

            nn.InstanceNorm2d(out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class UNetEncodeBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(UNetEncodeBlock, self).__init__()

        self.block = UnetBasicBlock(in_dim, out_dim)
        self.downsample = nn.Conv2d(out_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.block(x)
        x2 = self.downsample(x1)

        return x1, x2


class UNetDecodeBlock(nn.Module):
    def __init__(self,in_dim, out_dim, rezero=True):
        super(UNetDecodeBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_dim, in_dim, kernel_size=2, stride=2)
        self.block = UnetBasicBlock(in_dim * 2, out_dim)
        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, x, x1):
        x = self.re_alpha * self.upsample(x)
        x1 = torch.cat([x, x1], dim=1)
        x1 = self.block(x1)

        return x1


class DTGAN(nn.Module):
    def __init__(self, opt, in_img_dim, out_img_dim):
        super(DTGAN, self).__init__()

        self.pre_process = Pre_process(img_dim=in_img_dim, in_dim=64)
        self.enc_block_1 = UNetEncodeBlock(in_dim=64, out_dim=64)
        self.enc_block_2 = UNetEncodeBlock(in_dim=64, out_dim=128)
        self.enc_block_3 = UNetEncodeBlock(in_dim=128, out_dim=256)
        self.enc_block_4 = UNetEncodeBlock(in_dim=256, out_dim=512)

        self.dec_block_1 = UNetDecodeBlock(in_dim=512, out_dim=256)
        self.dec_block_2 = UNetDecodeBlock(in_dim=256, out_dim=128)
        self.dec_block_3 = UNetDecodeBlock(in_dim=128, out_dim=64)
        self.dec_block_4 = UNetDecodeBlock(in_dim=64, out_dim=32)
        self.post_process = Post_process(out_dim=32, img_dim=out_img_dim)

        self.channeltransformer = ChannelTransformer(opt)

        self.vit = SpatialTransformer(depth=6, dim_in=512, dim_out=512, stride_kv=1, stride_q=1, padding_q=1, padding_kv=1)

    def forward(self, x):
        # b, n, h, w = x.shape
        # if n == 1:
        #     x = x.repeat(1, 3, 1, 1)
        x = self.pre_process(x)
        x1_1, x1_2 = self.enc_block_1(x)
        x2_1, x2_2 = self.enc_block_2(x1_2)
        x3_1, x3_2 = self.enc_block_3(x2_2)
        x4_1, x4_2 = self.enc_block_4(x3_2)

        # x1_1, x2_1, x3_1, x4_1 = self.channeltransformer(x1_1, x2_1, x3_1, x4_1)
        x_r = x4_2

        x_d_1 = self.dec_block_1(x_r, x4_1)
        x_d_2 = self.dec_block_2(x_d_1, x3_1)
        x_d_3 = self.dec_block_3(x_d_2, x2_1)
        x_d_4 = self.dec_block_4(x_d_3, x1_1)

        output = self.post_process(x_d_4)

        return output


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
