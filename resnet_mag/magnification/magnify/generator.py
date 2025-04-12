import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary

"""
    对应原始论文中的运动放大网络，VAE结构，分为encoder，decoder和manipulator
"""

def _make_layer(block, in_planes, out_planes, num_layers, kernel_size=3, stride=1):
    layers = []
    for i in range(num_layers):
        layers.append(block(in_planes, out_planes, kernel_size, stride))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_planes, output_planes, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        p = (kernel_size - 1) // 2
        self.pad1 = nn.ReflectionPad2d(p)
        self.conv1 = nn.Conv2d(in_planes, output_planes, kernel_size=kernel_size,
                               stride=stride, bias=False)
        self.pad2 = nn.ReflectionPad2d(p)
        self.conv2 = nn.Conv2d(in_planes, output_planes, kernel_size=kernel_size,
                               stride=stride, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.conv1(self.pad1(x)))
        y = self.conv2(self.pad2(y))
        return y + x


class ConvBlock(nn.Module):
    def __init__(self, in_planes, output_planes, kernel_size=7, stride=1):
        super(ConvBlock, self).__init__()
        p = 3
        self.pad1 = nn.ReflectionPad2d(p)
        self.conv1 = nn.Conv2d(in_planes, output_planes, kernel_size=kernel_size,
                               stride=stride, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv1(self.pad1(x)))


class ConvBlockAfter(nn.Module):
    def __init__(self, in_planes, output_planes, kernel_size=3, stride=1):
        super(ConvBlockAfter, self).__init__()
        p = 1
        self.pad1 = nn.ReflectionPad2d(p)
        self.conv1 = nn.Conv2d(in_planes, output_planes, kernel_size=kernel_size,
                               stride=stride, bias=False)

    def forward(self, x):
        return self.conv1(self.pad1(x))


class Encoder(nn.Module):
    def __init__(self, num_resblk):
        super(Encoder, self).__init__()
        # common representation
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=1, bias=False)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=False)
        self.resblks = _make_layer(ResBlock, 32, 32, num_resblk)
        self.relu = nn.ReLU(inplace=True)

        # shape representation
        self.pad1_shape = nn.ReflectionPad2d(1)
        self.conv1_shape = nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False)
        self.resblks_shape = _make_layer(ResBlock, 32, 32, 2)

    def forward(self, x):
        c = self.relu(self.conv1(self.pad1(x)))
        c = self.relu(self.conv2(self.pad2(c)))
        c = self.resblks(c)

        m = self.relu(self.conv1_shape(self.pad1_shape(c)))
        m = self.resblks_shape(m)

        return m


# class Manipulator(nn.Module):
#     def __init__(self, num_resblk):
#         super(Manipulator, self).__init__()
#         self.convblks = _make_layer(ConvBlock, 32, 32, 1, kernel_size=7, stride=1)
#         self.convblks_after = _make_layer(ConvBlockAfter, 32, 32, 1, kernel_size=3, stride=1)
#         self.resblks = _make_layer(ResBlock, 32, 32, num_resblk, kernel_size=3, stride=1)

#     def forward(self, x_a, x_b, amp):
#         diff = x_b - x_a
#         diff = self.convblks(diff)
#         diff = (amp - 1.0) * diff
#         diff = self.convblks_after(diff)
#         diff = self.resblks(diff)

#         return x_b + diff
    
class Manipulator(nn.Module):
    def __init__(self, num_resblk):
        super(Manipulator, self).__init__()
        self.convblks = _make_layer(ConvBlock, 32, 32, 1, kernel_size=7, stride=1)
        self.convblks_after = _make_layer(ConvBlockAfter, 32, 32, 1, kernel_size=3, stride=1)
        self.resblks = _make_layer(ResBlock, 32, 32, num_resblk, kernel_size=3, stride=1)

    def forward(self, x_a, x_b, amp):
        diff = x_b - x_a
        diff = self.convblks(diff)
        diff = (amp - 1.0) * diff
        diff = self.convblks_after(diff)
        diff = self.resblks(diff)

        return diff, x_b + diff    


class Decoder(nn.Module):
    def __init__(self, num_resblk):
        super(Decoder, self).__init__()

        self.resblks = _make_layer(ResBlock, 32, 32, num_resblk)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False)
        self.pad2 = nn.ReflectionPad2d(3)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=7, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, m):  # v: texture, m: shape
        c = self.resblks(m)
        c = self.upsample(c)
        c = self.relu(self.conv1(self.pad1(c)))
        c = self.conv2(self.pad2(c))

        return c


class MagNet(nn.Module):
    def __init__(self, num_resblk_enc=3, num_resblk_man=1, num_resblk_dec=9):
        super(MagNet, self).__init__()
        self.encoder = Encoder(num_resblk=num_resblk_enc)
        self.manipulator = Manipulator(num_resblk=num_resblk_man)
        self.decoder = Decoder(num_resblk=num_resblk_dec)


    def forward(self, x_a, x_b, amp):
        m_a = self.encoder(x_a)
        m_b = self.encoder(x_b)

        m_diff, m_diff_enc = self.manipulator(m_a, m_b, amp)

        y_hat = self.decoder(m_diff_enc)

        return y_hat, m_a, m_b


# if __name__ == '__main__':
#     # ee = Encoder(num_resblk=3).cuda()
#     # summary(ee, torch.ones((4, 3, 384, 384)).cuda())

#     dd = Decoder(num_resblk=9).cuda()
#     summary(dd, torch.ones((4, 32, 192, 192)).cuda())
#     # model = torch.nn.DataParallel(model).cuda()
#     # print(model)

