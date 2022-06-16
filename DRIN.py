'''
DRIN

Meng Z, Zhao F, Liang M, et al. Deep residual involution 
network for hyperspectral image classification[J]. Remote 
Sensing, 2021, 13(16): 3055.

'''

import torch
from torch import nn
from mmcv.cnn import ConvModule


class involution(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size,
                 stride,
                 reduct=4,
                 groups=16):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = reduct
        self.groups = groups
        self.group_channels = self.channels // self.groups
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels // reduction_ratio,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size**2 * self.groups,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out


class IVoubottleneck(nn.Module):
    def __init__(self, inplanes, planes, iksize=5, reduct=4, groups=24):
        super(IVoubottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = involution(planes, iksize, 1, reduct, groups)
        
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, 1, 1, 0, bias=False)
        
    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += identity

        return out


class DRIN(nn.Module):
    def __init__(self, num_classes, channels, reduct=4, groups=12, iksize=5, numblocks=3):
        super(DRIN, self).__init__()

        self.inplanes = 96
        self.midinplanes = 24
        self.conv1 = nn.Conv2d(channels, self.inplanes, kernel_size=1, stride=1, padding=0, bias=False)
 
        layers = []
        for _ in range(numblocks):
            layers.append(IVoubottleneck(self.inplanes, self.midinplanes, iksize=iksize, reduct=reduct, groups=groups))
        self.block = nn.Sequential(*layers)

        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.linear = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block(out)
        out = self.relu(self.bn(out))

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


if __name__ == '__main__':

    model = DRIN(num_classes=16, channels=200)
    model.eval()
    print(model)
    input = torch.randn(100, 200, 11, 11)
    y = model(input)
    print(y.size())
