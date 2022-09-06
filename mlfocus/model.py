from typing import Callable, List, Optional

from torch import nn
from torchvision import models
import torch

BasicBlock = models.resnet.BasicBlock


class ResnetModel(nn.Module):
    def __init__(self, in_channels: int = 2):
        super().__init__()
        # num_classes=1, because our output is a single scalar
        # weights=None, because
        self.model = models.resnet18(weights=None, num_classes=1)
        # 7 and 2 historic to alex net
        self.model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    def forward(self, x):
        return self.model(x)


from torch.nn.modules.utils import _pair


models.resnet.ResNet


class SmallResNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 1,
        groups: int = 1,
        depth=2,
        width_per_group: int = 64,
        first_stride=2,
        inchannels=64,
        first_kernel=7,
    ) -> None:
        super().__init__()
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = inchannels
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            in_channels,
            self.inplanes,
            kernel_size=first_kernel,
            stride=first_stride,
            padding=3,
            padding_mode="reflect",
            bias=True,
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        layer1 = self._make_layer(64, 2)
        layer2 = self._make_layer(128, 2, stride=2)
        layer3 = self._make_layer(256, 2, stride=2)
        layer4 = self._make_layer(512, 2, stride=2)

        self.layers = nn.Sequential(*(layer1, layer2, layer3, layer4)[:depth])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            [None, 64, 128, 256, 512][depth] * BasicBlock.expansion, num_classes
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        planes: int,
        blocks: int = 2,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        block = BasicBlock
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x = self.layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
