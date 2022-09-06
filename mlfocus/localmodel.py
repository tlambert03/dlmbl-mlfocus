from turtle import forward
from torch import nn
import torch
from torch.nn.modules.utils import _pair


class LocallyConnected2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_size: int | tuple[int, int],
        kernel_size: int = 3,
        stride: int = 1,
        bias=False,
    ):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(
                1,
                out_channels,
                in_channels,
                output_size[0],
                output_size[1],
                kernel_size**2,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter("bias", None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class SuperModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.local1 = LocallyConnected2d(
            in_channels=2,
            out_channels=8,
            output_size=(62, 30),
            kernel_size=7,
            stride=1,
            bias=True,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.local2 = LocallyConnected2d(
            in_channels=8,
            out_channels=16,
            output_size=(30, 14),
            kernel_size=5,
            stride=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(16)
        self.local3 = LocallyConnected2d(
            in_channels=16,
            out_channels=32,
            output_size=(28, 12),
            kernel_size=3,
            stride=1,
            bias=False,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.local1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        
        x = self.local2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # x = self.local3(x)
        # x = self.relu(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
