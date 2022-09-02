from torch import nn
from torchvision import models


class FocusModel2d(nn.Module):
    def __init__(self):
        super().__init__()
        # num_classes=1, because our output is a single scalar
        # weights=None, because 
        self.model = models.resnet18(weights=None, num_classes=1)
        # 7 and 2 historic to alex net
        self.model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.model(x)
