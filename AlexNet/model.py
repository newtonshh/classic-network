import torch
from torch import nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    '''AlexNet'''

    def __init__(self):
        super(AlexNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.ReLU = nn.ReLU()
        self.s1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()

        self.f6 = nn.Linear(in_features=6 * 6 * 256, out_features=4096)
        self.f7 = nn.Linear(in_features=4096, out_features=4096)
        self.f8 = nn.Linear(in_features=4096, out_features=1000)
        self.f9 = nn.Linear(in_features=1000, out_features=10)

    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.s1(x)
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s5(x)
        x = self.flatten(x)
        x = self.f6(x)
        # Dropout：随机地将输入中50%的神经元激活设为0，即去掉了一些神经节点，防止过拟合
        # “失活的”神经元不再进行前向传播并且不参与反向传播，这个技术减少了复杂的神经元之间的相互影响
        x = F.dropout(x, p=0.5)
        x = self.f7(x)
        x = F.dropout(x, p=0.5)
        x = self.f8(x)
        x = F.dropout(x, p=0.5)
        x = self.f9(x)
        return x
