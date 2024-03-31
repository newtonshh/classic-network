import torch
from torch import nn


class LeNet(nn.Module):
    '''LeNet'''

    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.flatten = nn.Flatten();
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        y = self.sigmoid(self.c1(x))
        y = self.s2(y)
        y = self.sigmoid(self.c3(y))
        y = self.s4(y)
        y = self.sigmoid(self.c5(y))
        y = self.flatten(y)
        y = self.f6(y)
        y = self.out(y)
        return y


if __name__ == "__main__":
    x = torch.rand(1, 1, 28, 28)
    model = LeNet()
    y = model(x)
    print(y.size())
