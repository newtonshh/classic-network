import torch
from torch import nn


class SoftMax(nn.Module):
    '''softmax'''

    def __init__(self):
        super(SoftMax, self).__init__()
        self.flatten = nn.Flatten()
        self.liner = nn.Linear(784, 10)

    def forward(self, x):
        y = self.flatten(x)
        y = self.liner(y)
        return y


if __name__ == "__main__":
    x = torch.rand(1, 1, 28, 28)
    model = SoftMax()
    y = model(x)
    print(y.size())
