from model import LeNet

import torch
import torch.utils.data
from torch import nn
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os
import time
import sys

data_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='../data', train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=0)

test_dataset = datasets.MNIST(root='../data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True, num_workers=0)

device = ''
if sys.platform.startswith('win'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif sys.platform.startswith('darwin'):
    device = 'mps'
print(device)

model = LeNet().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        output = model(x)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y == pred) / output.shape[0]

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()

        n = n + 1

    train_loss = loss / n
    train_acc = current / n
    print('train_loss ' + str(train_loss))
    print('train_acc ' + str(train_acc))


def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]

            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

        evl_loss = loss / n
        evl_acc = current / n
        print('train_loss ' + str(evl_loss))
        print('train_acc ' + str(evl_acc))

        return evl_acc


epoch = 10
min_acc = 0
for t in range(epoch):
    # 记录程序开始运行的时间
    start = time.time()
    print(f'epoch {t + 1}\n------------')
    train(train_dataloader, model, loss_fn, optimizer)
    a = val(test_dataloader, model, loss_fn)
    # 记录程序结束运行的时间
    end = time.time()
    # 计算程序运行时间
    elapsed = end - start
    print(f"程序运行时间为{elapsed}毫秒")
    if a > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir(folder)
        min_acc = a
        print('save best model')
        torch.save(model.state_dict(), 'save_model/best_model.pth')

print('done')
