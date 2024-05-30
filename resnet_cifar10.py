import numpy as np
import torch
from torchvision.datasets import CIFAR10 # 导入 pytorch 内置的 CIFAR10 数据
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F     # 激励函数都在这
from torch import nn
import torch.utils.data as Data
import time
import torchvision.transforms as transforms


# print(train_dataset[0])

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.blk1 = ResidualBlock(64, 64, stride=1)
        self.blk2 = ResidualBlock(64, 128, stride=2)
        self.blk3 = ResidualBlock(128, 256, stride=2)
        self.blk4 = ResidualBlock(256, 512, stride=2)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



blk = ResidualBlock(64, 128, stride=4)
x = torch.randn(2, 64, 32, 32)
out = blk(x)
print('block:', out.shape)

x = torch.randn(2, 3, 32, 32)
model = ResNet18()
out = model(x)
print('resnet:', out.shape)


def train(net, train_data, test_data, num_epochs, optimizer, criterion):
    acc_train_list = []
    loss_train_list = []
    acc_test_list = []
    loss_test_list = []

    for epoch in range(num_epochs):
        start = time.time()
        # 开始训练
        acc_train = 0.0
        loss_train = 0.0
        acc_test = 0.0
        loss_test = 0.0
        net.train()
        for i, data in enumerate(train_data):
            x_train, y_train = data
            y_out = net(x_train)
            loss = criterion(y_out, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, y_pred_train = y_out.max(1)
            acc_train += np.sum(y_pred_train.numpy() == y_train.numpy())
            loss_train += loss.data.numpy()*x_train.shape[0]
            if i % 5 == 0:
                print('epoch[%d]batch[%d/781]'%(epoch,i),  np.mean(y_pred_train.numpy() == y_train.numpy()))
        acc_train_list.append(acc_train/len(train_dataset))
        loss_train_list.append(loss_train/len(train_dataset))

        net.eval()
        for x_test, y_test in test_data:
            y_test_out = net(x_test)
            loss2 = criterion(y_test_out, y_test)
            _, y_pred_test = y_test_out.max(1)
            acc_test += np.sum(y_pred_test.numpy() == y_test.numpy())
            loss_test += loss2.data.numpy()*x_test.shape[0]
        acc_test_list.append(acc_test / len(test_dataset))
        loss_test_list.append(loss_test / len(test_dataset))

        time_elapsed = time.time() - start
        print('epoch[%d/20]:训练集loss[%.4f], acc[%.4f].测试集loss[%.4f], acc[%.4f]' % (epoch, loss_train/len(train_dataset), acc_train/len(train_dataset),
                                                                     loss_test / len(test_dataset), acc_test / len(test_dataset)))
        print('Epoch training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


# train(net, train_data, test_data, 5, optimizer, criterion)

net = ResNet18()
for name,parameters in net.named_parameters():
    print(name,':',parameters.size())


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),  # 轉換為Tensor，並歸一化至[0, 1]
    ])

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)  # 训练数据集
    train_data = Data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_data = Data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    net = ResNet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), 1e-3)  # 使用随机梯度下降，学习率 0.1

    train(net, train_data, test_data, 5, optimizer, criterion)











