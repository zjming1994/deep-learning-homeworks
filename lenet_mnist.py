import numpy as np
import torch
from torchvision.datasets import mnist # 导入 pytorch 内置的 mnist 数据
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F     # 激励函数都在这
from torch import nn
import torch.utils.data as Data
import time
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),  # 轉換為Tensor，並歸一化至[0, 1]
])

train_dataset = mnist.MNIST('E:/课程资料/deep learning/data', train=True, transform=transform, download=True)
test_dataset = mnist.MNIST('E:/课程资料/deep learning/data', train=False, transform=transform, download=True)
train_data = Data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_data = Data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


net = LeNet()
print(net)

# for param in net.parameters():
#     print(param)

for name,parameters in net.named_parameters():
    print(name,':',parameters.size())


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-1) # 使用随机梯度下降，学习率 0.1

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
        for x_train, y_train in train_data:
            y_out = net(x_train)
            loss = criterion(y_out, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, y_pred_train = y_out.max(1)
            # print(y_out[0])
            # print(y_pred[0])
            acc_train += np.sum(y_pred_train.numpy() == y_train.numpy())
            loss_train += loss.data.numpy()*x_train.shape[0]
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

train(net, train_data, test_data, 5, optimizer, criterion)


