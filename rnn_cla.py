import torch
import torch.utils.data as Data
import pandas as pd
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import torch.optim as optim

data = pd.read_csv("ibdmdata.csv", index_col=0)

train=data.iloc[:60000]
x_train=train.iloc[:,:130].values
y_train=train.iloc[:,130:].values

valid = data.iloc[60000:]
x_valid=valid.iloc[:,:130].values
y_valid=valid.iloc[:,130:].values

x_train = torch.LongTensor(x_train)
y_train = torch.Tensor(y_train)
x_valid = torch.LongTensor(x_valid)
y_valid = torch.Tensor(y_valid)

train_dataset = Data.TensorDataset(x_train, y_train)
test_dataset = Data.TensorDataset(x_valid, y_valid)
train_data = Data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_data = Data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)


class lstm_ibdm(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers,
                 bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, text):
        text = text.permute(1, 0)   # 64 130
        embedded = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout1(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        out = self.fc(hidden)
        return out


test_net = lstm_ibdm(vocab_size=6000, embedding_dim=128, hidden_dim=64, n_layers=2, bidirectional=True, dropout=0.3)
optimizer = optim.SGD(test_net.parameters(), lr = 0.1)
criterion = nn.BCELoss()

def train():
    acc_train_list = []
    loss_train_list = []
    acc_test_list = []
    loss_test_list = []
    for epoch in range(100):
        # 开始训练
        acc_train = 0.0
        loss_train = 0.0
        acc_test = 0.0
        loss_test = 0.0
        test_net.train()
        for i,(x_train, y_train) in enumerate(train_data):
            y_out = test_net(x_train)
            loss = criterion(y_out, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_train = (y_out > 0.5) * 1
            acc_train += np.sum(y_pred_train.numpy() == y_train.numpy())
            loss_train += loss.data.numpy()*x_train.shape[0]
            if i % 10 == 0:
                print('epoch[%d]batch[%d/930]'%(epoch,i),np.mean(y_pred_train.numpy() == y_train.numpy()), loss.data.numpy())
        acc_train_list.append(acc_train/len(train_dataset))
        loss_train_list.append(loss_train/len(train_dataset))

        test_net.eval()
        for x_test, y_test in test_data:
            y_test_out = test_net(x_test)
            loss2 = criterion(y_test_out, y_test)
            y_pred_test = (y_test_out > 0.5) * 1
            acc_test += np.sum(y_pred_test.numpy() == y_test.numpy())
            loss_test += loss2.data.numpy()*x_test.shape[0]
        acc_test_list.append(acc_test / len(test_dataset))
        loss_test_list.append(loss_test / len(test_dataset))

        print('epoch[%d/100]:训练集loss[%.4f], acc[%.4f].测试集loss[%.4f], acc[%.4f]' % (epoch, loss_train/len(train_dataset), acc_train/len(train_dataset),
                                                                     loss_test / len(test_dataset), acc_test / len(test_dataset)))


train()





