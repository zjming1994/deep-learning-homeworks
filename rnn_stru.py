import torch
import torch.nn as nn

# single_rnn = nn.RNN(input_size=4, hidden_size=3, num_layers=2)
# x = torch.randn(5, 1, 4)   # seq=5, batch=1, input_feature=4(=input_size=)
# out, h_n = single_rnn(x)
#
# print(out.shape)
# print(out)
# print(h_n.shape)
# print(h_n)
#
# for name,parameters in single_rnn.named_parameters():
#     print(name,':',parameters.size())
#
single_lstm = nn.LSTM(input_size=4, hidden_size=3, num_layers=2, bidirectional=True)
x = torch.randn(5, 1, 4)   # seq=5, batch=1, input_feature=4(=input_size=)
out, (h_n, c_n) = single_lstm(x)

print(out.shape)
print(out)
print(h_n.shape)
print(h_n)
print(h_n[-2, :, :])
print(h_n[-1, :, :])
hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
print(hidden)
# print(c_n.shape)
# print(c_n)
#
# for name,parameters in single_lstm.named_parameters():
#     print(name,':',parameters.size())
#
#
# single_gru = nn.GRU(input_size=4, hidden_size=3, num_layers=1)
# x = torch.randn(5, 1, 4)   # seq=5, batch=1, input_feature=4(=input_size=)
# out, h_n = single_gru(x)
#
# print(out.shape)
# print(out)
# print(h_n.shape)
# print(h_n)
#
# for name,parameters in single_gru.named_parameters():
#     print(name,':',parameters.size())
#
#
#
#
#
#
