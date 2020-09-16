from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

result = Variable(torch.LongTensor(1, 5))
result = result.unsqueeze(1)

class encoder(nn.Module):
    def __init__(self,seq_len, hidden_size, dim_size):
        super(encoder, self).__init__()
        self.embeding = nn.Embedding(num_embeddings=seq_len, embedding_dim=dim_size)
        self.lstm = nn.LSTM(seq_len, hidden_size)
        self.hidden = hidden_size
        self.seq = seq_len
        # self.slef_att_quary = nn.Linear(seq_len, dim_size//2)
        # self.slef_att_key = nn.Linear(seq_len, dim_size // 2)
        # self.slef_att_value = nn.Linear(seq_len, dim_size // 2)


    def forward(self, x):
        #print(x.shape)
        output = self.embeding(x)
        #print(output.view(-1,1,5).shape)
        output ,status = self.lstm(output.view(-1,1,5))
        return output

enc = encoder(5, 5, 5)
#print(enc(result))

class decoder(nn.Module):
    def __init__(self,seq_len, hidden_size, dim_size):
        super(decoder, self).__init__()
        self.embed = nn.Embedding(num_embeddings=5, embedding_dim=dim_size)
        self.lstm = nn.LSTM(seq_len, hidden_size)
        self.hidden = hidden_size
        self.seq = seq_len
        # self.slef_att_quary = nn.Linear(seq_len, dim_size//2)
        # self.slef_att_key = nn.Linear(seq_len, dim_size // 2)
        # self.slef_att_value = nn.Linear(seq_len, dim_size // 2)
        self.out = nn.Linear(hidden_size, 5)
        self.softmax = nn.LogSoftmax(dim=0)
    def forward(self, output):
        output = self.embed(output.view(1, -1))
        output = F.relu(output)
        output ,status = self.lstm(output)

        #output = self.softmax(self.out(output))
        #print(output)
        return output

dec = decoder(5,5,5)

resul = Variable(torch.randn(1, 5)).long()
resul = result.unsqueeze(1)


print(dec(resul))