import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchtext
import nltk
import urllib.request
import pandas as pd
from konlpy.tag import Okt
from torchtext.data import Field, BucketIterator,TabularDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext import data

tokenizer = Okt()

ID = data.Field(sequential = False,
                use_vocab = False) # 실제 사용은 하지 않을 예정

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=tokenizer.morphs, # 토크나이저로는 Mecab 사용.
                  lower=True,
                  batch_first=True,
                  fix_length=20)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   is_target=True)

from torchtext.data import TabularDataset

train_data, test_data = TabularDataset.splits(
        path='.', train='ratings_train.txt', test='ratings_test.txt', format='tsv',
        fields=[('id', ID), ('text', TEXT), ('label', LABEL)], skip_header=True)

print(vars(train_data[0]))

TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
print(TEXT.vocab.stoi)

from torchtext.data import Iterator
batch_size = 50
train_loader = Iterator(dataset=train_data, batch_size = batch_size, shuffle=True)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)

class CNN(nn.Module):
    def __init__(self, batch ,vocab_size, length ,input_size): #(batch_size,출현 단어 개수, 1개 문장 tensor 길이, embedding_dim)
        super(CNN, self).__init__()
        self.input_size = input_size
        self.batch = batch
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,  # 워드 임베딩 ()
                                            embedding_dim=input_size)

        self.conv3 = nn.Conv2d(1, 1, (3, input_size),bias=True)
        self.conv4 = nn.Conv2d(1, 1, (4, input_size),bias=True)
        self.conv5 = nn.Conv2d(1, 1, (5, input_size),bias=True)
        self.Max3_pool = nn.MaxPool2d((length - 3 + 1, 1))
        self.Max4_pool = nn.MaxPool2d((length - 4 + 1, 1))
        self.Max5_pool = nn.MaxPool2d((length - 5 + 1, 1))
        self.linear1 = nn.Linear(3, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 1. 임베딩 층
        # 크기변화: (배치 크기, 시퀀스 길이) => (배치 크기, 시퀀스 길이, 임베딩 차원)
        #print(x.shape)
        output = self.embedding_layer(x)
        # print(output.shape)
        x = output.view(self.batch, 1, -1, self.input_size)  # channel(단어 개수), sequence length(임베딩 차원)

        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))

        # Pooling
        x1 = F.relu(self.Max3_pool(x1))
        x2 = F.relu(self.Max4_pool(x2))
        x3 = F.relu(self.Max5_pool(x3))

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)
        #print(x.shape)
        x = x.view(self.batch, -1)
        self.dropout
        # project the features to the labels
        x = F.sigmoid(self.linear1(x))

        return x


net = CNN(50, len(TEXT.vocab), 20, 20) # batch_size, 총단어 개수, pad_sequence문장길이,)
                                          # embeding dimension 개수

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr =0.001)
losses=[]
acc=[]
for epoch in range(20):
    for batch in train_loader:
        optimizer.zero_grad()
        predictions = net(batch.text)
        #print(hypothesis)
        loss = criterion(predictions, batch.label.float())
        #print(cost)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f'Epoch {epoch + 1},training loss: {torch.tensor(losses).mean()}')

test_losses = []  # track loss
num_correct = 0

net.eval()
# iterate over test data
for batch in test_loader:


    # get predicted outputs
    output = net(batch.text)

    # calculate loss
    test_loss = criterion(output.squeeze(), batch.label.float())
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    # compare predictions to true label
    correct_tensor = pred.eq(batch.label.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy())
    num_correct += np.sum(correct)

# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct / len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))