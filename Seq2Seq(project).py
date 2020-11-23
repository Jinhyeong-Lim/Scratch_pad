import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from konlpy.tag import Okt
from torchtext import data
import random
from torch.autograd import Variable
import nltk
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from konlpy.tag import Okt
from torchtext import data


okt = Okt()

en_sentence = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize= word_tokenize,
                  lower=True,
                  batch_first=True,
                  fix_length=50) # 모든 text length를 fix_length에 맞추고 길이가 부족하면  padding

fr_sentence = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize= word_tokenize,
                  lower=True,
                  batch_first=True,
                  fix_length=50) # 모든 text length를 fix_length에 맞추고 길이가 부족하면  padding


from torchtext.data import TabularDataset
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

train_data, test_data = TabularDataset.splits(
        path='.', train='fra2.txt', test='fra3.txt', format='tsv',
        fields=[('source', en_sentence), ('sentence', fr_sentence)])
print(vars(train_data[2]))
print(len(train_data))
print(len(test_data))
en_sentence.build_vocab(train_data, min_freq=1,max_size=150000) # 최소 10번 이상 나온 단어 word2index
print(en_sentence.vocab.stoi)
fr_sentence.build_vocab(train_data, min_freq=1,max_size=150000) # 최소 10번 이상 나온 단어 word2index
print(fr_sentence.vocab.stoi)
from torchtext.data import Iterator
batch_size = 10
train_loader = Iterator(dataset=train_data, batch_size = batch_size, shuffle=True)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)

print(vars(train_data[0]))
print(vars(train_data[1]))
print(vars(train_data[2]))
print(vars(train_data[3]))
print(vars(train_data[4]))
print(vars(train_data[5]))
print(vars(train_data[6]))
print(vars(train_data[7]))

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # outputs are always from the top hidden layer
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
        return outputs



OUTPUT_DIM = len(en_sentence.vocab)
INPUT_DIM = len(fr_sentence.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 4
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, 0.5)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, 0.5)

net = Seq2Seq(enc, dec)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr =0.001)
losses=[]
acc=[]

for epoch in range(500):
    for batch in train_loader:
        optimizer.zero_grad()
        #print(batch.sentence.shape)
        output = net(batch.sentence, batch.source)
        #print(output.shape)
        #print(batch.sentence.shape)
        #print(hypothesis)
        output_dim = output.shape[-1]
        #print(output.shape)
        #print(batch.sentence.shape)
        output = output[1:].view(-1, output_dim)
        batch.source = batch.source[1:].view(-1)

        loss = criterion(output,batch.source)
        #print(cost)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f'Epoch {epoch + 1},training loss: {torch.tensor(losses).mean()}')

net.eval()
epoch_loss = 0
num_correct=0
from torchtext.data.metrics import bleu_score
with torch.no_grad():
    for batch in test_loader:
        # teacher_forcing_ratio = 0 (아무것도 알려주면 안 됨)
        outputs = net(batch.sentence, batch.source,0)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        output_dim = outputs.shape[-1]

        outputs = outputs[1:].view(-1, output_dim)
        batch.source = batch.source[1:].view(-1)
        print(outputs, batch.source)


