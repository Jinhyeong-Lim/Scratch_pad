import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

sentence = "양팀은 6일 울산 문수월드컵경기장에서 19라운드 경기를 치른다. 선두 울산은 이 경기를 잡으면 2위 전북 현대와의 승점 차이를 7점까지 벌릴 수 있다. 우승으로 가는 발판을 마련할 수 있는 기회. 광주는 지난 대구FC와의 맞대결에서 난타전 끝에 6대4로 승리하며 상승 분위기를 만들었다. 대어 울산을 잡는다면 단숨에 6위까지도 치고 올라갈 수 있는 상황이다.".split()
vocab = list(set(sentence))
word2index = {tkn: i for i, tkn in enumerate(vocab, 1)}  # 단어에 고유한 정수 부여
word2index['<unk>']=0
index2word = {v: k for k, v in word2index.items()}

def build_data(sentence, word2index):
    encoded = [word2index[token] for token in sentence] # 각 문자를 정수로 변환.
    input_seq, label_seq = encoded[:-1], encoded[1:] # 입력 시퀀스와 레이블 시퀀스를 분리
    input_seq = torch.LongTensor(input_seq).unsqueeze(0) # 배치 차원 추가
    label_seq = torch.LongTensor(label_seq).unsqueeze(0) # 배치 차원 추가
    return input_seq, label_seq
X, Y = build_data(sentence, word2index)
print(X.shape)
print(Y.shape)
print(len(index2word))

class Net(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, batch_first=True):
        super(Net, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, # 워드 임베딩
                                            embedding_dim=input_size)
        self.rnn_layer = nn.LSTM(input_size, hidden_size, # 입력 차원, 은닉 상태의 크기 정의
                                batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, vocab_size) # 출력은 원-핫 벡터의 크기를 가져야함. 또는 단어 집합의 크기만큼 가져야함.
        self.conv2d = nn.Conv1d(48, 48, 3, padding=1)
    def forward(self, x):
        # 1. 임베딩 층
        # 크기변화: (배치 크기, 시퀀스 길이) => (배치 크기, 시퀀스 길이, 임베딩 차원)
        output = self.embedding_layer(x)
        print(output.shape)
        output.view(48, 1, 3)
        output =  F.relu(self.conv2d(output))
        print(output.shape)
        # 2. RNN 층
        # 크기변화: (배치 크기, 시퀀스 길이, 임베딩 차원)
        # => output (배치 크기, 시퀀스 길이, 은닉층 크기), hidden (1, 배치 크기, 은닉층 크기)
        output , _hidden= self.rnn_layer(output)
        print (output.shape)
        # 3. 최종 출력층
        # 크기변화: (배치 크기, 시퀀스 길이, 은닉층 크기) => (배치 크기, 시퀀스 길이, 단어장 크기)
        output = self.linear(output)
        print(output.shape)
        # 4. view를 통해서 배치 차원 제거
        # 크기변화: (배치 크기, 시퀀스 길이, 단어장 크기) => (배치 크기*시퀀스 길이, 단어장 크기)
        return output.view(-1, output.size(2))

# 하이퍼 파라미터
vocab_size = len(word2index)  # 단어장의 크기는 임베딩 층, 최종 출력층에 사용된다. <unk> 토큰을 크기에 포함한다.
input_size = 3  # 임베딩 된 차원의 크기 및 RNN 층 입력 차원의 크기
hidden_size = 20  # RNN의 은닉층 크기

model = Net(vocab_size, input_size, hidden_size, batch_first=True)
# 손실함수 정의
loss_function = nn.CrossEntropyLoss() # 소프트맥스 함수 포함이며 실제값은 원-핫 인코딩 안 해도 됨.
# 옵티마이저 정의
optimizer = optim.Adam(params=model.parameters())

decode = lambda y: [index2word.get(x) for x in y]

for step in range(10000):
    # 경사 초기화
    optimizer.zero_grad()
    # 순방향 전파
    output = model(X)
    print(output.shape)
    # 손실값 계산
    loss = loss_function(output, Y.view(-1))
    # 역방향 전파
    loss.backward()
    # 매개변수 업데이트
    optimizer.step()
    # 기록

    print("[{:02d}/201] {:.4f} ".format(step+1, loss))
    pred = output.softmax(-1).argmax(-1).tolist()
    print(sentence[0] ," ".join(decode(pred)))