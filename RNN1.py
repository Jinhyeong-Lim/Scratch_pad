import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
sentence = ("흥국생명은 김연경의 합류로 높이와 수비 모두를 보강했지만, 토털 배구를 앞세운 GS칼텍스의 패기를 막지 못했다.")
sen_word = sentence.split()
print(sen_word)

char_set = set(sentence.split()) # 중복을 제거한 문자 집합 생성
#print(char_set)
char_dic = {word: i for i, word in enumerate(char_set)} # 각 문자에 정수 인코딩
print(char_dic)
dic_size = len(char_dic)
hidden_size = dic_size
sequence_length = 5  # 임의 숫자 지정

x_data = []
y_data = []


for i in range(0, len(sen_word) - sequence_length):
    st =""
    for word in sen_word[i:i + sequence_length]:
        if(word != sen_word[i + sequence_length-1]): st = st + str(word)+" "

    x_str = st
    st=""
    for word in sen_word[i + 1: i + sequence_length + 1]:
        if(word != sen_word[i + sequence_length-1]): st = st + str(word)+" "

    y_str = st
    print(i, x_str, '->', y_str)
    #print(x_str.split())
    #print(y_str.split())
    x_data.append([char_dic[word] for word in x_str.split()])
    y_data.append([char_dic[word] for word in y_str.split()])

print(x_data[0])
print(y_data[0])

x_one_hot = [np.eye(len(char_dic))[x] for x in x_data] # x 데이터는 원-핫 인코딩
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)
#print(x_one_hot)

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers): # 현재 hidden_size는 dic_size와 같음.
        super(Net, self).__init__()
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x
net = Net(dic_size, hidden_size, 2) # 이번에는 층을 두 개 쌓습니다.

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), 0.0001)
outputs = net(X)
#print(outputs)

for i in range(20000):
    optimizer.zero_grad()
    outputs = net(X)
   # print(outputs.shape)# (77, 3, 58) 크기를 가진 텐서를 매 에포크마다 모델의 입력으로 사용
    loss = criterion(outputs.view(-1, dic_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    # results의 텐서 크기는 (170, 10)
    results = outputs.argmax(dim=2)
    predict_str = ""

    for j, result in enumerate(results):
        #print(j, list(result))
        if j == 0:  # 처음에는 예측 결과를 전부 가져오지만
            predict_str += ''.join([list(char_set)[t] for t in result])
        else:  # 그 다음에는 마지막 글자만 반복 추가
            predict_str += list(char_set)[result[-1]]
    print(i, "loss: ", loss.item(),"\n","prediction str: 흥국생명은 {}".format(predict_str))

