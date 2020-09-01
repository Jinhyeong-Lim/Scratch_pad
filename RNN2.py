import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

input_str = 'apple'
label_str = 'pple!'
char_vocab = sorted(list(set(input_str+label_str)))
vocab_size = len(char_vocab)
print ('문자 집합의 크기 : {}'.format(vocab_size))

char_to_index={}
for i,c in enumerate(char_vocab):
    char_to_index[c]=i

index_to_char={}
for key, value in char_to_index.items():
    index_to_char[value] = key


x_data = [char_to_index[c] for c in input_str]
y_data = [char_to_index[c] for c in label_str]

x_data=[x_data]
y_data=[y_data]

x_one_hot = [np.eye(vocab_size)[x] for x in x_data] #np.eye 단위행렬
print(x_one_hot)

class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True) # RNN 셀 구현
        self.fc = torch.nn.Linear(hidden_size, output_size, bias=True) # 출력층 구현

    def forward(self, x): # 구현한 RNN 셀과 출력층을 연결
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x

X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)
net = Net(5, 5, 5)
print('훈련 데이터의 크기 : {}'.format(X.shape))
print('레이블의 크기 : {}'.format(Y.shape))

outputs = net(X)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), 0.01)

for i in range(100):
    optimizer.zero_grad()
    outputs = net(X)
    loss = criterion(outputs.view(-1, 5), Y.view(-1)) # view를 하는 이유는 Batch 차원 제거를 위해
    loss.backward() # 기울기 계산
    optimizer.step() # 아까 optimizer 선언 시 넣어둔 파라미터 업데이트

    # 아래 세 줄은 모델이 실제 어떻게 예측했는지를 확인하기 위한 코드.
    result = outputs.data.numpy().argmax(axis=2) # 최종 예측값인 각 time-step 별 5차원 벡터에 대해서 가장 높은 값의 인덱스를 선택
    result_str = ''.join([index_to_char[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)