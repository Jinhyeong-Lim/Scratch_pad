import requests
import re
from gensim.models.word2vec import Word2Vec
import pandas as pd
import urllib.request
import numpy as np
from konlpy.tag import Okt
from konlpy.tag import Komoran
from sklearn.decomposition import PCA
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.text import Tokenizer

#폰트 가져오기
font_path = 'C:/Windows/Fonts/EBS훈민정음R.ttf'
fontprop = fm.FontProperties(fname=font_path, size=18)
plt.rc('font', family='NanumGothic') # For Windows

okt=Okt()

def plot_2d_graph(vocabs, xs, ys):
    plt.figure(figsize=(8,6))
    plt.scatter(xs, ys, marker='*')
    for i,v in enumerate(vocabs):
        plt.annotate(v, xy=(xs[i], ys[i]))

train_data = pd.read_csv('Test.txt')

train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# 한글과 공백을 제외하고 모두 제거
train_data['document'].replace('', np.nan, inplace=True)
# 공백은 Null 값으로 변경
train_data = train_data.dropna(how = 'any')

train_X=[]
train_y=[]

#불용어 지정
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
for sentence in train_data['document']:
    temp_x=[]
    temp_x = okt.morphs(sentence, stem=True, norm=True)
    # norm은 그래욬ㅋㅋㅋ -> 그래요
    # stem은 원형으로
    temp_x = [word for word in temp_x if not word in stopwords]
    train_X.append(temp_x)
#print(train_X) ['송민지', '한테', '돈', '보내다'], ['돈', '보내다'], ['돈', '보내다'], ['돈', '이체', '해주다'] ...

train_y=['송금', '잔액', '공인인증서', '거래내역', '이체내역조회']

train_seq=[]

# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거. 0번 패딩 토큰을 고려하여 +1
tokenizer = Tokenizer(19416)
tokenizer.fit_on_texts(train_X)
train_seq = tokenizer.texts_to_sequences(train_X) #토큰을 벡터화 함
train_lab = tokenizer.texts_to_sequences(train_y) #라벨을 벡터화 함 [[9], [2], [], [], []] 왜 이런식으로 나오는지 모르겠음

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 8

train_inputs=[]
test_inputs=[]
train_inputs = pad_sequences(train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', value=0)
lab_inputs = pad_sequences(train_lab, maxlen=1, padding='post')
#print(train_inputs) [17 7 59 11] => [17 7 59 11 0 0 0 0]
#print(train_inputs.shape) (4928, 8)

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
random_seed = 42
test_split = 0.2

y=np.array(train_data['label'])
X_train, X_eval, y_train, y_eval = train_test_split(train_inputs, y, test_size=test_split, random_state=random_seed)

lgs = OneVsRestClassifier(LogisticRegression(class_weight='balanced'))
lgs.fit(X_train, y_train)

tr=[]
a='국민은행계좌로 소은지에게 오백원 송금해줘' # 나중에 text 파일에서 불러올 수 있도록 할 예정
a=okt.morphs(a, stem=True)
temp_w = [word for word in a if not word in stopwords]
tr.append(temp_w)
print(tr)

tokenizer.fit_on_texts(tr)
seq = tokenizer.texts_to_sequences(tr)

#정확도 계산
i=[]
i = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', value=0)
print(lgs.decision_function(i))
print("accu: ", format((int)(lgs.score(X_eval, y_eval)*100)) , "%")

print(lgs.predict(i))
