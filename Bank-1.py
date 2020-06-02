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
font_path = 'C:/Windows/Fonts/EBS훈민정음R.ttf'
fontprop = fm.FontProperties(fname=font_path, size=18)
plt.rc('font', family='NanumGothic') # For Windows
okt=Okt()
from tensorflow.python.keras.preprocessing.text import Tokenizer
def plot_2d_graph(vocabs, xs, ys):
    plt.figure(figsize=(8,6))
    plt.scatter(xs, ys, marker='*')
    for i,v in enumerate(vocabs):
        plt.annotate(v, xy=(xs[i], ys[i]))





train_data = pd.read_csv('Test.txt')
#test_data = pd.read_csv('Test.txt')
print(train_data)
#print(test_data)

train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# 한글과 공백을 제외하고 모두 제거
train_data['document'].replace('', np.nan, inplace=True)
# 공백은 Null 값으로 변경
train_data = train_data.dropna(how = 'any')

train_X=[]
train_y=[]
#test_X=[]
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
for sentence in train_data['document']:
    temp_x=[]
    temp_x = okt.morphs(sentence, stem=True, norm=True)
    # norm은 현대적인말 그래욬ㅋㅋㅋ -> 그래요
    # stem은 그래요 -> 그렇다 원형으로 바꾸어 준다
    temp_x = [word for word in temp_x if not word in stopwords]
    train_X.append(temp_x)

#
train_y=['송금', '잔액', '공인인증서', '거래내역', '이체내역조회']
print(train_X)
train_seq=[]
#test_seq=[]
# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거. 0번 패딩 토큰을 고려하여 +1
tokenizer = Tokenizer(19416)
tokenizer.fit_on_texts(train_X)
train_seq = tokenizer.texts_to_sequences(train_X)
train_lab = tokenizer.texts_to_sequences(train_y)
print(train_seq)
print(train_lab)

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 6

train_inputs=[]
test_inputs=[]
train_inputs = pad_sequences(train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
lab_inputs = pad_sequences(train_lab, maxlen=1, padding='post')
print(lab_inputs)
print(train_inputs.shape)

#test_seq = tokenizer.texts_to_sequences(test_X)

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
a='저번주 이체 내역 조회'
a=okt.morphs(a, stem=True)
temp_w = [word for word in a if not word in stopwords]
tr.append(temp_w)
print(tr)
tokenizer.fit_on_texts(tr)
seq = tokenizer.texts_to_sequences(tr)
i=[]

print(lgs.decision_function(seq))
i = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
print("accu: ", format((int)(lgs.score(X_eval, y_eval)*100)) , "%")



print(lgs.predict(i))


from gensim.models import Word2Vec as w2v
