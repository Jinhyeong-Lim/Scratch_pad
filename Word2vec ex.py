from typing import List, Any

from gensim.models.word2vec import Word2Vec
import re
import pandas as pd
import urllib.request
import requests
import numpy as np
from konlpy.tag import Okt
from tensorflow.python.keras.preprocessing.text import Tokenizer

okt=Okt()
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")

train_data = pd.read_table('ratings_train.txt')

train_data.drop_duplicates(subset=['document'], inplace=True)
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# 한글과 공백을 제외하고 모두 제거
train_data['document'].replace('', np.nan, inplace=True)
# 공백은 Null 값으로 변경
train_data = train_data.dropna(how = 'any')

dat=[]
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
for sentence in train_data['document']:
    temp_x=[]
    temp_x = okt.morphs(sentence, norm=True, stem=True)
    # norm은 현대적인말 그래욬ㅋㅋㅋ -> 그래요
    # stem은 그래요 -> 그렇다 원형으로 바꾸어 준다
    temp_x = [word for word in temp_x if not word in stopwords]
    dat.append(temp_x)

print(dat[:3])

print(dat[1])

model = Word2Vec(dat,         # 리스트 형태의 데이터
                 sg=1,         # 0: CBOW, 1: Skip-gram
                 size=100,     # 벡터 크기
                 window=3,     # 고려할 앞뒤 폭(앞뒤 3단어)
                 min_count=3,  # 사용할 단어의 최소 빈도(3회 이하 단어 무시)
                 workers=4)    # 동시에 처리할 작업 수(코어 수와 비슷하게 설정)

model.save('word2vec.model') #save를 통해 word2vec저장
model = Word2Vec.load(('word2vec.model')) #저장한 모델을 불러온다.
print(model.wv.similarity('영화', '사랑')) # 영화와 사랑 단어 관계
print(model.wv.most_similar('노잼')) # 노잼과 가장 비슷한 부류 단어