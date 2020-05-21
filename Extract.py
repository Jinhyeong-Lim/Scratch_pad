import pandas as pd # 데이터를 원하는데로 가공하는 방법 도와줌
import urllib.request # url작업을 위한 여러 모듈을 모은 패키지
import matplotlib.pyplot as plt
import re  # 정규 표현식 모듈
from konlpy.tag import Okt  #한국어
from gensim.models.word2vec import Word2Vec
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np

okt = Okt()
text = "돈을 보내고 싶어"
crr=[]
brr=[]
stopword=['을', '싶다']


for word in okt.pos(text, stem=True):  # 어간 추출
    if word[0] not in ['을', '싶다']:  # 명사, 동사, 형용사
        brr.append((word[0], word[1]))

print(brr)