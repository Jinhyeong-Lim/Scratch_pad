import sys
import os
import json
from nltk import ngrams
from collections import defaultdict
from konlpy.tag import Okt
import torch

okt = Okt()
def to_ngrams(words, n):
    ngrams = []
    for a in words:
        for b in range(0, len(a) - n + 1):
            ngrams.append(tuple(words[b:b+n]))
    return ngrams
ngram_counter = defaultdict(int)

directory = os.listdir('C:/Users/default.DESKTOP-6FG4SCS/anaconda3/envs/tensorflow/exa/Wiki/text')
#print(directory)
result=[]
n=0
k=0
for files in directory:
    qw = os.path.join('C:/Users/default.DESKTOP-6FG4SCS/anaconda3/envs/tensorflow/exa/Wiki/text', files)
    directory1 = os.listdir(qw)
    #print(directory1)

    for file in directory1:
        with open(os.path.join(qw, file), 'r' ,encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line)
                            if not line: break  # 모두 읽으면 while문 종료.
                            n = n + 1
                            if n % 5000 == 0:  # 5,000의 배수로 While문이 실행될 때마다 몇 번째 While문 실행인지 출력.
                                print("%d번째 While문." % n)
                            tokenlist = okt.pos(line, stem=True, norm=True)  # 단어 토큰화
                            temp = []
                            for word in tokenlist:
                                if word[1] in ["Noun"]:  # 명사일 때만
                                    temp.append((word[0]))  # 해당 단어를 저장함

                            if temp:  # 만약 이번에 읽은 데이터에 명사가 존재할 경우에만
                                result.append(temp)
                            break    # 결과에 저장
                        k=k+1
                        break
    print(file)
    break


print(result[:100])

from gensim.models import Word2Vec
model = Word2Vec(result, size=100, window=5, min_count=5, workers=4, sg=0)
model_result1=model.wv.most_similar("한국")
model_result3=model.wv.most_similar('오바마')
print(model_result1)
torch.save(model, 'C:/Users/default.DESKTOP-6FG4SCS/anaconda3/envs/pytorch/untitled1/NLP/model.word2vec')


model12 = torch.load('C:/Users/default.DESKTOP-6FG4SCS/anaconda3/envs/pytorch/untitled1/NLP/model.word2vec')
model_result2=model12.wv.most_similar("한국")
model_result4=model12.wv.most_similar('오바마')
print(model_result2)
print(model_result4[:100])
print(model12[:100])

