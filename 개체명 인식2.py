import numpy as np


sentence = '소은지 국민은행계좌에20만원이체해줘'
sentences = '공인인증서재발급'

from konlpy.tag import Komoran
komoran= Komoran()
print(komoran.pos(sentence))
print(komoran.pos(sentences))
print("\n")
#빠른 속도와 보통의 정확도

from konlpy.tag import Hannanum
hannanum= Hannanum()
print(hannanum.pos(sentence))
print(hannanum.pos(sentences))
print("\n")
#빠른 속도와 보통의 정확도

from konlpy.tag import Okt
okt= Okt()
print(okt.pos(sentence, stem=True))
print(okt.pos(sentences, stem=True))
print("\n")
#어느 정도의 띄어쓰기 되어있는 "인터넷"영화평/상품명을 처리할때

from konlpy.tag import Kkma
kkma= Kkma()
print(kkma.pos(sentence))
print(kkma.pos(sentences))
print("\n")
#속도는 느리더라도 정확하고 상세한 품사 정보를 원할때

