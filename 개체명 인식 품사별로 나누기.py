
from konlpy.tag import Komoran
from konlpy.tag import Okt
from ckonlpy.tag import Twitter
okt = Okt()
twitter = Twitter()

sentence = 'IBK기업은행 '
sentences = '소은지국민은행계좌로30만원이체해줘'
komoran = Komoran()


twitter.add_dictionary('이체해줘', 'Noun')
twitter.add_dictionary('KB 국민은행', 'Noun')

komoran = Komoran(userdic="C:/Users/ADMIN/Desktop/dic.txt")


print(twitter.pos(sentence, stem=True))
print(twitter.pos(sentences, stem=True))

print(komoran.pos(sentence))
print(komoran.pos(sentences))

arr = komoran.pos(sentence)
for word, tag in arr :
    if(tag=='VV') : print("|||||||")
    print(word, tag)
    if(tag=='JKO' or tag=='JKB' or tag=='JKS') : print("|||||||")

brr=komoran.pos(sentences)
for word, tag in brr :

    if(tag=='VV' or tag=='XSV') :
        print("|||||||")

    print(word, tag)
    if(tag=='JKO' or tag=='JKB' or tag=='JKS') : print("|||||||")
