f = open('ko_wiki_small.txt')
i = 0
while True:
    line = f.readline()
    if line != '\n':
        i += 1
        print("%d번째 줄 :" %i + line)
    if i == 5:
        break
f.close()

from konlpy.tag import Okt
okt = Okt()
n = 0
result = []
with open('ko_wiki_small.txt') as f_r:
    for line in f_r:
        if line:
            n += 1
            if n%5000 == 0: # 5,0000의 배수로 While문이 실행될 때마다
                # 몇번째 while문인지 출력
                print("%d번째 while문."%n)
            tokenlist = okt.pos(line, stem=True, norm=True) #단어토큰화
            temp = []
            for word in tokenlist:
                if word[1] in ["Noun"]: #명사일 때만
                    temp.append((word[0])) #해당 단어를 저장함
                    print(word[0])
            if temp: # 만약 이번에 읽은 데이터에 명사가 존재할 경우에만
                result.append(temp)



