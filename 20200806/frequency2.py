import collections
import re
qwe=[]
line=[]

with open("C:/Users/default.DESKTOP-6FG4SCS/Desktop/news.txt", encoding='utf-8') as f:
    token2idx_dict={}

    s=str(f.read()).split()
    qwe = collections.Counter(s)
    lw = sorted(qwe.items(), key=lambda x: x[1], reverse=True)
    for word, cnt in qwe.items():
        print("{}: {}".format(word, cnt))
    print("\n")
    for word, cnt in qwe.most_common(5):
        print("{}: {}".format(word, cnt))
    print("\n")
    for word, cnt in lw:
        if(cnt > 29) : print("{}: {}".format(word, cnt))



