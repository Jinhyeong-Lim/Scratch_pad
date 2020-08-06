line = []
token2idx_dict = {}
with open("C:/Users/default.DESKTOP-6FG4SCS/Desktop/news.txt", encoding='utf-8') as f:
    for line in f:
        for word in line.lower().strip().split():
            if word in token2idx_dict:
                token2idx_dict[word]+=1
            else:
                token2idx_dict[word] = 1

    lw = sorted(token2idx_dict.items(), key=(lambda x: x[1]), reverse=True)
    print(lw[:100])
    print("\n")
    token=input()

    for i in range(100):
        if token in lw[i][0]:
            print(token," : ", i)
            break
        if i==99:
            print(token, " : -1")

    for key, value in lw[:100]:
            print("{}: {}".format(key, value))