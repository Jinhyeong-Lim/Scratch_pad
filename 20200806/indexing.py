line = []
token2idx_dict = {}
with open("C:/Users/default.DESKTOP-6FG4SCS/Desktop/news.txt", encoding='utf-8') as f:
    for line in f:
        for word in line.strip().split():
                if word in token2idx_dict:
                    continue
                else:
                    token2idx_dict[word] = len(token2idx_dict)

    token=input()
    if token in token2idx_dict:
            print("{}: {}".format(token, token2idx_dict[token]))

    print(token2idx_dict.get('doctors', '-1'))

    for key, value in token2idx_dict.items():
            print("{}: {}".format(key, value))