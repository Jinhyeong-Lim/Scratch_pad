token2idx_dict = {}
line=[]
with open("C:/Users/default.DESKTOP-6FG4SCS/Desktop/news.txt", encoding='utf-8') as f:
    for line in f:
        for word in line.strip().split():
                if word in token2idx_dict:
                    token2idx_dict[word]+=1
                else:
                    token2idx_dict[word] = 1

print("전체 단어 수: %d"%len(token2idx_dict))
print(token2idx_dict)
print("\n")
word_list = sorted(token2idx_dict.items(), key=lambda x: x[1], reverse=True)
for key, value in word_list:
    print("{}: {}.".format(key, value))
print("\n")
for key, value in word_list[:5]:
    print("{}: {}.".format(key, value))
q=[]
print("\n")
for key, value in word_list:
    if(value > 29) :
        print("{}: {}.".format(key, value))
        break


