def check_ngram_overlap(text1, text2, n, pad_left=False, pad_right=False):
    list1 = []
    list2 = []
    list1 = text1.split()
    list2 = text2.split()
    word1 = []
    word2 = []
    check = []
    for i in range(len(list1) - n + 1):
        word1.append([list1[i:i + n]])
    for i in range(len(list2) - n + 1):
        word2.append([list2[i:i + n]])

    for x in word1:
            if (x in word2): check.append(x)

    if(len(check)) : return True
    else : return False


text1 = "json is an open standard file format"
text2 = "xml is another open standard file format"
print(check_ngram_overlap(text1, text2, 2))
print(check_ngram_overlap(text1, text2, 5))