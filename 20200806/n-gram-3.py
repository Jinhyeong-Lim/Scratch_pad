def ngram_overlap(text1, text2, n, pad_left=False, pad_right=False):
    list1=[]
    list2=[]
    list1= text1.split()
    list2=text2.split()
    word1 = []
    word2 = []
    check =[]
    for i in range(len(list1) - n + 1):
        word1.append(list1[i:i + n])
    for i in range(len(list2) - n + 1):
        word2.append(list2[i:i + n])

    x1 = set(word1)
    x2 = set(word2)
    
    return x1&x2

text1 = "json is an open standard file format"
text2 = "xml is another open standard file format"
n_list=[2,3,4]
w = True
for n in n_list:
    print("{}-gram".format(n))
    common = ngram_overlap(text1, text2,n,w)
    print(common)
