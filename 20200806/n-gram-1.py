def ngram(text, n, pad_left=False, pad_right=False):
    list =[]
    list = text.split()
    word=[]
    for i in range(len(list) - n + 1):
        word.append(list[i:i + n])
    return word

text = "i am a boy"
n_list=[2,3,4]

for n in n_list:
    print("{}-gram".format(n))
    for gram in ngram(text, n):
        print(gram)











